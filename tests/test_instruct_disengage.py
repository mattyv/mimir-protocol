"""Model-free invariants for the strict-vs-sticky disengagement eval.

Amended per Fable's review of the first two runs: drop the persistent/stance
arms (near-parity, not decisive), test STRICT-gated (KV only on turns naming
the term) vs STICKY-gated(K=2) (KV persists K turns after the last mention)
vs sticky+stance. Sessions are shaped engage -> offtopic -> followup
(term-less) -> offtopic to expose the real tradeoff: strict may miss the
returning follow-up; sticky may bleed during the offtopic turn inside its
window. Scoring is symmetric (the same api-pattern check drives both
engagement and bleed) and engage/followup turns carry a syntax-fidelity gold
in addition to the loose API gold, since a re-engaged-from-history answer can
match the loose gold while getting the DSL's syntax wrong.
"""

from __future__ import annotations

from marker.instruct import (
    IM_END,
    IM_START,
    chat_multiturn_suffix,
    chat_system_open,
)
from marker.run_instruct_disengage import (
    SESSIONS,
    SKILLS,
    STANCE,
    STICKY_K,
    active_terms_for_turn,
    build_system_body,
    score_turn,
)

# ── Multi-turn chat layout (unchanged machinery) ────────────────────────────────


def test_multiturn_suffix_closes_system_replays_history_and_adds_turn():
    s = chat_multiturn_suffix([("user", "hi"), ("assistant", "hello")], "next?")
    assert s.startswith(f"{IM_END}\n")
    assert f"{IM_START}user\nhi{IM_END}" in s
    assert f"{IM_START}assistant\nhello{IM_END}" in s
    assert s.rstrip().endswith(f"{IM_START}assistant")
    assert s.index("next?") > s.index("hello")


def test_system_open_has_sink_and_is_left_open():
    p = chat_system_open("About ilp_for: stuff")
    assert p.startswith(f"{IM_START}system\n")
    assert IM_END not in p


# ── Sticky activation: distance-since-last-mention (no decrementing counter,
# so there is no off-by-one between "K turns of grace" and "K decrements") ──────


def test_strict_gated_only_current_turn_terms():
    active, last_seen = active_terms_for_turn(
        "strict-gated", session_skills=["A", "B"], last_seen={"A": 0}, turn_idx=3, current={"B"}
    )
    assert active == ["B"]  # A's leftover sticky state (if any) is irrelevant to strict


def test_sticky_gated_active_on_mention_turn():
    active, last_seen = active_terms_for_turn(
        "sticky-gated-k2", session_skills=["A"], last_seen={}, turn_idx=0, current={"A"}
    )
    assert active == ["A"] and last_seen["A"] == 0


def test_sticky_gated_survives_k_silent_turns_after_mention():
    # Mentioned at turn 0 with K=2: must stay active through turns 1 AND 2
    # (distance 1 and 2, both <= K), evicted only at turn 3 (distance 3 > K).
    last_seen = {"A": 0}
    for turn_idx in (1, 2):
        active, last_seen = active_terms_for_turn(
            "sticky-gated-k2",
            session_skills=["A"],
            last_seen=last_seen,
            turn_idx=turn_idx,
            current=set(),
        )
        assert active == ["A"], f"turn {turn_idx}: expected still active within K={STICKY_K}"

    active, last_seen = active_terms_for_turn(
        "sticky-gated-k2", session_skills=["A"], last_seen=last_seen, turn_idx=3, current=set()
    )
    assert active == []  # distance 3 > K=2 -> evicted


def test_re_mention_resets_the_distance():
    last_seen = {"A": 0}
    active, last_seen = active_terms_for_turn(
        "sticky-gated-k2", session_skills=["A"], last_seen=last_seen, turn_idx=2, current={"A"}
    )
    assert active == ["A"] and last_seen["A"] == 2  # distance reset to 0 as of turn 2
    active2, _ = active_terms_for_turn(
        "sticky-gated-k2", session_skills=["A"], last_seen=last_seen, turn_idx=4, current=set()
    )
    assert active2 == ["A"]  # distance from the re-mention (2) is only 2, still <= K


def test_sticky_with_stance_uses_the_same_active_set_as_sticky():
    a1, ls1 = active_terms_for_turn(
        "sticky-gated-k2", session_skills=["A"], last_seen={}, turn_idx=0, current={"A"}
    )
    a2, ls2 = active_terms_for_turn(
        "sticky-gated-k2-stance", session_skills=["A"], last_seen={}, turn_idx=0, current={"A"}
    )
    assert a1 == a2 and ls1 == ls2


# ── System body ──────────────────────────────────────────────────────────────────


def test_build_body_empty_is_bare_assistant():
    b = build_system_body([], stance=False)
    assert "About" not in b and STANCE not in b


def test_build_body_stance_only_added_when_requested():
    plain = build_system_body(["ilp_for"], stance=False)
    staged = build_system_body(["ilp_for"], stance=True)
    assert STANCE not in plain
    assert STANCE in staged
    assert "About ilp_for:" in plain and "About ilp_for:" in staged


# ── Symmetric scoring (same instrument both directions) ─────────────────────────


def test_score_offtopic_clean_when_no_skill_api_present():
    t = {"kind": "offtopic", "term": None, "gold": None, "fidelity": None}
    ok, tag, matched = score_turn("def reverse(s): return s[::-1]", t, ["ilp_for"], SKILLS)
    assert ok and matched == set()


def test_score_offtopic_bleeds_when_target_api_present():
    t = {"kind": "offtopic", "term": None, "gold": None, "fidelity": None}
    out = "ILP_FOR_AUTO(auto i, 0, n, Sum, int) {} ILP_END;"
    ok, tag, matched = score_turn(out, t, ["ilp_for"], SKILLS)
    assert not ok and matched == {"ilp_for"}
    assert "BLED" in tag


def test_score_engage_requires_gold_fidelity_and_api_match():
    t = {
        "kind": "engage",
        "term": "ilp_for",
        "gold": "ILP_FOR",
        "fidelity": "ILP_END",
    }
    good = "ILP_FOR_AUTO(auto i, 0, n, Sum, double) { total += data[i]; } ILP_END;"
    ok, tag, matched = score_turn(good, t, ["ilp_for"], SKILLS)
    assert ok and matched == {"ilp_for"}

    no_terminator = "ILP_FOR_AUTO(auto i, 0, n, Sum, double) { total += data[i]; }"
    ok2, tag2, _ = score_turn(no_terminator, t, ["ilp_for"], SKILLS)
    assert not ok2  # gold present but fidelity (terminator) missing


def test_score_engage_fails_on_misroute():
    t = {"kind": "engage", "term": "ilp_for", "gold": "ILP_FOR", "fidelity": None}
    out = "ILP_FOR_AUTO(auto i, 0, n, Sum, int) {} ILP_END; client.emit('x', y, ttl=30)"
    ok, tag, matched = score_turn(out, t, ["ilp_for", "InternalBus"], SKILLS)
    assert not ok  # InternalBus API leaked on an ilp_for-targeted turn
    assert "InternalBus" in matched


# ── Session data hygiene ────────────────────────────────────────────────────────


def test_every_session_skill_registered():
    for sess in SESSIONS:
        for term in sess["skills"]:
            assert term in SKILLS, f"{sess['name']}: unknown skill {term}"


def test_sessions_have_the_gap_shape():
    # engage -> ... -> a term-less followup -> offtopic, with at least one
    # intervening turn that doesn't name the follow-up's term (a distraction —
    # either an off-topic request, or (interleaved_gap) engaging a different
    # skill; both are distractions from the follow-up target's point of view).
    for sess in SESSIONS:
        turns = sess["turns"]
        kinds = [t["kind"] for t in turns]
        assert kinds[0] == "engage", sess["name"]
        assert "followup" in kinds, sess["name"]
        fu_idx = kinds.index("followup")
        target = turns[fu_idx]["term"]
        distractions = [t for t in turns[1:fu_idx] if t["term"] != target]
        assert distractions, f"{sess['name']}: no distraction before followup"
        assert kinds[-1] == "offtopic", sess["name"]


def test_followup_turns_are_term_less():
    for sess in SESSIONS:
        for t in sess["turns"]:
            if t["kind"] == "followup":
                assert t["term"] is not None  # scoring target is still declared
                assert t["term"].lower() not in t["q"].lower(), (
                    f"{sess['name']}: followup must not name the term"
                )


def test_engage_turns_name_their_term():
    for sess in SESSIONS:
        for t in sess["turns"]:
            if t["kind"] == "engage":
                assert t["term"].lower() in t["q"].lower(), sess["name"]


def test_offtopic_turns_name_no_session_skill():
    for sess in SESSIONS:
        for t in sess["turns"]:
            if t["kind"] == "offtopic":
                assert t["term"] is None and t["gold"] is None
                for term in sess["skills"]:
                    assert term.lower() not in t["q"].lower(), (
                        f"{sess['name']}: offtopic names {term}"
                    )


def test_there_is_a_cross_skill_session():
    assert any(len(s["skills"]) >= 2 for s in SESSIONS), "need a routing/gap combo session"
