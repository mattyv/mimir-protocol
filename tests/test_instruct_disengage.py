"""Model-free invariants for the multi-turn skill-disengagement eval.

The failure this eval targets: once a skill term appears in a chat session, its
injected KV persists and the model over-applies the DSL on later, off-topic
turns. These tests pin the multi-turn chat layout, the (now multi-skill)
session shape, and the per-mechanism system-body logic — no model needed.
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
    active_terms_for_turn,
    build_system_body,
)

# ── Multi-turn chat layout ──────────────────────────────────────────────────────


def test_multiturn_suffix_closes_system_replays_history_and_adds_turn():
    s = chat_multiturn_suffix([("user", "hi"), ("assistant", "hello")], "next?")
    assert s.startswith(f"{IM_END}\n")  # closes the open system block first
    assert f"{IM_START}user\nhi{IM_END}" in s
    assert f"{IM_START}assistant\nhello{IM_END}" in s
    assert s.rstrip().endswith(f"{IM_START}assistant")
    assert s.index("next?") > s.index("hello")  # current turn after history


def test_multiturn_suffix_empty_history_is_a_single_turn():
    s = chat_multiturn_suffix([], "only question?")
    assert s.startswith(f"{IM_END}\n{IM_START}user\nonly question?{IM_END}")
    assert s.rstrip().endswith(f"{IM_START}assistant")


def test_system_open_has_sink_and_is_left_open():
    p = chat_system_open("About ilp_for: stuff")
    assert p.startswith(f"{IM_START}system\n")
    assert "About ilp_for: stuff" in p
    assert IM_END not in p


# ── System body construction ────────────────────────────────────────────────────


def test_build_body_empty_is_bare_assistant():
    b = build_system_body([], stance=False)
    assert "About" not in b
    assert STANCE not in b


def test_build_body_includes_each_active_skill_desc():
    b = build_system_body(["ilp_for", "InternalBus"], stance=False)
    assert "About ilp_for:" in b and "About InternalBus:" in b
    assert STANCE not in b


def test_build_body_stance_appends_clause():
    b = build_system_body(["ilp_for"], stance=True)
    assert "About ilp_for:" in b
    assert STANCE in b


# ── Per-mechanism active-term selection ─────────────────────────────────────────


def test_persistent_and_stance_use_all_seen_terms():
    for mech in ("persistent", "stance"):
        got = active_terms_for_turn(mech, session_skills=["A", "B"], seen={"A"}, current=set())
        assert got == ["A"]  # seen so far, even though not named this turn


def test_term_gated_uses_only_current_turn_terms():
    got = active_terms_for_turn(
        "term-gated", session_skills=["A", "B"], seen={"A", "B"}, current={"B"}
    )
    assert got == ["B"]  # only what the current turn names
    off = active_terms_for_turn(
        "term-gated", session_skills=["A", "B"], seen={"A", "B"}, current=set()
    )
    assert off == []  # no term this turn -> bare system


# ── Session data hygiene ────────────────────────────────────────────────────────


def test_every_session_skill_is_registered_with_an_api_re():
    for sess in SESSIONS:
        for term in sess["skills"]:
            assert term in SKILLS, f"{sess['name']}: unknown skill {term}"
            assert SKILLS[term]["api_re"] is not None


def test_sessions_have_engage_and_offtopic_turns():
    for sess in SESSIONS:
        kinds = {t["kind"] for t in sess["turns"]}
        assert "engage" in kinds and "offtopic" in kinds, sess["name"]


def test_engage_followup_name_their_target_offtopic_names_nothing():
    for sess in SESSIONS:
        for t in sess["turns"]:
            if t["kind"] in ("engage", "followup"):
                assert t["term"] in sess["skills"], f"{sess['name']}: bad target"
                assert t["gold"] is not None
                if t["kind"] == "engage":
                    # an engage turn must name its target skill
                    assert t["term"].lower() in t["q"].lower(), sess["name"]
            else:  # offtopic
                assert t["term"] is None and t["gold"] is None
                # off-topic turns must not name ANY of the session's skills
                for term in sess["skills"]:
                    assert term.lower() not in t["q"].lower(), (
                        f"{sess['name']}: offtopic names {term}"
                    )


def test_there_is_an_interleaved_two_skill_session():
    assert any(len(s["skills"]) >= 2 for s in SESSIONS), "need a cross-skill routing session"
