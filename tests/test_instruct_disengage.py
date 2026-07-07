"""Model-free invariants for the multi-turn skill-disengagement eval.

The failure this eval targets: once a skill term appears in a chat session, its
injected KV persists and the model over-applies the DSL on later, off-topic
turns (the bleed we measured single-turn as NEGATIVE 0/2). These tests pin the
multi-turn chat layout, the session shape (engage / follow-up / off-topic), and
the per-mechanism system-text logic — no model needed.
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
    STANCE,
    disengage_system_text,
)

# ── Multi-turn chat layout ──────────────────────────────────────────────────────


def test_multiturn_suffix_closes_system_replays_history_and_adds_turn():
    s = chat_multiturn_suffix([("user", "hi"), ("assistant", "hello")], "next?")
    assert s.startswith(f"{IM_END}\n")  # closes the open system block first
    assert f"{IM_START}user\nhi{IM_END}" in s
    assert f"{IM_START}assistant\nhello{IM_END}" in s
    # current turn is last, ending on the assistant generation prompt
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


# ── Per-mechanism system text ───────────────────────────────────────────────────


def test_persistent_always_has_skill():
    on = disengage_system_text("persistent", "ilp_for", "DESC", term_present=True)
    off = disengage_system_text("persistent", "ilp_for", "DESC", term_present=False)
    assert "DESC" in on and "DESC" in off  # skill present regardless of the turn


def test_term_gated_drops_skill_when_term_absent():
    on = disengage_system_text("term-gated", "ilp_for", "DESC", term_present=True)
    off = disengage_system_text("term-gated", "ilp_for", "DESC", term_present=False)
    assert "DESC" in on
    assert "DESC" not in off  # skill removed on a no-term turn


def test_stance_keeps_skill_and_adds_the_clause():
    on = disengage_system_text("stance", "ilp_for", "DESC", term_present=True)
    off = disengage_system_text("stance", "ilp_for", "DESC", term_present=False)
    for body in (on, off):
        assert "DESC" in body  # skill always present (like persistent)
        assert STANCE in body  # + the "only when asked" clause


# ── Session data hygiene ────────────────────────────────────────────────────────


def test_sessions_have_engage_followup_and_offtopic_turns():
    for sess in SESSIONS:
        turns = sess["turns"]
        kinds = [t["kind"] for t in turns]
        assert "engage" in kinds and "followup" in kinds and "offtopic" in kinds
        # engage turn names the term; followup and offtopic do NOT.
        term = sess["term"].lower()
        for t in turns:
            named = term in t["q"].lower()
            if t["kind"] == "engage":
                assert named, f"{sess['term']}: engage turn must name the term"
            else:
                assert not named, f"{sess['term']}: {t['kind']} turn must not name the term"


def test_engage_and_followup_want_the_dsl_offtopic_forbids_it():
    for sess in SESSIONS:
        for t in sess["turns"]:
            if t["kind"] in ("engage", "followup"):
                assert t["gold"] is not None, "engage/followup must have a DSL gold"
            else:
                assert t["gold"] is None, "offtopic must be a no-bleed (gold=None) turn"
