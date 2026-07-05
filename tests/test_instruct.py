"""Model-free invariants for the instruct Phase-1 harness."""

from __future__ import annotations

from marker.instruct import (
    IM_END,
    IM_START,
    chat_live_suffix,
    chat_system_prefix,
    declined,
    injected_position_ranges,
    matches,
)
from marker.run_instruct_phase1 import BOUNDARY, PREAMBLE, USE_AXIOMS, _heldout_probes
from marker.run_prefix_tuned import TUNED_AXIOMS

# ── Chat layout ───────────────────────────────────────────────────────────────


def test_system_prefix_has_sink_and_description():
    p = chat_system_prefix("BalancePublisher", "polls every 250 ms")
    assert p.startswith(f"{IM_START}system\n")  # attention sink genuinely first
    assert "About BalancePublisher:" in p
    assert "polls every 250 ms" in p
    assert IM_END not in p  # system block left open for the live tokens to continue


def test_live_suffix_with_preamble_closes_block_and_adds_user_turn():
    s = chat_live_suffix("How fast?", "Answer confidently.")
    assert "Answer confidently." in s
    assert f"{IM_END}\n{IM_START}user\nHow fast?{IM_END}" in s
    assert s.endswith(f"{IM_START}assistant\n")
    # preamble precedes the close of the system block
    assert s.index("Answer confidently.") < s.index(IM_END)


def test_live_suffix_no_preamble_starts_by_closing_system():
    s = chat_live_suffix("How fast?", None)
    assert s.startswith(f"{IM_END}\n{IM_START}user\n")
    assert "How fast?" in s


def test_prefix_plus_suffix_is_valid_single_conversation():
    # Concatenated they must form one coherent system+user+assistant sequence.
    full = chat_system_prefix("X", "desc here") + chat_live_suffix("q?", PREAMBLE)
    assert full.count(f"{IM_START}system") == 1  # exactly one system block
    assert f"{IM_START}user" in full
    assert full.rstrip().endswith(f"{IM_START}assistant")


# ── Positional invariant (Phase 0) ─────────────────────────────────────────────


def test_position_ranges_non_overlapping_and_monotone():
    kv_r, live_r = injected_position_ranges(12, 20)
    assert kv_r == range(0, 12)
    assert live_r == range(12, 32)
    assert set(kv_r).isdisjoint(set(live_r))
    assert kv_r.stop == live_r.start  # contiguous, no gap, no overlap


def test_position_ranges_zero_live():
    kv_r, live_r = injected_position_ranges(5, 0)
    assert list(live_r) == []
    assert kv_r.stop == 5


# ── Scoring helpers ─────────────────────────────────────────────────────────────


def test_matches_digit_boundary():
    assert matches("it is 250 milliseconds", "250 milli")
    assert not matches("value 1000", "100")
    assert matches("value 100.", "100")


def test_declined_detects_refusal_phrasings():
    assert declined("The description doesn't specify the language.")
    assert declined("That information is not mentioned in the material.")
    assert declined("I'm unable to determine that from the description.")


def test_declined_false_on_confident_answer():
    assert not declined("BalancePublisher is written in Java.")
    assert not declined("It uses 512 MB of memory.")


# ── Data hygiene ────────────────────────────────────────────────────────────────


def test_used_axioms_exist_and_have_boundary_probes():
    names = {a["name"] for a in TUNED_AXIOMS}
    for n in USE_AXIOMS:
        assert n in names, f"{n} not in TUNED_AXIOMS"
        assert n in BOUNDARY and len(BOUNDARY[n]) >= 2, f"{n} missing boundary probes"


def test_heldout_golds_are_answerable_from_fact_text():
    by_name = {a["name"]: a for a in TUNED_AXIOMS}
    for n in USE_AXIOMS:
        axiom = by_name[n]
        ftext = axiom["fact_text"].lower()
        for _q, gold in _heldout_probes(axiom):
            assert gold.lower() in ftext, f"{n}: heldout gold {gold!r} absent from fact_text"


def test_boundary_questions_are_out_of_scope():
    # A boundary question must NOT have its topic covered by the fact_text, or
    # "decline" would be the wrong answer. Spot-check by keyword.
    by_name = {a["name"]: a for a in TUNED_AXIOMS}
    scope_keywords = {
        "BalancePublisher": ["language", "memory"],
        "FluxomService": ["cloud provider", "engineers"],
        "MeshPublisher": ["port", "deployed"],
    }
    for n in USE_AXIOMS:
        ftext = by_name[n]["fact_text"].lower()
        for kw in scope_keywords[n]:
            assert kw not in ftext, f"{n}: boundary keyword {kw!r} unexpectedly in fact_text"


def test_preamble_has_boundary_clause():
    # The load-bearing anti-hallucination sentence must be present.
    assert "doesn't specify" in PREAMBLE.lower() or "does not cover" in PREAMBLE.lower()
