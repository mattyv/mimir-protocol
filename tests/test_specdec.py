"""Model-free invariants for the speculative draft-and-verify decoder.

The core of spec decode is pure accounting: given the drafter's proposed
tokens and the verifier's per-position argmax picks (from ONE parallel pass),
how many draft tokens are accepted, which token arrives free, and how the
totals conserve. These tests pin that logic without loading any model.

The model-level guarantee (output byte-identical to the verifier's own greedy
decode) is exercised by the runner's smoke mode, where drafter == verifier
makes 100% acceptance mathematically expected and identity a hard assert.
"""

from __future__ import annotations

from marker.specdec import accept_prefix, trim_at_eos

# ── accept_prefix: (draft, picks) -> (n_accepted, free_token) ───────────────────
# picks has len(draft)+1 entries: the verifier's argmax at each draft position,
# plus its prediction after consuming the full draft (the all-accepted bonus).


def test_divergence_mid_draft():
    n, free = accept_prefix([5, 6, 7, 8], [5, 6, 9, 8, 11])
    assert n == 2  # draft[2]=7 != picks[2]=9
    assert free == 9  # the verifier's own token at the divergence, free


def test_divergence_at_first_token():
    n, free = accept_prefix([5, 6], [4, 6, 7])
    assert n == 0
    assert free == 4  # even a total miss yields one correct token


def test_full_acceptance_yields_bonus_token():
    n, free = accept_prefix([5, 6, 7], [5, 6, 7, 8])
    assert n == 3
    assert free == 8  # prediction after the whole draft — also from the same pass


def test_empty_draft_still_yields_one_token():
    n, free = accept_prefix([], [42])
    assert n == 0
    assert free == 42  # degenerate case = vanilla decode of one token


def test_tokens_gained_per_round_is_accepted_plus_one():
    # The conservation law behind "every pass nets >= 1 token".
    for draft, picks in [
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3], [9, 2, 3, 4]),
        ([7], [7, 8]),
        ([7], [9, 8]),
    ]:
        n, _ = accept_prefix(draft, picks)
        gained = n + 1
        assert 1 <= gained <= len(draft) + 1


# ── trim_at_eos: sequence hygiene at the stop token ─────────────────────────────


def test_trim_stops_at_eos_inclusive():
    assert trim_at_eos([5, 6, 99, 7, 8], eos_id=99) == [5, 6, 99]


def test_trim_no_eos_returns_all():
    assert trim_at_eos([5, 6, 7], eos_id=99) == [5, 6, 7]


def test_trim_eos_first():
    assert trim_at_eos([99, 1, 2], eos_id=99) == [99]


def test_trim_none_eos_returns_all():
    assert trim_at_eos([5, 6, 7], eos_id=None) == [5, 6, 7]
