"""Tests for the Stage-3b draft-and-verify harness (draft_verify.py).

Model-free: the advance-vs-restate rate (Fable's forward-motion gate — chaining
needs drafts that MOVE to the next step, not restate the current one) and the
verify selection (pick the draft the big model likes best on the real context).
"""

from __future__ import annotations

from marker.draft_verify import advance_rate, pick_by_score


def _f1(a, b):
    from collections import Counter

    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    o = sum((ca & cb).values())
    return 0.0 if o == 0 else 2 * o / (len(a) + len(b)) * (o / max(o, 1))  # rough


def test_advance_rate_counts_forward_drafts():
    # draft closer to NEXT than CURRENT counts as advancing
    cur = [[1, 2, 3], [1, 2, 3]]
    nxt = [[4, 5, 6], [4, 5, 6]]
    drafts = [[4, 5, 6], [1, 2, 3]]  # first advances (==next), second restates (==cur)
    from collections import Counter

    def sim(a, b):
        ca, cb = Counter(a), Counter(b)
        return sum((ca & cb).values())

    assert advance_rate(drafts, cur, nxt, sim) == 0.5


def test_advance_rate_all_forward_and_all_restate():
    cur, nxt = [[1, 1]], [[2, 2]]
    from collections import Counter

    sim = lambda a, b: sum((Counter(a) & Counter(b)).values())  # noqa: E731
    assert advance_rate([[2, 2]], cur, nxt, sim) == 1.0
    assert advance_rate([[1, 1]], cur, nxt, sim) == 0.0


def test_pick_by_score_returns_best_and_index():
    cands = [["a"], ["b"], ["c"]]
    # lower score = better (e.g. NLL); pick 'b'
    best, idx = pick_by_score(cands, [2.0, 0.5, 1.0])
    assert best == ["b"] and idx == 1


def test_pick_by_score_ties_take_first():
    best, idx = pick_by_score([["a"], ["b"]], [1.0, 1.0])
    assert idx == 0
