"""Tests for the bridge-ladder readout math (run_bridge.py pure helpers).

The experiment injects a thought of step n and scores step n+1 across a LADDER
of rungs (none / full text / true gist KV / bridge(true summary) /
bridge(predicted summary) / shuffled control), all as teacher-forced tail NLL
on the SAME (n, n+1) pairs. These pin the pair-selection and the NLL->PPL->
gap_closed aggregation; the experiment's numbers live in the manifest.
"""

from __future__ import annotations

import math

from marker.run_bridge import ladder_gap_closed, pred_pairs


def test_pred_pairs_needs_history_and_a_next_step():
    # steps 0..L-1; inject step n (n>=1 so a prediction exists), score n+1
    # (n<=L-2 so the next step exists) -> n in [1, L-2]
    assert pred_pairs(5) == [1, 2, 3]
    assert pred_pairs(3) == [1]
    assert pred_pairs(2) == []  # no n with both history and a next step
    assert pred_pairs(1) == []


def test_ladder_gap_closed_anchors_none_zero_full_one():
    # gap_closed is defined so none->0.0 and full->1.0 by construction
    nlls = {
        "none": [math.log(10.0)] * 4,  # ppl 10
        "full": [math.log(2.0)] * 4,  # ppl 2
        "gist_true": [math.log(3.0)] * 4,  # ppl 3
    }
    out = ladder_gap_closed(nlls)
    assert out["none"]["gap_closed"] == 0.0
    assert out["full"]["gap_closed"] == 1.0
    # gist_true closes (10-3)/(10-2) = 0.875 of the gap
    assert abs(out["gist_true"]["gap_closed"] - 0.875) < 1e-6
    assert abs(out["gist_true"]["ppl"] - 3.0) < 1e-6


def test_ladder_gap_closed_worse_than_nothing_is_negative():
    # a misleading (shuffled) thought raises PPL above none -> gap_closed < 0
    nlls = {
        "none": [math.log(10.0)],
        "full": [math.log(2.0)],
        "shuffled": [math.log(20.0)],  # ppl 20 > none
    }
    out = ladder_gap_closed(nlls)
    assert out["shuffled"]["gap_closed"] < 0


def test_ladder_gap_closed_skips_empty_and_degenerate():
    # a rung with no pairs is reported as None, not a crash; a zero none->full
    # gap (no headroom) clamps gap_closed to 0.0 rather than dividing by zero
    nlls = {
        "none": [math.log(5.0)],
        "full": [math.log(5.0)],  # no headroom
        "gist_true": [math.log(4.0)],
        "empty": [],
    }
    out = ladder_gap_closed(nlls)
    assert out["gist_true"]["gap_closed"] == 0.0
    assert out["empty"]["ppl"] is None
    assert out["empty"]["gap_closed"] is None
