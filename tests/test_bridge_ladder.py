"""Tests for the bridge-ladder readout math (run_bridge.py pure helpers).

The experiment injects a thought of step n and scores step n+1 across a LADDER
of rungs (none / full text / true gist KV / bridge(true summary) /
bridge(predicted summary) / shuffled control), all as teacher-forced tail NLL
on the SAME (n, n+1) pairs. These pin the pair-selection and the NLL->PPL->
gap_closed aggregation; the experiment's numbers live in the manifest.
"""

from __future__ import annotations

import math

import torch

from marker.predictor import NextThoughtPredictor
from marker.run_bridge import ladder_gap_closed, noised, pred_pairs, predict_step


def test_pred_pairs_needs_history_and_a_next_step():
    # steps 0..L-1; inject step n (n>=1 so a prediction exists), score n+1
    # (n<=L-2 so the next step exists) -> n in [1, L-2]
    assert pred_pairs(5) == [1, 2, 3]
    assert pred_pairs(3) == [1]
    assert pred_pairs(2) == []  # no n with both history and a next step
    assert pred_pairs(1) == []


def test_predict_step_windowed_positions_and_causality():
    # the predictor's position table is only trained for window-local indices;
    # predict_step must (a) clamp the input to the last <=window steps so
    # positions stay in-distribution, and (b) be causally blind to step n's own
    # summary even though it's included as the masked target position.
    torch.manual_seed(0)
    m = NextThoughtPredictor(d=8, k=4, d_model=16, layers=1, heads=2, max_sents=8).to("cpu")
    m.eval()
    summ = torch.randn(20, 4, 8)  # 20 steps >> window
    with torch.no_grad():
        # deep-in-doc step: would index position 18 unwindowed (max_sents=8 -> crash)
        p = predict_step(m, summ, n=18, window=8)
        assert p.shape == (4, 8)
        # (b) changing step n's own summary must NOT change the prediction of n
        summ2 = summ.clone()
        summ2[18] = torch.randn(4, 8)
        p2 = predict_step(m, summ2, n=18, window=8)
        assert torch.allclose(p, p2, atol=1e-6)
        # changing a step INSIDE the history window must change it
        summ3 = summ.clone()
        summ3[17] = torch.randn(4, 8)
        p3 = predict_step(m, summ3, n=18, window=8)
        assert not torch.allclose(p, p3, atol=1e-4)
        # changing a step OUTSIDE the window must not change it
        summ4 = summ.clone()
        summ4[2] = torch.randn(4, 8)
        p4 = predict_step(m, summ4, n=18, window=8)
        assert torch.allclose(p, p4, atol=1e-6)


def test_noised_hits_target_cosine_band():
    # anti-hashing jitter: noised(g, ratio) adds per-slot noise with norm =
    # ratio * ||slot||. ratio 1.0 -> expected cosine ~1/sqrt(2) ~ 0.71 to the
    # clean summary (the predictor's error distance); ratio 0 -> identity.
    torch.manual_seed(0)
    g = torch.randn(8, 512)
    z = noised(g, 1.0, torch.Generator().manual_seed(1))
    cos = torch.nn.functional.cosine_similarity(g, z, dim=-1)
    assert 0.6 < cos.mean() < 0.8, f"ratio 1.0 should land near cos 0.71, got {cos.mean():.3f}"
    z0 = noised(g, 0.0, torch.Generator().manual_seed(1))
    assert torch.equal(z0, g)


def test_noised_preserves_shape_and_grad_free_input():
    g = torch.randn(3, 4, 16)
    z = noised(g, 0.5, torch.Generator().manual_seed(0))
    assert z.shape == g.shape
    cos = torch.nn.functional.cosine_similarity(g, z, dim=-1)
    assert (cos > 0.8).all()  # ratio 0.5 -> cos ~ 0.89, comfortably above 0.8


def test_predict_step_earliest_valid_n():
    # n=1: history is just step 0 -> input [step0, step1(masked)], one readout
    torch.manual_seed(1)
    m = NextThoughtPredictor(d=8, k=4, d_model=16, layers=1, heads=2, max_sents=8)
    m.eval()
    summ = torch.randn(3, 4, 8)
    with torch.no_grad():
        p = predict_step(m, summ, n=1, window=8)
    assert p.shape == (4, 8)


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
