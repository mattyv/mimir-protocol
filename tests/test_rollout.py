"""Tests for the latent chain rollout (rollout.py).

Pin the mechanical invariants: the free-running chain feeds its own predictions
back, its first step equals teacher-forcing (all-real history), it respects the
window, and the drift aggregator anchors none/full correctly. The drift NUMBERS
(does it hold or spiral) live in the run's manifest, not here.
"""

from __future__ import annotations

import math

import torch

from marker.predictor import NextThoughtPredictor
from marker.rollout import drift_by_depth, rollout, teacher_forced
from marker.run_bridge import predict_step


def _pred():
    torch.manual_seed(0)
    m = NextThoughtPredictor(d=8, k=4, d_model=16, layers=1, heads=2, max_sents=16)
    return m.eval()


def test_rollout_shapes_and_determinism():
    m = _pred()
    prefix = torch.randn(3, 4, 8)
    a = rollout(m, prefix, depth=5, window=6)
    b = rollout(m, prefix, depth=5, window=6)
    assert a.shape == (5, 4, 8)
    assert torch.equal(a, b)  # eval mode, no_grad -> deterministic


def test_rollout_first_step_equals_teacher_forced():
    # depth-1 rollout predicts step P from the ALL-REAL prefix, which is exactly
    # predict_step at position P on a summ whose first P rows are the prefix
    m = _pred()
    summ = torch.randn(8, 4, 8)
    prefix = summ[:4]
    roll0 = rollout(m, prefix, depth=1, window=6)[0]
    ps = predict_step(m, summ, n=4, window=6)  # position 4, history = summ[:4]
    assert torch.allclose(roll0, ps, atol=1e-6)


def test_rollout_feeds_predictions_back():
    # step 2 of the chain must depend on step 1's PREDICTION: perturbing the
    # predictor's step-1 output (by changing the prefix it derives from) changes
    # step 2. Concretely: rollout depth-2 step-2 differs from a teacher-forced
    # step-2 that used the TRUE thought at P instead of the predicted one.
    m = _pred()
    summ = torch.randn(8, 4, 8)
    prefix = summ[:4]
    roll = rollout(m, prefix, depth=2, window=6)  # [2,4,8]
    # teacher-forced step P+1 uses the TRUE thought at P (summ[4]); free-running
    # used its own prediction there -> the two step-2 outputs must differ
    tf_step2 = predict_step(m, summ, n=5, window=6)
    assert not torch.allclose(roll[1], tf_step2, atol=1e-4)


def test_rollout_respects_window():
    # with window=2, predicting step t sees only thought t-1 (+ masked dummy);
    # changing an earlier prefix thought must NOT change the first prediction
    m = _pred()
    prefix = torch.randn(5, 4, 8)
    r1 = rollout(m, prefix, depth=1, window=2)
    p2 = prefix.clone()
    p2[0] = torch.randn(4, 8)  # far outside the window-2 tail
    r2 = rollout(m, p2, depth=1, window=2)
    assert torch.allclose(r1, r2, atol=1e-6)


def test_teacher_forced_stops_at_doc_end():
    m = _pred()
    summ = torch.randn(6, 4, 8)  # only 6 steps
    tf = teacher_forced(m, summ, prefix_len=4, depth=10, window=6)
    assert tf.shape[0] == 2  # steps 4, 5 only (positions 6+ don't exist)


def test_drift_by_depth_anchors_and_cosines():
    by_depth = {
        1: {
            "none": [math.log(10.0)],
            "full": [math.log(2.0)],
            "free": [math.log(3.0)],
            "free_cos": [0.9],
            "tf_cos": [0.95],
        },
        2: {
            "none": [math.log(10.0)],
            "full": [math.log(2.0)],
            "free": [math.log(8.0)],  # drifted: closer to none
            "free_cos": [0.5],
            "tf_cos": [0.9],
        },
    }
    out = drift_by_depth(by_depth)
    assert out[1]["none"] == 0.0 and out[1]["full"] == 1.0
    assert abs(out[1]["free"] - 0.875) < 1e-6  # (10-3)/(10-2)
    assert out[2]["free"] < out[1]["free"]  # drift: gap_closed falls with depth
    assert out[1]["free_cos"] == 0.9 and out[2]["free_cos"] == 0.5
    assert out[1]["n"] == 1
