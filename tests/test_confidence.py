"""Tests for the fast-lane confidence probe (confidence.py).

The probe asks: does an INFERENCE-TIME confidence signal separate the
predictor's correct next-thought guesses from its wrong ones? (FASTLANE_PLAN
"cheap-first probe".) These pin the mechanics — AUC math, per-example
within-doc correctness, the confidence signals, tercile bucketing — not the
experiment's numeric outcome (which lives in the probe's manifest).
"""

from __future__ import annotations

import torch

from marker.confidence import (
    coverage_curve,
    dropout_agreement,
    mc_dropout_pool,
    precision_at_coverage,
    prediction_norm,
    rank_auc,
    retrieval_margin,
    slot_cosine,
    tercile_report,
    within_doc_correct,
)
from marker.predictor import NextThoughtPredictor

# ── AUC (model-free) ─────────────────────────────────────────────────────────


def test_auc_perfect_separation():
    # every positive scores above every negative -> AUC 1.0
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert rank_auc(scores, labels) == 1.0


def test_auc_perfect_inversion():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([0, 0, 1, 1])
    assert rank_auc(scores, labels) == 0.0


def test_auc_chance_with_ties():
    # all scores identical -> no separation -> 0.5 (ties average to 0.5)
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    labels = torch.tensor([0, 1, 0, 1])
    assert abs(rank_auc(scores, labels) - 0.5) < 1e-6


def test_auc_degenerate_labels_is_nan_safe():
    # all-correct or all-wrong: AUC undefined -> returns None, not a crash
    scores = torch.tensor([0.1, 0.2, 0.3])
    assert rank_auc(scores, torch.tensor([1, 1, 1])) is None
    assert rank_auc(scores, torch.tensor([0, 0, 0])) is None


# ── per-example within-doc correctness ───────────────────────────────────────


def test_within_doc_correct_identity_all_right():
    # predictions == targets, so each retrieves itself -> all correct
    t = torch.randn(6, 12)
    doc = torch.tensor([0, 0, 0, 1, 1, 1])
    ok = within_doc_correct(t, t, doc)
    assert ok.dtype == torch.bool and ok.shape == (6,)
    assert ok.all()


def test_within_doc_correct_only_pools_same_doc():
    # row 0's prediction points exactly at target 3 (a DIFFERENT doc). Within
    # its own doc {0,1,2} target 0 is still its best match -> counted correct,
    # because the cross-doc distractor is masked out.
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0], [0.5, 0.5]])
    p = t.clone()
    p[0] = torch.tensor([1.0, 0.0])  # identical to t[0] AND t[3]
    doc = torch.tensor([0, 0, 0, 1, 1, 1])
    ok = within_doc_correct(p, t, doc)
    assert bool(ok[0]) is True


def test_within_doc_correct_singleton_doc_is_trivially_right():
    # a doc with one step: the only candidate is the true target
    t = torch.randn(3, 8)
    doc = torch.tensor([0, 1, 2])
    assert within_doc_correct(torch.randn(3, 8), t, doc).all()


# ── confidence signals ───────────────────────────────────────────────────────


def test_prediction_norm_matches_l2():
    p = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    n = prediction_norm(p)
    assert torch.allclose(n, torch.tensor([5.0, 0.0, 1.0]))


def test_retrieval_margin_sharp_vs_diffuse():
    # bank has two orthogonal directions. A pred aligned with one is SHARP
    # (big top1-top2 gap); a pred at 45° is diffuse (small gap).
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sharp = retrieval_margin(torch.tensor([[1.0, 0.0]]), bank)
    diffuse = retrieval_margin(torch.tensor([[1.0, 1.0]]), bank)
    assert sharp.item() > diffuse.item()
    assert abs(diffuse.item()) < 1e-6  # equal cosine to both -> zero margin


def test_dropout_agreement_high_when_samples_agree():
    # 3 near-identical samples -> agreement ~1; scattered samples -> lower
    agree = torch.stack([torch.ones(4, 8) + 0.001 * torch.randn(4, 8) for _ in range(3)])
    scatter = torch.stack([torch.randn(4, 8) for _ in range(3)])
    a = dropout_agreement(agree)
    s = dropout_agreement(scatter)
    assert a.shape == (4,)
    assert a.mean() > s.mean()
    assert a.mean() > 0.9


def test_mc_dropout_pool_shape_and_variation():
    # MC-dropout returns [n_samples, N, d_model]; with dropout on, samples differ
    torch.manual_seed(0)
    m = NextThoughtPredictor(d=8, k=4, d_model=16, layers=2, heads=2)
    g = torch.randn(2, 5, 4, 8)
    samples = mc_dropout_pool(m, g, n_samples=4)
    assert samples.shape[0] == 4 and samples.shape[2] == 16
    # at least some variation across dropout samples (dropout was active)
    assert samples.std(dim=0).mean() > 0


# ── tercile report ───────────────────────────────────────────────────────────


def test_tercile_report_separates_when_confidence_tracks_correctness():
    # confidence perfectly ranks correctness: low tercile all-wrong, high all-right
    conf = torch.linspace(0, 1, 30)
    correct = (conf > 0.5).to(torch.bool)
    rep = tercile_report(conf, correct)
    assert rep["low"]["acc"] == 0.0
    assert rep["high"]["acc"] == 1.0
    assert rep["auc"] > 0.9
    assert rep["n"] == 30


def test_tercile_report_flat_when_confidence_useless():
    # correctness independent of confidence -> terciles all ~equal, AUC ~0.5
    torch.manual_seed(0)
    conf = torch.rand(300)
    correct = (torch.rand(300) > 0.5).to(torch.bool)
    rep = tercile_report(conf, correct)
    assert abs(rep["auc"] - 0.5) < 0.15


# ── precision @ coverage (the deployable gate) ───────────────────────────────


def test_precision_at_coverage_picks_the_confident_slice():
    # confidence == correctness rank: the top-20% are all correct, base is 0.5
    conf = torch.linspace(0, 1, 100)
    correct = (conf >= 0.5).to(torch.bool)
    r = precision_at_coverage(conf, correct, coverage=0.2)
    assert r["acc"] == 1.0  # top-20% confident are all right
    assert r["n"] == 20
    assert r["base"] == 0.5
    assert r["lift"] == 0.5  # 1.0 - 0.5


def test_precision_at_coverage_useless_signal_no_lift():
    # confidence unrelated to correctness -> top slice ~ base, lift ~ 0
    torch.manual_seed(0)
    conf = torch.rand(400)
    correct = (torch.rand(400) > 0.6).to(torch.bool)
    r = precision_at_coverage(conf, correct, coverage=0.2)
    assert abs(r["lift"]) < 0.15


def test_coverage_curve_spans_fractions():
    conf = torch.linspace(0, 1, 50)
    correct = (conf >= 0.5).to(torch.bool)
    curve = coverage_curve(conf, correct, fractions=(0.1, 0.5))
    assert [p["coverage"] for p in curve] == [0.1, 0.5]
    # tighter coverage is purer here (all top-10% correct; top-50% is the boundary)
    assert curve[0]["acc"] >= curve[1]["acc"]


# ── absolute slot cosine (injection-relevant label) ──────────────────────────


def test_slot_cosine_identical_is_one():
    x = torch.randn(4, 8, 16)
    c = slot_cosine(x, x.clone())
    assert c.shape == (4,)
    assert torch.allclose(c, torch.ones(4), atol=1e-5)


def test_slot_cosine_opposite_is_negative_one():
    x = torch.randn(3, 8, 16)
    assert torch.allclose(slot_cosine(x, -x), -torch.ones(3), atol=1e-5)
