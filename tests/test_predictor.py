"""Tests for the Stage-2 next-thought predictor (predictor.py).

Model-free: block-causal mask values, loss/metric math on crafted tensors.
Slow: the predictor overfits a tiny fixed gist sequence (learns thought
succession) and its retrieval recall rises above chance.
"""

from __future__ import annotations

import pytest
import torch

from marker.predictor import (
    NextThoughtPredictor,
    build_block_causal_mask,
    info_nce_loss,
    prediction_diversity,
    recall_at_k,
    regression_loss,
)

NEG = float("-inf")


# ── block-causal mask ───────────────────────────────────────────────────────────


def test_mask_sentence_causal():
    m = build_block_causal_mask(n_sents=3, k=2)  # T=6, sentences {0,1},{2,3},{4,5}
    assert m.shape == (6, 6)
    # sentence 0 (tokens 0,1) cannot see sentence 1 (tokens 2,3)
    assert m[0, 2] == NEG and m[1, 3] == NEG
    # sentence 1 CAN see sentence 0 and itself
    assert m[2, 0] == 0.0 and m[2, 3] == 0.0
    # within a sentence, both slots mutually visible (not token-causal)
    assert m[0, 1] == 0.0 and m[1, 0] == 0.0
    # last sentence sees everything before
    assert (m[4, :] == 0.0).all()


# ── losses / metrics ────────────────────────────────────────────────────────────


def test_regression_loss_zero_when_aligned():
    x = torch.randn(2, 3, 8, 16)
    assert regression_loss(x, x.clone()) < 1e-6
    assert regression_loss(x, -x) > 1.9  # opposite -> cos -1 -> loss 2


def test_info_nce_low_when_pred_matches_target():
    torch.manual_seed(0)
    t = torch.randn(16, 32)
    same = info_nce_loss(t, t)  # perfect alignment
    shuffled = info_nce_loss(t[torch.randperm(16)], t)  # mismatched
    assert same < shuffled


def test_recall_at_k_perfect_and_chance():
    t = torch.randn(20, 32)
    assert recall_at_k(t, t, topk=1) == 1.0  # each matches itself
    import random  # noqa: PLC0415

    random.seed(0)
    rnd = torch.randn(200, 32)
    r = recall_at_k(rnd, torch.randn(200, 32), topk=5)
    assert r < 0.15  # ~ 5/200 chance, well below a real signal


def test_diversity_high_on_collapse_low_on_varied():
    collapsed = torch.ones(10, 16) + 0.001 * torch.randn(10, 16)
    varied = torch.randn(10, 16)
    assert prediction_diversity(collapsed) > 0.9
    assert prediction_diversity(varied) < 0.5


# ── slow: the predictor learns ──────────────────────────────────────────────────


@pytest.mark.slow
def test_predictor_shapes():
    m = NextThoughtPredictor(d=16, k=4, d_model=32, layers=2, heads=4)
    g = torch.randn(2, 5, 4, 16)  # B=2, L=5 sentences, k=4, d=16
    out = m(g)
    assert out.shape == (2, 4, 4, 16)  # predicts sentences 1..4 from 0..3


@pytest.mark.slow
def test_predictor_overfits_a_fixed_sequence():
    # a small deterministic gist sequence — the predictor should learn to
    # predict each next sentence's slots (regression loss drops sharply).
    torch.manual_seed(0)
    m = NextThoughtPredictor(d=8, k=4, d_model=48, layers=2, heads=4)
    g = torch.randn(4, 6, 4, 8)  # 4 sequences of 6 sentences
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for i in range(120):
        opt.zero_grad()
        pred = m(g)
        loss = regression_loss(pred, g[:, 1:])
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.5, f"predictor didn't learn: {first:.3f} -> {last:.3f}"


@pytest.mark.slow
def test_retrieval_recall_rises_with_training():
    # after overfitting, the predicted next-gist should retrieve its true
    # target above chance (recall@1 on the pooled projections).
    torch.manual_seed(1)
    m = NextThoughtPredictor(d=8, k=4, d_model=48, layers=2, heads=4)
    g = torch.randn(8, 5, 4, 8)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        pred = m(g)
        tgt = g[:, 1:]
        pp = m.pool(pred).reshape(-1, m.pool_proj.out_features)
        tp = m.pool(tgt).reshape(-1, m.pool_proj.out_features)
        (regression_loss(pred, tgt) + info_nce_loss(pp, tp)).backward()
        opt.step()
    with torch.no_grad():
        pred = m(g)
        pp = m.pool(pred).reshape(-1, m.pool_proj.out_features)
        tp = m.pool(g[:, 1:]).reshape(-1, m.pool_proj.out_features)
    assert recall_at_k(pp, tp, topk=1) > 0.5  # memorized -> strong retrieval
