"""Model-free invariants for Stage-0 soft-token feedback (softloop.py).

The soft loop's math is pure tensor manipulation: a temperature-softened,
top-p-truncated distribution over the vocabulary, mixed against the input
embedding matrix, with a deterministic snapping schedule. All of it is
testable with stub tensors; the model-level invariant (k=1 == greedy decode,
token for token) is asserted by the runner's smoke mode.
"""

from __future__ import annotations

import math

import torch

from marker.softloop import entropy, is_snap_step, mix_embedding, soft_distribution

# ── soft_distribution ───────────────────────────────────────────────────────────


def test_distribution_sums_to_one():
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
    p = soft_distribution(logits, tau=0.7, top_p=0.95)
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)


def test_top_p_zeroes_the_tail():
    # One dominant logit: with top_p=0.5 only the top token survives.
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
    p = soft_distribution(logits, tau=1.0, top_p=0.5)
    assert p[0] > 0.999
    assert torch.count_nonzero(p) == 1


def test_top_p_keeps_at_least_one_token():
    # Even a tiny top_p must keep the argmax (never an all-zero distribution).
    logits = torch.tensor([1.0, 1.0, 1.0])
    p = soft_distribution(logits, tau=1.0, top_p=0.01)
    assert torch.count_nonzero(p) >= 1
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6)


def test_low_tau_approaches_one_hot():
    logits = torch.tensor([3.0, 2.9, 0.0])
    p = soft_distribution(logits, tau=0.01, top_p=1.0)
    assert p[0] > 0.99  # near-tie sharpened decisively by low temperature


def test_argmax_preserved_by_truncation():
    logits = torch.randn(50)
    p = soft_distribution(logits, tau=0.7, top_p=0.9)
    assert int(p.argmax()) == int(logits.argmax())


# ── entropy ─────────────────────────────────────────────────────────────────────


def test_entropy_uniform_is_log_n():
    p = torch.full((8,), 1 / 8)
    assert math.isclose(float(entropy(p)), math.log(8), rel_tol=1e-5)


def test_entropy_one_hot_is_zero():
    p = torch.zeros(8)
    p[3] = 1.0
    assert float(entropy(p)) < 1e-6


# ── mix_embedding ───────────────────────────────────────────────────────────────


def test_mix_is_probability_weighted_sum():
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    p = torch.tensor([0.5, 0.5, 0.0])
    e = mix_embedding(p, E)
    assert torch.allclose(e, torch.tensor([0.5, 0.5]))


def test_mix_one_hot_recovers_embedding_row():
    E = torch.randn(5, 3)
    p = torch.zeros(5)
    p[2] = 1.0
    assert torch.allclose(mix_embedding(p, E), E[2])


# ── snapping schedule ───────────────────────────────────────────────────────────


def test_k1_snaps_every_step():
    assert all(is_snap_step(i, 1) for i in range(10))


def test_k4_snaps_every_fourth():
    flags = [is_snap_step(i, 4) for i in range(8)]
    assert flags == [False, False, False, True, False, False, False, True]


def test_k_none_never_snaps():
    assert not any(is_snap_step(i, None) for i in range(100))
