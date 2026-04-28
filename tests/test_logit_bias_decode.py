"""Mechanical invariants for decode-time logit biasing.

The mechanism: at every decoded step, add α · (W_U @ v) to the
next-token logits — bypassing residual-stream injection entirely.
Tests assert α=0 is a no-op and the bias has the right shape.
"""

from __future__ import annotations

import numpy as np
import torch

from marker.run_logit_bias_decode import compute_logit_bias


class _FakeLMHead:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


def test_zero_alpha_returns_zero_bias() -> None:
    vocab, hidden = 64, 8
    weight = torch.randn(vocab, hidden)
    v = np.random.randn(hidden).astype(np.float32)
    bias = compute_logit_bias(weight, v, alpha=0.0)
    assert torch.allclose(bias, torch.zeros(vocab, dtype=bias.dtype))


def test_bias_shape_matches_vocab() -> None:
    vocab, hidden = 64, 8
    weight = torch.randn(vocab, hidden)
    v = np.random.randn(hidden).astype(np.float32)
    bias = compute_logit_bias(weight, v, alpha=1.0)
    assert bias.shape == (vocab,)


def test_alpha_scales_linearly() -> None:
    vocab, hidden = 64, 8
    weight = torch.randn(vocab, hidden)
    v = np.random.randn(hidden).astype(np.float32)
    b1 = compute_logit_bias(weight, v, alpha=1.0)
    b2 = compute_logit_bias(weight, v, alpha=2.0)
    assert torch.allclose(b2, 2.0 * b1, atol=1e-5)
