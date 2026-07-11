"""Model-free invariants for gist-space whitening (whiten.py).

Sentence-embedding / gist space is anisotropic — cone-shaped, variance in a
few directions (spec killer #2). Whitening transforms gists to zero-mean,
identity-covariance so the predictor regresses in a clean space. These pin the
statistical invariants: round-trip exactness, whitened mean ~0, whitened
covariance ~I.
"""

from __future__ import annotations

import torch

from marker.whiten import Whitener


def _anisotropic_gists(n=2000, d=16):
    torch.manual_seed(0)
    # cone-shaped: big shared mean + variance concentrated in a few directions
    base = torch.randn(n, d)
    base[:, 0] *= 8.0
    base[:, 1] *= 4.0
    return base + torch.tensor([5.0] + [0.0] * (d - 1))


def test_round_trip_is_identity():
    g = _anisotropic_gists()
    w = Whitener.fit(g)
    back = w.inverse(w.transform(g))
    assert torch.allclose(g, back, atol=1e-3)


def test_whitened_mean_is_zero():
    g = _anisotropic_gists()
    w = Whitener.fit(g)
    wm = w.transform(g).mean(0)
    assert torch.allclose(wm, torch.zeros_like(wm), atol=1e-4)


def test_whitened_covariance_is_identity():
    g = _anisotropic_gists()
    w = Whitener.fit(g)
    z = w.transform(g)
    cov = (z.T @ z) / (z.shape[0] - 1)
    assert torch.allclose(cov, torch.eye(z.shape[1]), atol=1e-2)


def test_whitened_directions_have_unit_variance():
    # the anisotropy (dims 0,1 huge) must be gone after whitening
    g = _anisotropic_gists()
    w = Whitener.fit(g)
    var = w.transform(g).var(0)
    assert torch.allclose(var, torch.ones_like(var), atol=5e-2)


def test_save_load_round_trip(tmp_path):
    g = _anisotropic_gists()
    w = Whitener.fit(g)
    p = tmp_path / "whiten.pt"
    w.save(p)
    w2 = Whitener.load(p)
    assert torch.allclose(w.transform(g), w2.transform(g), atol=1e-5)
