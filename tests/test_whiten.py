"""Model-free invariants for gist-space whitening (whiten.py).

Sentence-embedding / gist space is anisotropic — cone-shaped, variance in a
few directions (spec killer #2). Whitening transforms gists to zero-mean,
identity-covariance so the predictor regresses in a clean space. These pin the
statistical invariants: round-trip exactness, whitened mean ~0, whitened
covariance ~I.
"""

from __future__ import annotations

import torch

from marker.whiten import PerSlotWhitener, Whitener


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


# ── streaming fit: same result as in-memory, without materializing all gists ────


def test_streaming_fit_matches_in_memory():
    g = _anisotropic_gists(n=3000, d=16)
    full = Whitener.fit(g)
    chunks = [g[i : i + 250] for i in range(0, len(g), 250)]  # 12 chunks
    stream = Whitener.fit_streaming(iter(chunks))
    # same statistical fit -> same whitened output
    assert torch.allclose(full.transform(g), stream.transform(g), atol=1e-2)
    assert torch.allclose(full.mean, stream.mean, atol=1e-3)


# ── per-slot whiteners: 8 slots with different distributions, each whitened ──────


def _per_slot_gists(n=1500, k=8, d=8):
    torch.manual_seed(1)
    g = torch.randn(n, k, d)
    for s in range(k):  # give each slot index a different scale + mean
        g[:, s, :] *= 1.0 + s
        g[:, s, 0] += s
    return g


def test_per_slot_whitens_each_slot_independently():
    g = _per_slot_gists()
    w = PerSlotWhitener.fit(g)
    z = w.transform(g)  # [N, k, d]
    for s in range(g.shape[1]):
        zs = z[:, s, :]
        assert torch.allclose(zs.mean(0), torch.zeros(g.shape[2]), atol=1e-4), f"slot {s} mean"
        cov = (zs.T @ zs) / (zs.shape[0] - 1)
        assert torch.allclose(cov, torch.eye(g.shape[2]), atol=5e-2), f"slot {s} cov"


def test_per_slot_round_trip():
    g = _per_slot_gists()
    w = PerSlotWhitener.fit(g)
    assert torch.allclose(w.inverse(w.transform(g)), g, atol=1e-3)
