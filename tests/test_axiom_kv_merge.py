"""Mechanical invariants for AxiomKV merging and RoPE correction.

CONCLUSIONS.md documents why naive multi-prefix concatenation fails: each
KV's keys carry RoPE rotations for capture positions 0..n-1, so concatenated
prefixes overlap geometrically. merge_axiom_kvs must re-rotate each
non-first KV's keys to match its cache-slot position (same fix as
prefix_tuning.combined_cache).
"""

from __future__ import annotations

import torch

from marker.prefix_tuning import _rope_offset
from marker.run_axiom_mlp_demo import AxiomKV, SmallMLP, merge_axiom_kvs

THETA = 10000.0


def _fake_kv(n_layers: int = 2, n_tokens: int = 5, seed: int = 0) -> AxiomKV:
    g = torch.Generator().manual_seed(seed)
    shape = (1, 2, n_tokens, 8)
    return AxiomKV(
        n_layers=n_layers,
        keys=[torch.randn(shape, generator=g) for _ in range(n_layers)],
        values=[torch.randn(shape, generator=g) for _ in range(n_layers)],
    )


def test_zero_init_mlp_is_noop():
    mlp = SmallMLP(16, r=4)
    x = torch.randn(3, 16)
    assert torch.equal(mlp(x), torch.zeros(3, 16))


def test_merge_single_kv_is_identity():
    kv = _fake_kv()
    merged = merge_axiom_kvs([kv], rope_theta=THETA)
    for layer in range(kv.n_layers):
        assert torch.equal(merged.keys[layer], kv.keys[layer])
        assert torch.equal(merged.values[layer], kv.values[layer])


def test_merge_rotates_second_kv_keys_only():
    a, b = _fake_kv(seed=0), _fake_kv(seed=1)
    merged = merge_axiom_kvs([a, b], rope_theta=THETA)
    n_a = a.keys[0].shape[2]
    for layer in range(a.n_layers):
        # first KV's keys unchanged (offset 0)
        assert torch.equal(merged.keys[layer][:, :, :n_a], a.keys[layer])
        # second KV's keys re-rotated by n_a positions
        expected = _rope_offset(b.keys[layer], n_a, THETA, b.keys[layer].shape[-1])
        assert torch.allclose(merged.keys[layer][:, :, n_a:], expected)
        assert not torch.allclose(merged.keys[layer][:, :, n_a:], b.keys[layer])
        # values are never rotated
        assert torch.equal(merged.values[layer][:, :, n_a:], b.values[layer])


def test_base_offset_rotates_first_kv():
    kv = _fake_kv()
    merged = merge_axiom_kvs([kv], rope_theta=THETA, base_offset=7)
    expected = _rope_offset(kv.keys[0], 7, THETA, kv.keys[0].shape[-1])
    assert torch.allclose(merged.keys[0], expected)


def test_rope_offset_preserves_norm():
    k = torch.randn(1, 2, 5, 8)
    rotated = _rope_offset(k, 13, THETA, 8)
    assert torch.allclose(rotated.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_rope_offset_composes_additively():
    k = torch.randn(1, 2, 5, 8)
    once = _rope_offset(k, 9, THETA, 8)
    twice = _rope_offset(_rope_offset(k, 4, THETA, 8), 5, THETA, 8)
    assert torch.allclose(once, twice, atol=1e-5)
