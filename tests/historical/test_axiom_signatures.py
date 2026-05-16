"""Mechanical invariants for per-axiom signature injection (Path 3).

Hypothesis: per-axiom K-vector fingerprints break the binding-ID
collision that causes the model to blend facts across stacked prefixes.

These tests assert the mechanical contract of the signature operation
(deterministic, axiom-specific, K-only). Whether the model actually
uses the signature to disambiguate facts is the demo's question.
"""

from __future__ import annotations

import pytest
import torch


def _make_prefix_like(n_total_layers: int, n_kv_heads: int, head_dim: int, n_tokens: int):  # noqa: ANN202
    """Build a minimal Prefix object with random K/V at every layer."""
    from marker.prefix_tuning import Prefix

    torch.manual_seed(0)
    keys = [
        torch.nn.Parameter(torch.randn(1, n_kv_heads, n_tokens, head_dim, dtype=torch.float32))
        for _ in range(n_total_layers)
    ]
    values = [
        torch.nn.Parameter(torch.randn(1, n_kv_heads, n_tokens, head_dim, dtype=torch.float32))
        for _ in range(n_total_layers)
    ]
    target_layers = list(range(n_total_layers))
    per_layer_shapes = [(n_kv_heads, n_tokens, head_dim)] * n_total_layers
    return Prefix(
        n_tokens=n_tokens,
        n_total_layers=n_total_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        target_layers=target_layers,
        keys=keys,
        values=values,
        per_layer_shapes=per_layer_shapes,
    )


# ---------- signature_vector ----------


def test_signature_vector_shape():
    from marker.historical.axiom_signatures import signature_vector

    sig = signature_vector("Flurgan_000", n_layers=4, n_kv_heads=2, head_dim=8)
    assert sig.shape == (4, 2, 8)
    assert sig.dtype == torch.float32


def test_signature_vector_deterministic():
    """Same name → identical fingerprint, always."""
    from marker.historical.axiom_signatures import signature_vector

    a = signature_vector("Flurgan_000", n_layers=4, n_kv_heads=2, head_dim=8)
    b = signature_vector("Flurgan_000", n_layers=4, n_kv_heads=2, head_dim=8)
    assert torch.equal(a, b)


def test_signature_vector_axiom_specific():
    """Different names → different fingerprints."""
    from marker.historical.axiom_signatures import signature_vector

    a = signature_vector("Flurgan_000", n_layers=4, n_kv_heads=2, head_dim=8)
    b = signature_vector("Boggin_001", n_layers=4, n_kv_heads=2, head_dim=8)
    assert (a - b).abs().max().item() > 1e-3


def test_signature_vector_unit_norm_per_layer_per_head():
    """Each (layer, head) signature is a unit vector — magnitude scaling
    is applied externally in `apply_signatures`. Keeps the contract
    clean and lets the demo sweep magnitudes."""
    from marker.historical.axiom_signatures import signature_vector

    sig = signature_vector("Flurgan_000", n_layers=4, n_kv_heads=2, head_dim=8)
    norms = sig.norm(dim=-1)  # (n_layers, n_kv_heads)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------- apply_signatures ----------


def test_apply_signatures_zero_magnitude_is_noop():
    """magnitude=0 must leave K (and V) byte-identical."""
    from marker.historical.axiom_signatures import apply_signatures

    p = _make_prefix_like(n_total_layers=4, n_kv_heads=2, head_dim=8, n_tokens=6)
    k_before = [k.clone() for k in p.keys]
    v_before = [v.clone() for v in p.values]
    out = apply_signatures([p], ["Flurgan_000"], magnitude=0.0)
    assert len(out) == 1
    for i in range(p.n_total_layers):
        assert torch.equal(out[0].keys[i], k_before[i])
        assert torch.equal(out[0].values[i], v_before[i])


def test_apply_signatures_only_modifies_K():
    """V is byte-identical before and after; K differs."""
    from marker.historical.axiom_signatures import apply_signatures

    p = _make_prefix_like(n_total_layers=4, n_kv_heads=2, head_dim=8, n_tokens=6)
    v_before = [v.clone() for v in p.values]
    k_before = [k.clone() for k in p.keys]
    out = apply_signatures([p], ["Flurgan_000"], magnitude=0.1)
    for i in range(p.n_total_layers):
        assert torch.equal(out[0].values[i], v_before[i]), f"V changed at layer {i}"
        assert not torch.equal(out[0].keys[i], k_before[i]), f"K unchanged at layer {i}"


def test_apply_signatures_per_axiom_distinct_offset():
    """Two axioms with different names get different K offsets."""
    from marker.historical.axiom_signatures import apply_signatures

    p1 = _make_prefix_like(n_total_layers=4, n_kv_heads=2, head_dim=8, n_tokens=6)
    p2 = _make_prefix_like(n_total_layers=4, n_kv_heads=2, head_dim=8, n_tokens=6)
    # Make p2's K start identical to p1's K so we can detect divergence.
    for i in range(p1.n_total_layers):
        p2.keys[i].data.copy_(p1.keys[i].data)
    out = apply_signatures([p1, p2], ["Flurgan_000", "Boggin_001"], magnitude=0.1)
    # K vectors at the same layer/position should now differ between the two
    diffs = [(out[0].keys[i] - out[1].keys[i]).abs().max().item() for i in range(p1.n_total_layers)]
    assert all(d > 1e-3 for d in diffs)


def test_apply_signatures_same_name_same_offset():
    """Two prefixes given the SAME name get the SAME K offset (idempotent
    per-name)."""
    from marker.historical.axiom_signatures import apply_signatures

    p1 = _make_prefix_like(n_total_layers=4, n_kv_heads=2, head_dim=8, n_tokens=6)
    p2 = _make_prefix_like(n_total_layers=4, n_kv_heads=2, head_dim=8, n_tokens=6)
    for i in range(p1.n_total_layers):
        p2.keys[i].data.copy_(p1.keys[i].data)
    out = apply_signatures([p1, p2], ["Flurgan_000", "Flurgan_000"], magnitude=0.1)
    for i in range(p1.n_total_layers):
        assert torch.equal(out[0].keys[i], out[1].keys[i])


def test_apply_signatures_magnitude_controls_size():
    """Doubling magnitude doubles the offset added to K."""
    from marker.historical.axiom_signatures import apply_signatures

    p_orig = _make_prefix_like(n_total_layers=2, n_kv_heads=2, head_dim=8, n_tokens=4)
    p_a = _make_prefix_like(n_total_layers=2, n_kv_heads=2, head_dim=8, n_tokens=4)
    p_b = _make_prefix_like(n_total_layers=2, n_kv_heads=2, head_dim=8, n_tokens=4)
    for i in range(p_orig.n_total_layers):
        p_a.keys[i].data.copy_(p_orig.keys[i].data)
        p_b.keys[i].data.copy_(p_orig.keys[i].data)
    out_a = apply_signatures([p_a], ["Flurgan_000"], magnitude=0.1)
    out_b = apply_signatures([p_b], ["Flurgan_000"], magnitude=0.2)
    for i in range(p_orig.n_total_layers):
        offset_a = (out_a[0].keys[i] - p_orig.keys[i]).abs().max().item()
        offset_b = (out_b[0].keys[i] - p_orig.keys[i]).abs().max().item()
        assert offset_b == pytest.approx(2 * offset_a, rel=1e-4)


def test_apply_signatures_returns_new_objects():
    """Inputs must not be mutated."""
    from marker.historical.axiom_signatures import apply_signatures

    p = _make_prefix_like(n_total_layers=2, n_kv_heads=2, head_dim=8, n_tokens=4)
    k_before = [k.clone() for k in p.keys]
    _ = apply_signatures([p], ["Flurgan_000"], magnitude=0.1)
    for i in range(p.n_total_layers):
        assert torch.equal(p.keys[i], k_before[i]), f"input K mutated at layer {i}"
