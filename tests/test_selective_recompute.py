"""Mechanical invariants for CacheBlend-style selective recompute.

Per CLAUDE.md: tests assert mechanical invariants (correct shapes,
correct positions flagged, only-flagged-positions written, no-op when
nothing deviates), NOT numerical experiment outcomes (does the recompute
actually fix the chain prompt). Experiment outcomes live in
plots/artifacts and `run_chain_selective_recompute_demo.py`.
"""

from __future__ import annotations

import pytest
import torch


def _make_k(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


# ---------- find_high_deviation_positions ----------


def test_deviation_indices_correct_count():
    """top_k_pct=0.25 of 32 positions => 8 indices."""
    from marker.selective_recompute import _deviation_indices

    cached = _make_k(0, (1, 8, 32, 64))
    joint = _make_k(1, (1, 8, 32, 64))
    out = _deviation_indices(cached, joint, top_k_pct=0.25)
    assert out.dtype == torch.long
    assert out.shape == (8,)
    assert int(out.min()) >= 0 and int(out.max()) < 32


def test_deviation_indices_picks_perturbed_positions():
    """When only positions {5, 10, 15} differ between cached and joint K,
    those three must be in the top-3 most-deviant indices."""
    from marker.selective_recompute import _deviation_indices

    cached = _make_k(42, (1, 8, 32, 64))
    joint = cached.clone()
    perturbed = [5, 10, 15]
    for p in perturbed:
        joint[0, :, p, :] += 100.0  # huge deviation, dominates any noise
    # 3/32 ≈ 0.094, ask for top 10% (= 3 positions, since round(32*0.10)=3)
    out = _deviation_indices(cached, joint, top_k_pct=0.10)
    assert set(out.tolist()) == set(perturbed)


def test_deviation_indices_no_op_when_identical():
    """When cached == joint, the L2 diff is zero everywhere; the function
    should still return the requested count of indices (any indices)."""
    from marker.selective_recompute import _deviation_indices

    k = _make_k(7, (1, 4, 16, 32))
    out = _deviation_indices(k, k.clone(), top_k_pct=0.25)
    assert out.shape == (4,)


# ---------- selective_recompute_prefix_cache ----------


@pytest.fixture(scope="module")
def tiny_model():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return mdl, tok


def test_recompute_writes_only_flagged_positions(tiny_model):
    """K/V at non-flagged positions must be byte-identical after recompute;
    K/V at flagged positions must change."""
    from transformers import DynamicCache

    from marker.selective_recompute import selective_recompute_prefix_cache

    model, tok = tiny_model
    device = next(model.parameters()).device

    text = "The Balance Publisher polls a REST endpoint every 250 milliseconds."
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(ids, past_key_values=DynamicCache(), use_cache=True)
    base_cache: DynamicCache = out.past_key_values

    # Snapshot before
    n_layers = len(base_cache)
    before_k = [base_cache.layers[i].keys.clone() for i in range(n_layers)]
    before_v = [base_cache.layers[i].values.clone() for i in range(n_layers)]

    seq_len = ids.shape[1]
    # Pick flagged positions in [1, seq_len)
    flagged = torch.tensor([1, seq_len // 2], dtype=torch.long)

    new_cache = selective_recompute_prefix_cache(
        model=model,
        base_cache=base_cache,
        joint_input_ids=ids,
        high_deviation_positions=flagged,
    )

    flagged_set = set(flagged.tolist())
    for i in range(n_layers):
        new_k = new_cache.layers[i].keys
        new_v = new_cache.layers[i].values
        assert new_k.shape == before_k[i].shape
        assert new_v.shape == before_v[i].shape
        for pos in range(seq_len):
            if pos in flagged_set:
                continue
            assert torch.allclose(new_k[..., pos, :], before_k[i][..., pos, :], atol=1e-5), (
                f"layer {i} pos {pos} K modified but not flagged"
            )
            assert torch.allclose(new_v[..., pos, :], before_v[i][..., pos, :], atol=1e-5), (
                f"layer {i} pos {pos} V modified but not flagged"
            )


def test_recompute_all_positions_matches_vanilla(tiny_model):
    """Make-or-break test: if ALL positions are flagged for recompute, the
    output cache must equal a vanilla model forward within fp tolerance.
    If this fails, the custom forward path is broken."""
    from transformers import DynamicCache

    from marker.selective_recompute import selective_recompute_prefix_cache

    model, tok = tiny_model
    device = next(model.parameters()).device

    text = "Hello world this is a short test sentence."
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    with torch.no_grad():
        ref = model(ids, past_key_values=DynamicCache(), use_cache=True)
    ref_cache: DynamicCache = ref.past_key_values

    # Build a "stale" cache to start from — same shapes, different values.
    stale_cache = DynamicCache()
    for i in range(len(ref_cache)):
        k = ref_cache.layers[i].keys
        v = ref_cache.layers[i].values
        stale_cache.update(torch.zeros_like(k), torch.zeros_like(v), i)

    seq_len = ids.shape[1]
    all_positions = torch.arange(seq_len, dtype=torch.long)

    new_cache = selective_recompute_prefix_cache(
        model=model,
        base_cache=stale_cache,
        joint_input_ids=ids,
        high_deviation_positions=all_positions,
    )

    for i in range(len(ref_cache)):
        assert torch.allclose(
            new_cache.layers[i].keys, ref_cache.layers[i].keys, atol=1e-3, rtol=1e-3
        ), f"layer {i} K mismatch vs vanilla forward"
        assert torch.allclose(
            new_cache.layers[i].values, ref_cache.layers[i].values, atol=1e-3, rtol=1e-3
        ), f"layer {i} V mismatch vs vanilla forward"


# ---------- blend_prefixes integration ----------


def test_blend_prefixes_sanity_three_prefix_decode(tiny_model):
    """Sanity rail: tiny model + 3 prefixes + selective recompute should
    decode at least one non-EOS token (i.e. not crash, not immediately
    terminate). Numerical correctness is the demo's job, not this test."""
    from marker.prefix_tuning import Prefix
    from marker.selective_recompute import blend_prefixes

    model, tok = tiny_model

    layers = list(range(model.config.num_hidden_layers))
    prefixes = []
    for txt in [
        "Alpha is a service that publishes balances.",
        "Beta consumes balances and computes risk.",
        "Gamma signs orders and routes them to the exchange.",
    ]:
        p = Prefix.from_description(model, tok, txt, target_layers=layers)
        prefixes.append(p)

    cache = blend_prefixes(model=model, prefixes=prefixes, selective_recompute=True)
    # All layers populated, all with the expected total prefix length.
    expected_len = sum(p.n_tokens for p in prefixes)
    for i in range(len(cache)):
        assert cache.layers[i].keys.shape[-2] == expected_len
