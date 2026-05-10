"""Mechanical invariants for per-block attention (custom SDPA).

Per-block attention runs a separate softmax inside each axiom's slot
range and mixes per-block outputs. Expected to fix attention-entropy
collapse for 3+ stacked prefixes — directly addresses the failure
mode APE only partially fixed.

These tests assert mechanical invariants, not experimental outcomes.
"""

from __future__ import annotations

import hashlib

import pytest
import torch


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


def _state_checksum(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------- pure-tensor unit tests for the per-block sdpa kernel ----------


def test_per_block_sdpa_single_block_matches_vanilla():
    """One block spanning the whole sequence must equal F.scaled_dot_product_attention."""
    import torch.nn.functional as F  # noqa: N812

    from marker.per_block_attention import per_block_sdpa

    torch.manual_seed(0)
    B, H, Lq, Lk, D = 1, 4, 3, 16, 32
    q = torch.randn(B, H, Lq, D)
    k = torch.randn(B, H, Lk, D)
    v = torch.randn(B, H, Lk, D)

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
    out = per_block_sdpa(q, k, v, boundaries=[(0, Lk)], combiner="uniform")
    assert torch.allclose(out, ref, atol=1e-5), "single-block per-block sdpa must equal vanilla"


def test_per_block_sdpa_uniform_two_blocks_matches_average_of_partial_attentions():
    """With 2 blocks and uniform combiner, output must equal
    (attn_over_block1 + attn_over_block2) / 2."""
    import torch.nn.functional as F  # noqa: N812

    from marker.per_block_attention import per_block_sdpa

    torch.manual_seed(1)
    B, H, Lq, D = 1, 2, 2, 8
    Lk1, Lk2 = 5, 7
    q = torch.randn(B, H, Lq, D)
    k1 = torch.randn(B, H, Lk1, D)
    k2 = torch.randn(B, H, Lk2, D)
    v1 = torch.randn(B, H, Lk1, D)
    v2 = torch.randn(B, H, Lk2, D)
    k = torch.cat([k1, k2], dim=2)
    v = torch.cat([v1, v2], dim=2)

    out = per_block_sdpa(q, k, v, boundaries=[(0, Lk1), (Lk1, Lk1 + Lk2)], combiner="uniform")
    a1 = F.scaled_dot_product_attention(q, k1, v1)
    a2 = F.scaled_dot_product_attention(q, k2, v2)
    expected = (a1 + a2) / 2.0
    assert torch.allclose(out, expected, atol=1e-5)


def test_per_block_sdpa_lse_combiner_recovers_vanilla():
    """LSE-weighted combiner is mathematically equivalent to vanilla flat
    attention. (Sanity check that the per-block decomposition is correct.)
    """
    import torch.nn.functional as F  # noqa: N812

    from marker.per_block_attention import per_block_sdpa

    torch.manual_seed(2)
    B, H, Lq, D = 1, 2, 2, 8
    Lk1, Lk2 = 5, 7
    q = torch.randn(B, H, Lq, D)
    k = torch.randn(B, H, Lk1 + Lk2, D)
    v = torch.randn(B, H, Lk1 + Lk2, D)

    ref = F.scaled_dot_product_attention(q, k, v)
    out = per_block_sdpa(q, k, v, boundaries=[(0, Lk1), (Lk1, Lk1 + Lk2)], combiner="lse")
    assert torch.allclose(out, ref, atol=1e-4), (
        "LSE combiner should mathematically recover vanilla attention"
    )


def test_per_block_sdpa_uniform_differs_from_vanilla():
    """Uniform combiner must differ from vanilla for >1 block (proves the
    intervention is doing something)."""
    import torch.nn.functional as F  # noqa: N812

    from marker.per_block_attention import per_block_sdpa

    torch.manual_seed(3)
    B, H, Lq, D = 1, 2, 2, 8
    Lk = 12
    q = torch.randn(B, H, Lq, D)
    k = torch.randn(B, H, Lk, D)
    v = torch.randn(B, H, Lk, D)

    ref = F.scaled_dot_product_attention(q, k, v)
    out = per_block_sdpa(q, k, v, boundaries=[(0, 4), (4, 8), (8, 12)], combiner="uniform")
    assert (out - ref).abs().max().item() > 1e-3


# ---------- model-level integration ----------


def test_install_no_op_at_one_block(tiny_model):
    """Installing per-block attention with a single block spanning the
    full key length must produce identical logits to vanilla forward."""
    from transformers.cache_utils import DynamicCache

    from marker.per_block_attention import (
        install_per_block_attention,
        set_block_boundaries,
    )

    model, tok = tiny_model
    ids = tok("Hello world test sentence here.", return_tensors="pt", add_special_tokens=False)
    ids = ids.input_ids
    with torch.no_grad():
        ref = model(ids, past_key_values=DynamicCache(), use_cache=True)

    handle = install_per_block_attention(model, combiner="uniform")
    try:
        # Set boundary that spans the full sequence (1 block) → should match vanilla
        # for the uniform combiner with one block (= per_block_sdpa identity).
        set_block_boundaries([(0, ids.shape[1])])
        with torch.no_grad():
            out = model(ids, past_key_values=DynamicCache(), use_cache=True)
    finally:
        handle.remove()
        set_block_boundaries(None)
    assert torch.allclose(out.logits, ref.logits, atol=1e-3, rtol=1e-3), (
        "single-block install should match vanilla logits"
    )


def test_install_does_not_change_weights(tiny_model):
    """Install / uninstall must leave model state_dict byte-identical."""
    from marker.per_block_attention import (
        install_per_block_attention,
        set_block_boundaries,
    )

    model, _ = tiny_model
    before = _state_checksum(model)
    handle = install_per_block_attention(model, combiner="uniform")
    set_block_boundaries(None)
    handle.remove()
    after = _state_checksum(model)
    assert before == after


def test_install_runs_5_block_decode(tiny_model):
    """Sanity rail: installing with 5 blocks and decoding should not crash."""
    from marker.per_block_attention import generate_with_per_block
    from marker.prefix_tuning import Prefix

    model, tok = tiny_model
    layers = list(range(model.config.num_hidden_layers))
    prefixes = [
        Prefix.from_description(model, tok, txt, target_layers=layers)
        for txt in [
            "EventLog stores click events in a Kafka topic.",
            "KafkaRouter reads from EventLog and routes to two topics.",
            "FeatureStore subscribes to EventLog and writes to Redis.",
            "ModelServer reads from FeatureStore via Redis.",
            "DataPipeline is composed of EventLog, KafkaRouter, FeatureStore, ModelServer.",
        ]
    ]
    out = generate_with_per_block(
        model=model,
        tokenizer=tok,
        prompt="What does EventLog store?",
        prefixes=prefixes,
        combiner="uniform",
        max_new=10,
    )
    assert isinstance(out, str)
