"""Model-level tests for the gist LoRA + training forward (gist_model.py).

Slow (tiny Qwen2 on CPU): trainable-set is LoRA+gist only, the sdpa assertion,
a finite loss, loss decreasing over a few steps on a fixed batch, and the
3-PPL direction (full <= none). gap_closed math is unit-tested model-free.
"""

from __future__ import annotations

import pytest
import torch

from marker.gist_model import (
    assert_attn_impl,
    attach_gist,
    gap_closed,
    gist_forward,
    three_ppls,
    trainable_param_names,
)

# ── model-free: gap_closed arithmetic ───────────────────────────────────────────


def test_gap_closed_full_match():
    assert gap_closed({"none": 100.0, "full": 10.0, "gist": 10.0}) == pytest.approx(1.0)


def test_gap_closed_no_help():
    assert gap_closed({"none": 100.0, "full": 10.0, "gist": 100.0}) == pytest.approx(0.0)


def test_gap_closed_half():
    assert gap_closed({"none": 100.0, "full": 20.0, "gist": 60.0}) == pytest.approx(0.5)


def test_gap_closed_degenerate_gap():
    # full not better than none -> gap undefined -> 0.0, no divide-by-zero
    assert gap_closed({"none": 10.0, "full": 10.0, "gist": 5.0}) == 0.0


# ── slow: tiny real model ───────────────────────────────────────────────────────


def _tiny_base():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    return model.eval()


@pytest.mark.slow
def test_assert_attn_impl_rejects_flash():
    m = _tiny_base()
    assert_attn_impl(m)  # sdpa ok
    m.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="4D masks"):
        assert_attn_impl(m)


@pytest.mark.slow
def test_only_lora_and_gist_are_trainable():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    trainable = trainable_param_names(peft_model)
    assert trainable, "no trainable params"
    assert all("lora" in n.lower() for n in trainable), (
        f"non-LoRA base param is trainable: {[n for n in trainable if 'lora' not in n.lower()]}"
    )
    assert gist.requires_grad


@pytest.mark.slow
def test_forward_finite_loss():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    loss = gist_forward(peft_model, gist, [[1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10]])
    assert torch.isfinite(loss)


@pytest.mark.slow
def test_loss_decreases_over_steps():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    params = [p for p in peft_model.parameters() if p.requires_grad] + [gist]
    opt = torch.optim.AdamW(params, lr=1e-2)
    spans, conts = [[1, 2, 3, 4]], [[5, 6, 7, 8]]
    first = last = None
    for i in range(25):
        opt.zero_grad()
        loss = gist_forward(peft_model, gist, spans, conts)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"


@pytest.mark.slow
def test_three_ppls_direction_full_le_none():
    base = _tiny_base()
    peft_model, gist = attach_gist(base, gist_k=4, r=4)
    # train a little so 'full' (raw span visible) genuinely beats 'none'
    params = [p for p in peft_model.parameters() if p.requires_grad] + [gist]
    opt = torch.optim.AdamW(params, lr=1e-2)
    spans, conts = [[1, 2, 3, 4]], [[5, 6, 7, 8]]
    for _ in range(30):
        opt.zero_grad()
        gist_forward(
            peft_model, gist, spans, conts, cont_sees=frozenset({"gist", "span"})
        ).backward()
        opt.step()
    ppls = three_ppls(peft_model, gist, spans, conts)
    assert all(torch.isfinite(torch.tensor(v)) for v in ppls.values())
    assert ppls["full"] <= ppls["none"] + 1e-3, ppls
