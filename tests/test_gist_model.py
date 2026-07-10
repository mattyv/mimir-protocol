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
    generate_from_gist,
    gist_forward,
    roll_spans,
    three_ppls,
    to_leaf_param,
    trainable_param_names,
)


def test_roll_spans_permutes_all_positions():
    # every continuation gets a DIFFERENT span (no fixed point) for n>=2
    spans = [[1], [2], [3]]
    rolled = roll_spans(spans)
    assert rolled == [[3], [1], [2]]
    assert all(r != o for r, o in zip(rolled, spans, strict=True))


# ── to_leaf_param: the GPU-only "non-leaf Tensor" optimizer crash ───────────────


def test_moved_param_is_non_leaf_but_to_leaf_param_fixes_it():
    p = torch.nn.Parameter(torch.randn(4, 8))
    # a device/dtype move returns a NON-leaf tensor that AdamW rejects (CPU
    # .to(cpu) is a no-op, so force it with a dtype move to reproduce on CPU)
    moved = p.to(torch.float64)
    assert not moved.is_leaf
    with pytest.raises(ValueError, match="non-leaf"):
        torch.optim.AdamW([moved])
    # to_leaf_param re-wraps as a leaf -> optimizer accepts it
    fixed = to_leaf_param(p, torch.device("cpu"))
    assert fixed.is_leaf and fixed.requires_grad
    torch.optim.AdamW([fixed])  # no raise


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
def test_adapter_save_load_round_trip(tmp_path):
    # The resume path: save_bundle -> fresh model -> set_peft_model_state_dict
    # (NOT load_adapter, which raises on the existing 'default' name — Fable
    # pre-launch finding #1). Loss on the same batch must match exactly.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from marker.hf_push import save_bundle

    spans, conts = [[1, 2, 3, 4]], [[5, 6, 7, 8]]

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    params = [p for p in pm.parameters() if p.requires_grad] + [gist]
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(5):  # train so LoRA weights are nonzero (init B=0 is trivial)
        opt.zero_grad()
        gist_forward(pm, gist, spans, conts).backward()
        opt.step()
    save_bundle(tmp_path, pm, gist, {"step": 5})

    base2 = _tiny_base()  # same seed -> identical base
    pm2, gist2 = attach_gist(base2, gist_k=4, r=4)
    adapter_state = load_file(str(tmp_path / "adapter_model.safetensors"))
    set_peft_model_state_dict(pm2, adapter_state)
    gist2.data = load_file(str(tmp_path / "gist.safetensors"))["gist"]

    with torch.no_grad():
        l1 = gist_forward(pm, gist, spans, conts)
        l2 = gist_forward(pm2, gist2, spans, conts)
    assert torch.isclose(l1, l2, atol=1e-5), f"resume mismatch: {l1} vs {l2}"


@pytest.mark.slow
def test_generate_from_gist_runs_and_respects_max_new():
    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    gen = generate_from_gist(pm, gist, [1, 2, 3], max_new=5)
    assert 1 <= len(gen) <= 5
    assert all(isinstance(t, int) for t in gen)


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
