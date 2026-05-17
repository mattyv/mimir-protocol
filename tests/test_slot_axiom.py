"""Mechanical invariants for per-axiom slot injection.

A SlotAxiom owns a contiguous range of dimensions in the residual stream
at one chosen layer. Its learnable vector is added to those dims at every
token position during the forward pass.

These tests assert the mechanical contract:
  - injection writes only to designated dims (non-slot dims unchanged)
  - model weights are not modified by hook attach/detach
  - multiple slots compose without interference (orthogonal by partition)
  - training a slot changes only the slot's parameter
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


# ---------- construction ----------


def test_slot_axiom_construction_shape(tiny_model):
    from marker.slot_axiom import SlotAxiom

    model, _ = tiny_model
    hidden = model.config.hidden_size
    sa = SlotAxiom.new(
        name="alpha", slot_start=0, slot_width=64, target_layer=10, hidden_size=hidden
    )
    assert sa.vector.shape == (64,)
    assert isinstance(sa.vector, torch.nn.Parameter)
    assert sa.slot_start == 0
    assert sa.slot_width == 64
    assert sa.target_layer == 10


def test_slot_axiom_init_is_zero():
    """Untrained slot should inject zero so it's a strict no-op vs vanilla."""
    from marker.slot_axiom import SlotAxiom

    sa = SlotAxiom.new(name="alpha", slot_start=0, slot_width=64, target_layer=10, hidden_size=512)
    assert torch.equal(sa.vector, torch.zeros(64))


# ---------- injection invariants ----------


def test_install_zero_slot_is_noop(tiny_model):
    """A zero-vector slot, when installed, must produce identical logits to vanilla."""
    from marker.slot_axiom import SlotAxiom, install_slot_hooks

    model, tok = tiny_model
    hidden = model.config.hidden_size
    ids = tok("hello world test sentence", return_tensors="pt", add_special_tokens=False).input_ids

    with torch.no_grad():
        ref = model(ids)

    sa = SlotAxiom.new(
        name="alpha", slot_start=0, slot_width=64, target_layer=5, hidden_size=hidden
    )
    handles = install_slot_hooks(model, [sa])
    try:
        with torch.no_grad():
            out = model(ids)
    finally:
        for h in handles:
            h.remove()
    assert torch.allclose(out.logits, ref.logits, atol=1e-5)


def test_install_nonzero_slot_changes_logits(tiny_model):
    """A non-zero slot vector must change the model's logits."""
    from marker.slot_axiom import SlotAxiom, install_slot_hooks

    model, tok = tiny_model
    hidden = model.config.hidden_size
    ids = tok("hello world test sentence", return_tensors="pt", add_special_tokens=False).input_ids

    with torch.no_grad():
        ref = model(ids)

    sa = SlotAxiom.new(
        name="alpha", slot_start=0, slot_width=64, target_layer=5, hidden_size=hidden
    )
    with torch.no_grad():
        sa.vector.copy_(torch.full((64,), 0.5))
    handles = install_slot_hooks(model, [sa])
    try:
        with torch.no_grad():
            out = model(ids)
    finally:
        for h in handles:
            h.remove()
    diff = (out.logits - ref.logits).abs().max().item()
    assert diff > 1e-3, f"slot injection had no effect (max diff = {diff})"


def test_install_does_not_change_weights(tiny_model):
    """Install / uninstall must leave model weights byte-identical."""
    from marker.slot_axiom import SlotAxiom, install_slot_hooks

    model, _ = tiny_model
    hidden = model.config.hidden_size
    before = _state_checksum(model)
    sa = SlotAxiom.new(
        name="alpha", slot_start=0, slot_width=64, target_layer=5, hidden_size=hidden
    )
    with torch.no_grad():
        sa.vector.copy_(torch.full((64,), 0.5))
    handles = install_slot_hooks(model, [sa])
    for h in handles:
        h.remove()
    after = _state_checksum(model)
    assert before == after


def test_slot_writes_only_to_designated_dims(tiny_model):
    """Capture residual stream BEFORE and AFTER the target layer; verify
    only dims [slot_start:slot_start+slot_width] differ, others are equal."""
    from marker.prefix_tuning import _get_layers
    from marker.slot_axiom import SlotAxiom, install_slot_hooks

    model, tok = tiny_model
    hidden = model.config.hidden_size
    ids = tok("hello world test", return_tensors="pt", add_special_tokens=False).input_ids
    layers = _get_layers(model)
    target_layer_idx = 5

    captured_ref: dict = {}
    captured_slot: dict = {}

    def make_capture(d: dict):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            d["after"] = tensor.detach().clone()

        return hook

    # Capture vanilla output of target layer
    h1 = layers[target_layer_idx].register_forward_hook(make_capture(captured_ref))
    with torch.no_grad():
        model(ids)
    h1.remove()

    # Now install slot at the same target layer, capture again
    sa = SlotAxiom.new(
        name="alpha",
        slot_start=10,
        slot_width=32,
        target_layer=target_layer_idx,
        hidden_size=hidden,
    )
    with torch.no_grad():
        sa.vector.copy_(torch.full((32,), 2.5))
    slot_handles = install_slot_hooks(model, [sa])
    h2 = layers[target_layer_idx].register_forward_hook(make_capture(captured_slot))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h2.remove()
        for h in slot_handles:
            h.remove()

    diff = captured_slot["after"] - captured_ref["after"]
    # Inside the slot: should differ by ~2.5 at every token position
    inside = diff[..., 10:42]
    outside_left = diff[..., :10]
    outside_right = diff[..., 42:]
    assert torch.allclose(outside_left, torch.zeros_like(outside_left), atol=1e-6), (
        "dims before slot were modified"
    )
    assert torch.allclose(outside_right, torch.zeros_like(outside_right), atol=1e-6), (
        "dims after slot were modified"
    )
    assert torch.allclose(inside, torch.full_like(inside, 2.5), atol=1e-5), (
        "slot dims did not receive expected offset"
    )


# ---------- multi-axiom non-interference ----------


def test_two_slots_dont_interfere(tiny_model):
    """Two slots in disjoint dim ranges each touch only their own dims."""
    from marker.prefix_tuning import _get_layers
    from marker.slot_axiom import SlotAxiom, install_slot_hooks

    model, tok = tiny_model
    hidden = model.config.hidden_size
    ids = tok("hello world", return_tensors="pt", add_special_tokens=False).input_ids
    layers = _get_layers(model)
    layer_idx = 5

    sa1 = SlotAxiom.new(
        name="alpha", slot_start=0, slot_width=32, target_layer=layer_idx, hidden_size=hidden
    )
    sa2 = SlotAxiom.new(
        name="beta", slot_start=64, slot_width=32, target_layer=layer_idx, hidden_size=hidden
    )
    with torch.no_grad():
        sa1.vector.copy_(torch.full((32,), 1.0))
        sa2.vector.copy_(torch.full((32,), 3.0))

    captured_ref: dict = {}
    captured_both: dict = {}

    def make_capture(d: dict):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            d["after"] = tensor.detach().clone()

        return hook

    h1 = layers[layer_idx].register_forward_hook(make_capture(captured_ref))
    with torch.no_grad():
        model(ids)
    h1.remove()

    handles = install_slot_hooks(model, [sa1, sa2])
    h2 = layers[layer_idx].register_forward_hook(make_capture(captured_both))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h2.remove()
        for h in handles:
            h.remove()

    diff = captured_both["after"] - captured_ref["after"]
    # Slot 1 [0..32]: should be +1
    assert torch.allclose(diff[..., :32], torch.full_like(diff[..., :32], 1.0), atol=1e-5)
    # Gap [32..64]: unchanged
    assert torch.allclose(diff[..., 32:64], torch.zeros_like(diff[..., 32:64]), atol=1e-6)
    # Slot 2 [64..96]: should be +3
    assert torch.allclose(diff[..., 64:96], torch.full_like(diff[..., 64:96], 3.0), atol=1e-5)
    # After both slots [96..]: unchanged
    assert torch.allclose(diff[..., 96:], torch.zeros_like(diff[..., 96:]), atol=1e-6)


# ---------- training ----------


def test_training_reduces_loss(tiny_model):
    """Training a slot on a short description should reduce the loss."""
    from marker.slot_axiom import SlotAxiom, train_slot

    model, tok = tiny_model
    hidden = model.config.hidden_size

    sa = SlotAxiom.new(
        name="alpha",
        slot_start=0,
        slot_width=256,
        target_layer=model.config.num_hidden_layers // 2,
        hidden_size=hidden,
    )
    description = "Flurgan is a microservice that polls every 11 milliseconds."
    losses = train_slot(model, tok, sa, description, n_steps=20, lr=0.05)
    assert losses[-1] < losses[0] - 0.1, (
        f"loss didn't decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


def test_qa_training_reduces_loss(tiny_model):
    """Training on Q+A pairs reduces the loss."""
    from marker.slot_axiom import SlotAxiom, train_slot_qa

    model, tok = tiny_model
    hidden = model.config.hidden_size
    sa = SlotAxiom.new(
        name="alpha",
        slot_start=0,
        slot_width=256,
        target_layer=model.config.num_hidden_layers // 2,
        hidden_size=hidden,
    )
    qa = [
        ("How often does Flurgan poll?", "Every 11 milliseconds."),
        ("What does Flurgan do?", "Flurgan polls every 11 milliseconds."),
    ]
    losses = train_slot_qa(model, tok, sa, qa, n_steps=20, lr=0.05)
    assert losses[-1] < losses[0] - 0.1, (
        f"loss didn't decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


def test_training_does_not_change_model_weights(tiny_model):
    """After training a slot, the model's state_dict is unchanged."""
    from marker.slot_axiom import SlotAxiom, train_slot

    model, tok = tiny_model
    hidden = model.config.hidden_size
    before = _state_checksum(model)
    sa = SlotAxiom.new(
        name="alpha",
        slot_start=0,
        slot_width=128,
        target_layer=5,
        hidden_size=hidden,
    )
    train_slot(model, tok, sa, "Flurgan polls every 11 ms.", n_steps=5, lr=0.05)
    after = _state_checksum(model)
    assert before == after


# ---------- batched soft prompt training equivalence ----------


def test_batched_soft_prompt_bs1_matches_v5(tiny_model):
    """Batched trainer at batch_size=1 + n_steps=5 should produce a
    trained vector arbitrarily close to the unbatched v5 trainer."""
    import random

    import torch as _torch

    from marker.soft_prompt_plus import (
        SoftPromptPlus,
        train_soft_prompt_plus_qa_v5,
        train_soft_prompt_plus_qa_v6_batched,
    )

    model, tok = tiny_model
    qa = [
        ("How often does Flurgan poll?", "Every 11 milliseconds."),
        ("What does Flurgan do?", "Flurgan polls every 11 milliseconds."),
    ]

    # v5 (unbatched)
    _torch.manual_seed(42)
    random.seed(0)
    sp_v5 = SoftPromptPlus.from_term(model, tok, term="Flurgan", n_ghost=4)
    train_soft_prompt_plus_qa_v5(
        model,
        tok,
        sp_v5,
        qa,
        n_steps=5,
        lr_start=0.01,
        lr_end=0.01,
        append_eos=True,
        norm_anchor_lambda=0.0,
    )

    # v6 batched, bs=1
    _torch.manual_seed(42)
    random.seed(0)
    sp_v6 = SoftPromptPlus.from_term(model, tok, term="Flurgan", n_ghost=4)
    train_soft_prompt_plus_qa_v6_batched(
        model,
        tok,
        sp_v6,
        qa,
        n_steps=5,
        batch_size=1,
        lr_start=0.01,
        lr_end=0.01,
        append_eos=True,
        norm_anchor_lambda=0.0,
    )

    # The two trained vectors should be close (not exact — different code paths
    # may have small fp differences, but functionally equivalent).
    diff = (sp_v5.vector.detach() - sp_v6.vector.detach()).abs().max().item()
    assert diff < 0.1, f"v5 vs v6@bs=1 diverged: max abs diff = {diff:.4f}"
