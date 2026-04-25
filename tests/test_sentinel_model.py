"""Mechanical invariants for the sentinel-LoRA model wrapper.

Per CLAUDE.md, these tests assert that the plumbing is wired (model loads,
LoRA wraps, generation round-trips) — not the experiment outcome (whether
the LoRA learns the protocol). The latter lives in eval artifacts.

Slow: Qwen 2.5 0.5B is ~1.2 GB, downloading on first run. After that the
HF cache makes these tests <30s.
"""

from __future__ import annotations

import pytest
import torch

from sentinel.model import SentinelModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture(scope="module")
def model() -> SentinelModel:
    return SentinelModel(model_name="Qwen/Qwen2.5-0.5B", device=DEVICE)


def test_model_loads_on_device(model: SentinelModel) -> None:
    assert model.device == DEVICE
    # Qwen2.5-0.5B has 24 hidden layers, hidden_size 896.
    assert model.config.hidden_size == 896
    assert model.config.num_hidden_layers == 24


def test_model_baseline_generation_round_trips(model: SentinelModel) -> None:
    out = model.generate("The capital of France is", max_new_tokens=4)
    assert isinstance(out, str)
    assert out.startswith("The capital of France is")
    # Trivial sanity: the model produced *something* novel after the prompt.
    assert len(out) > len("The capital of France is")


def test_lora_wraps_with_expected_trainable_param_count(model: SentinelModel) -> None:
    """Rank-16 LoRA on attn (q,k,v,o) + FFN (gate,up,down) of Qwen2.5-0.5B.
    Trainable parameters should be a small fraction of the base model."""
    base_total = sum(p.numel() for p in model.base.parameters())
    wrapped = model.with_lora(rank=16, alpha=32)
    trainable = sum(p.numel() for p in wrapped.peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.peft_model.parameters())

    assert trainable > 0, "no trainable params after LoRA wrap"
    # LoRA adapters should be << 5% of the base model.
    ratio = trainable / total
    assert ratio < 0.05, f"trainable ratio {ratio:.4f} too high — LoRA didn't freeze base"
    # And nontrivial — at least 0.05% — otherwise targets list is empty.
    assert ratio > 0.0005, f"trainable ratio {ratio:.4f} too low — LoRA targets missed"

    # Base weights must dominate (frozen params >> trainable LoRA params).
    frozen_total = sum(p.numel() for p in wrapped.peft_model.parameters() if not p.requires_grad)
    assert frozen_total > 0.9 * base_total, "too many base params are trainable post-wrap"


def test_lora_wrapped_model_still_generates(model: SentinelModel) -> None:
    """Untrained LoRA (zero-init B matrix) must not change baseline behavior."""
    base_out = model.generate("Once upon a time", max_new_tokens=5)
    wrapped = model.with_lora(rank=16, alpha=32)
    wrapped_out = wrapped.generate("Once upon a time", max_new_tokens=5)
    # Untrained LoRA initialises B to zeros, so adapter contribution is zero.
    # Greedy generation should match exactly.
    assert base_out == wrapped_out, (
        f"untrained LoRA changed greedy output:\n  base: {base_out!r}\n  wrapped: {wrapped_out!r}"
    )
