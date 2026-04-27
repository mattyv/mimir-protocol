"""Tests for AutoInjector — the multi-hook runtime that consumes a list of
AxiomPlan and applies each plan's mechanism stack at runtime.

Mechanical-invariant tests use a small real model (Qwen 0.5B) to verify
the hook + KV-cache + decode loop interact correctly across multiple
hooks. Slow but the only way to be confident."""

from __future__ import annotations

import numpy as np
import pytest

from marker.axiom_classifier import LexicalPrior
from marker.axiom_plan import AxiomPlan


@pytest.fixture(scope="module")
def small_model():  # noqa: ANN201
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    return model, tok


def test_auto_injector_empty_plans_is_noop(small_model) -> None:  # noqa: ANN001
    """No plans -> generate() must equal bare-model greedy decode."""
    import torch

    from marker.auto_injector import AutoInjector

    model, tok = small_model
    inj = AutoInjector(model, tok, plans=[])
    prompt = "The capital of France is"

    # Bare baseline
    device = next(model.parameters()).device
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids_baseline = torch.cat([ids, nxt], dim=1)
        for _ in range(9):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids_baseline = torch.cat([ids_baseline, nxt], dim=1)
    baseline = tok.decode(ids_baseline[0], skip_special_tokens=True)[len(prompt) :]

    out_text = inj.generate(prompt, max_new_tokens=10)
    assert out_text == baseline, (
        f"empty AutoInjector must be a no-op:\n hook: {out_text!r}\n base: {baseline!r}"
    )


def test_auto_injector_zero_alpha_is_noop(small_model) -> None:  # noqa: ANN001
    """A plan with α=0 across every mechanism must equal bare-model decode."""
    import torch

    from marker.auto_injector import AutoInjector

    model, tok = small_model
    hidden_size = model.config.hidden_size
    plan = AxiomPlan(
        term="balance_publisher",
        term_variants=["Balance Publisher"],
        lexical_prior=LexicalPrior.HIGH,
        complexity=1,
        mechanisms={
            "eop": {"layer": 17, "alpha": 0.0, "vector": np.ones(hidden_size, dtype=np.float32)},
            "steer": {"layer": 22, "alpha": 0.0, "vector": np.ones(hidden_size, dtype=np.float32)},
        },
        target_tokens=["experience"],
    )
    inj = AutoInjector(model, tok, plans=[plan])
    prompt = "The capital of France is"

    # Bare baseline
    device = next(model.parameters()).device
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids_baseline = torch.cat([ids, nxt], dim=1)
        for _ in range(9):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids_baseline = torch.cat([ids_baseline, nxt], dim=1)
    baseline = tok.decode(ids_baseline[0], skip_special_tokens=True)[len(prompt) :]

    out_text = inj.generate(prompt, max_new_tokens=10)
    assert out_text == baseline


def test_auto_injector_term_not_in_prompt_is_noop(small_model) -> None:  # noqa: ANN001
    """The term doesn't appear in the prompt → no positions match → injection
    short-circuits → output equals bare model."""
    import torch

    from marker.auto_injector import AutoInjector

    model, tok = small_model
    hidden_size = model.config.hidden_size
    # Non-zero alpha and vector, but the term doesn't appear in this prompt.
    plan = AxiomPlan(
        term="quibblefishbloop",
        term_variants=["quibblefishbloop"],
        lexical_prior=LexicalPrior.LOW,
        complexity=1,
        mechanisms={
            "eop": {
                "layer": 17,
                "alpha": 30.0,
                "vector": np.ones(hidden_size, dtype=np.float32) / np.sqrt(hidden_size),
            },
        },
        target_tokens=[],
    )
    inj = AutoInjector(model, tok, plans=[plan])
    prompt = "Hello world. Today is"

    device = next(model.parameters()).device
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids_baseline = torch.cat([ids, nxt], dim=1)
        for _ in range(7):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids_baseline = torch.cat([ids_baseline, nxt], dim=1)
    baseline = tok.decode(ids_baseline[0], skip_special_tokens=True)[len(prompt) :]

    out_text = inj.generate(prompt, max_new_tokens=8)
    assert out_text == baseline


def test_auto_injector_deterministic(small_model) -> None:  # noqa: ANN001
    """No hidden state should leak between generate() calls."""
    from marker.auto_injector import AutoInjector

    model, tok = small_model
    inj = AutoInjector(model, tok, plans=[])
    a = inj.generate("Hello world. Today is", max_new_tokens=8)
    b = inj.generate("Hello world. Today is", max_new_tokens=8)
    assert a == b
