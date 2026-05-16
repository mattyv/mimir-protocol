"""Tests for token-trigger-based injection — the runtime path that does NOT
rely on user-facing markers. The model sees the user's free text; we scan the
tokenized stream for any registered axiom term and inject at those positions."""

from __future__ import annotations

import numpy as np
import pytest

from marker.trigger_inject import Registry, find_matches


def make_registry(name_to_variants: dict[str, list[list[int]]]) -> Registry:
    reg = Registry()
    for name, variants in name_to_variants.items():
        reg._add_term(name, variants, vector=np.zeros(4, dtype=np.float32))
    return reg


def test_find_matches_single_term():
    reg = make_registry({"foo": [[10, 20]]})
    matches = find_matches([1, 10, 20, 3], reg)
    assert matches == [(1, 3, "foo")]


def test_find_matches_multiple_occurrences():
    reg = make_registry({"foo": [[10, 20]]})
    matches = find_matches([10, 20, 5, 10, 20], reg)
    assert [(s, e) for s, e, _ in matches] == [(0, 2), (3, 5)]


def test_find_matches_no_match():
    reg = make_registry({"foo": [[10, 20]]})
    matches = find_matches([1, 2, 3], reg)
    assert matches == []


def test_find_matches_longest_wins_on_overlap():
    reg = make_registry({"foo": [[10, 20]], "foobar": [[10, 20, 30]]})
    matches = find_matches([10, 20, 30], reg)
    assert len(matches) == 1
    assert matches[0][2] == "foobar"


def test_find_matches_multiple_variants_same_term():
    reg = make_registry({"foo": [[10, 20], [11, 21]]})  # cased variants
    matches = find_matches([1, 11, 21, 5, 10, 20], reg)
    assert [(s, e, n) for s, e, n in matches] == [(1, 3, "foo"), (4, 6, "foo")]


def test_find_matches_empty_variant_skipped():
    reg = make_registry({"foo": [[]]})
    matches = find_matches([1, 2, 3], reg)
    assert matches == []


# ------------------------------------------------------------------------
# Mechanical-invariant tests for the KV-cache-aware generate path.
# These load a small real model (Qwen 0.5B) — slow but the only way to
# verify the hook + cache + decode loop interact correctly.
# ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_runner():  # noqa: ANN201
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marker.trigger_inject import Registry, TriggerInjector

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    # Empty registry with one term so the hook has something to scan for,
    # but the term won't appear in test prompts.
    reg = Registry()
    reg.register(
        "balance_publisher",
        term_variants=["Balance Publisher"],
        vector=np.zeros(model.config.hidden_size, dtype=np.float32),
        tokenizer=tok,
    )
    inj = TriggerInjector(model, tok, layer=17, registry=reg, alpha=0.0)
    return inj


def test_generate_alpha_zero_matches_unhooked_model(small_runner) -> None:  # noqa: ANN001
    """At alpha=0 the hook short-circuits. Output must equal what the bare
    model produces with the same greedy + KV-cache decode path."""
    import torch

    inj = small_runner
    prompt = "The capital of France is"

    # Unhooked greedy generation, KV cache enabled — equivalent to inj at α=0.
    device = next(inj.model.parameters()).device
    ids = inj.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = inj.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids_baseline = torch.cat([ids, nxt], dim=1)
        for _ in range(9):
            out = inj.model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids_baseline = torch.cat([ids_baseline, nxt], dim=1)
    baseline_text = inj.tokenizer.decode(ids_baseline[0], skip_special_tokens=True)[len(prompt) :]

    inj.alpha = 0.0
    hook_text = inj.generate(prompt, max_new_tokens=10)

    assert hook_text == baseline_text, (
        f"α=0 must be a no-op:\n  hook: {hook_text!r}\n  base: {baseline_text!r}"
    )


def test_generate_alpha_zero_is_deterministic(small_runner) -> None:  # noqa: ANN001
    """No state leaks between generate() calls."""
    inj = small_runner
    inj.alpha = 0.0
    a = inj.generate("Hello world. Today is", max_new_tokens=8)
    b = inj.generate("Hello world. Today is", max_new_tokens=8)
    assert a == b
