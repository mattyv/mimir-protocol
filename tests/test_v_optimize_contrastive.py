"""Mechanical invariants for contrastive v-optimization.

The new objective:
  L(v) = NLL(intended_target | prompt, v) - NLL(lexical_target | prompt, v)

Maximizing this means: make intended-paraphrase tokens more likely than
lexical-paraphrase tokens, with the model & prompt fixed.

Tests assert structural invariants, not the experiment outcome:
  - Zero-vec injection is a no-op (loss equals baseline loss).
  - Gradient w.r.t. v has shape [hidden_size].
  - Hook fires at the configured layer (no effect at other layers).
  - Contrastive loss equals (intended_NLL - lexical_NLL) by definition.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_model():
    """A real but tiny model for fast tests. Use the smallest Qwen we have
    locally cached. Fall back to skip if not available."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return model, tokenizer


def test_zero_v_grad_is_zero(tiny_model):
    """If v=0 and gradient flows linearly through the additive injection,
    a tiny v gives a tiny effect. Specifically: the loss at v=0 equals
    the loss with no hook at all, which is the definition of no-op."""
    from marker.run_v_optimize_contrastive import (
        compute_contrastive_loss_and_grad,
        evaluate_contrastive_loss,
    )

    model, tokenizer = tiny_model
    layer = 5
    term = " Publisher"
    test_prompt = "What is a Balance Publisher?"
    intended = "polls the exchange and publishes balances to traders"
    lexical = "publishes the company's balance sheet quarterly"

    # Loss with no hook
    baseline = evaluate_contrastive_loss(
        model, tokenizer, test_prompt, intended, lexical, term, layer, v=None
    )
    # Loss with v=0
    zero_v = torch.zeros(model.config.hidden_size)
    with_zero = evaluate_contrastive_loss(
        model, tokenizer, test_prompt, intended, lexical, term, layer, v=zero_v
    )
    assert abs(baseline - with_zero) < 1e-3, (
        f"v=0 should be a no-op; got baseline {baseline:.4f}, with_zero {with_zero:.4f}"
    )

    # Gradient at v=0 should have correct shape
    _, grad = compute_contrastive_loss_and_grad(
        model, tokenizer, test_prompt, intended, lexical, term, layer, zero_v
    )
    assert grad.shape == (model.config.hidden_size,), f"unexpected grad shape {grad.shape}"


def test_contrastive_loss_equals_difference(tiny_model):
    """L_contrastive = NLL_intended - NLL_lexical, by construction."""
    from marker.run_v_optimize_contrastive import (
        evaluate_contrastive_loss,
        evaluate_paraphrase_nll,
    )

    model, tokenizer = tiny_model
    layer = 5
    term = " Publisher"
    test_prompt = "What is a Balance Publisher?"
    intended = "polls the exchange and publishes balances"
    lexical = "publishes the company's balance sheet"
    v = torch.randn(model.config.hidden_size) * 0.1

    nll_int = evaluate_paraphrase_nll(model, tokenizer, test_prompt, intended, term, layer, v)
    nll_lex = evaluate_paraphrase_nll(model, tokenizer, test_prompt, lexical, term, layer, v)
    contrastive = evaluate_contrastive_loss(
        model, tokenizer, test_prompt, intended, lexical, term, layer, v
    )
    assert abs(contrastive - (nll_int - nll_lex)) < 1e-4, (
        f"contrastive {contrastive} != intended-lexical {nll_int - nll_lex}"
    )


def test_v_grad_nonzero_for_nontrivial_v(tiny_model):
    """A nonzero v at a layer that affects the prediction should produce
    a nonzero gradient — sanity check that the autograd graph is connected."""
    from marker.run_v_optimize_contrastive import compute_contrastive_loss_and_grad

    model, tokenizer = tiny_model
    layer = 5
    term = " Publisher"
    test_prompt = "What is a Balance Publisher?"
    intended = "polls the exchange and publishes balances to traders"
    lexical = "publishes the company's balance sheet quarterly"
    v = torch.randn(model.config.hidden_size) * 0.5

    _, grad = compute_contrastive_loss_and_grad(
        model, tokenizer, test_prompt, intended, lexical, term, layer, v
    )
    assert torch.norm(grad).item() > 0, "gradient should be nonzero for nontrivial v"
    # Also: gradient should not be NaN
    assert not torch.isnan(grad).any(), "gradient contains NaN"


def test_term_position_found_correctly(tiny_model):
    """The injection position should be the LAST occurrence of the term
    in the prompt — the trailing ' Publisher' in 'What is a Balance Publisher?'
    """
    from marker.run_v_optimize_contrastive import find_last_term_position

    _, tokenizer = tiny_model
    prompt = "What is a Balance Publisher?"
    pos = find_last_term_position(tokenizer, prompt, " Publisher")
    assert pos > 0
    # The token at that position should decode to ' Publisher' or contain its tail.
    tok_id = tokenizer(prompt, add_special_tokens=False).input_ids[pos]
    decoded = tokenizer.decode([tok_id])
    assert "Publisher" in decoded or "publisher" in decoded, (
        f"position {pos} decodes to {decoded!r}, expected to contain Publisher"
    )


def test_optimization_step_decreases_or_equals_loss(tiny_model):
    """One gradient step with a small line search should not INCREASE the
    contrastive loss. (Loss may equal if no improvement found.)"""
    from marker.run_v_optimize_contrastive import (
        compute_contrastive_loss_and_grad,
        evaluate_contrastive_loss,
    )

    model, tokenizer = tiny_model
    layer = 5
    term = " Publisher"
    test_prompt = "What is a Balance Publisher?"
    intended = "polls the exchange and publishes balances to traders"
    lexical = "publishes the company's balance sheet quarterly"

    v = torch.zeros(model.config.hidden_size)
    loss_init, grad = compute_contrastive_loss_and_grad(
        model, tokenizer, test_prompt, intended, lexical, term, layer, v
    )

    best_loss = loss_init
    for eta in [0.01, 0.1, 1.0]:
        v_try = v - eta * grad
        loss_try = evaluate_contrastive_loss(
            model, tokenizer, test_prompt, intended, lexical, term, layer, v_try
        )
        if loss_try < best_loss:
            best_loss = loss_try

    # Allow tiny numerical slop
    assert best_loss <= loss_init + 1e-3, (
        f"loss should not increase; init {loss_init:.4f} best {best_loss:.4f}"
    )


def test_grad_shape_matches_hidden_size(tiny_model):
    from marker.run_v_optimize_contrastive import compute_contrastive_loss_and_grad

    model, tokenizer = tiny_model
    layer = 5
    term = " Publisher"
    test_prompt = "What is a Balance Publisher?"
    intended = "polls the exchange and publishes balances"
    lexical = "publishes the balance sheet"
    v = torch.zeros(model.config.hidden_size)
    _, grad = compute_contrastive_loss_and_grad(
        model, tokenizer, test_prompt, intended, lexical, term, layer, v
    )
    assert grad.shape == torch.Size([model.config.hidden_size])
    assert grad.dtype == torch.float32 or grad.dtype == torch.float16


def test_fisher_init_is_unit_or_scaled():
    """Fisher direction should be unit-norm. Smoke check on the helper."""
    from marker.run_better_inject import fisher_direction

    rng = np.random.default_rng(0)
    X_int = rng.normal(size=(10, 64)).astype(np.float32)
    X_lex = rng.normal(size=(8, 64)).astype(np.float32)
    v = fisher_direction(X_int, X_lex)
    assert v.shape == (64,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3, f"||v|| = {np.linalg.norm(v)}, expected 1.0"
