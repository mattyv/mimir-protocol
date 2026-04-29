"""Mechanical invariants for soft-prompt-per-axiom training.

The thesis: model weights stay frozen; only a small per-axiom embedding
vector is trained. These tests assert the invariants — not the
experimental outcome (does the trained prompt produce good outputs).
"""

from __future__ import annotations

import hashlib

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_model():
    """Smallest Qwen we have cached for fast tests."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return model, tokenizer


def _checksum_state_dict(model: torch.nn.Module) -> str:
    """Hash the model's full state_dict to detect any weight change."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def test_soft_prompt_has_shape_matching_term_tokens(tiny_model):
    """Soft prompt should be shape [num_term_tokens, hidden_size]."""
    from marker.soft_prompt import SoftPrompt

    model, tokenizer = tiny_model
    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")

    expected_num_tokens = len(tokenizer("Balance Publisher", add_special_tokens=False).input_ids)
    assert sp.vector.shape[0] == expected_num_tokens
    assert sp.vector.shape[1] == model.config.hidden_size


def test_soft_prompt_initialized_from_term_embedding(tiny_model):
    """Initial soft prompt should equal the term's natural embedding."""
    from marker.soft_prompt import SoftPrompt

    model, tokenizer = tiny_model
    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")

    term_ids = tokenizer("Balance Publisher", add_special_tokens=False).input_ids
    embed = model.get_input_embeddings()
    expected = embed.weight[term_ids].detach()
    assert torch.allclose(sp.vector.detach(), expected, atol=1e-6)


def test_model_weights_unchanged_after_training(tiny_model):
    """After running training, model state_dict checksum must be identical."""
    from marker.soft_prompt import SoftPrompt, train_soft_prompt

    model, tokenizer = tiny_model
    before = _checksum_state_dict(model)

    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")
    paraphrases = [
        "Balance Publisher polls our crypto exchange's REST API.",
        "Balance Publisher publishes balances to the trading system.",
    ]
    train_soft_prompt(model, tokenizer, sp, paraphrases, n_steps=2, lr=0.01)

    after = _checksum_state_dict(model)
    assert before == after, "model weights changed during soft-prompt training"


def test_only_soft_prompt_has_grad_after_backward(tiny_model):
    """After loss.backward(), only the soft prompt parameter should have a
    .grad. Model parameters' .grad should remain None."""
    from marker.soft_prompt import SoftPrompt, _training_step

    model, tokenizer = tiny_model
    # Ensure model is frozen
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")

    paraphrase = "Balance Publisher polls the exchange and publishes balances."
    loss = _training_step(model, tokenizer, sp, paraphrase)
    loss.backward()

    assert sp.vector.grad is not None
    assert sp.vector.grad.norm().item() > 0
    # Model params should still have grad=None
    for name, p in model.named_parameters():
        assert p.grad is None, f"model param {name} got a gradient"


def test_one_training_step_decreases_loss(tiny_model):
    """One Adam step on the soft prompt should not increase loss
    (mechanical invariant — decrease is expected, equality at worst)."""
    from marker.soft_prompt import SoftPrompt, train_soft_prompt

    model, tokenizer = tiny_model

    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")
    paraphrases = [
        "Balance Publisher polls our crypto exchange's REST API every 250ms.",
        "Balance Publisher publishes sub-account balances to the trading system.",
    ]
    losses = train_soft_prompt(model, tokenizer, sp, paraphrases, n_steps=5, lr=0.05)
    assert losses[-1] <= losses[0] + 1e-3


def test_inference_hook_substitutes_at_term_positions(tiny_model):
    """When the inference hook is active and the prompt contains the term,
    the output of embed_tokens at the term positions should equal the
    soft prompt vector — not the natural embedding."""
    from marker.soft_prompt import SoftPrompt, install_soft_prompt_hook

    model, tokenizer = tiny_model
    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")
    # Resize the soft prompt to match how many tokens the term takes IN CONTEXT
    # (with leading space), since that's the position-count we'll substitute.
    from marker.soft_prompt import find_term_positions

    in_context = "What is a Balance Publisher?"
    pos_in_context = find_term_positions(tokenizer, in_context, "Balance Publisher")
    if len(pos_in_context) != sp.vector.shape[0]:
        # Re-init soft prompt at the in-context tokenization
        ids_in_context = tokenizer(in_context, add_special_tokens=False).input_ids
        term_subseq = ids_in_context[pos_in_context[0] : pos_in_context[-1] + 1]
        embed = model.get_input_embeddings()
        with torch.no_grad():
            init = embed.weight[term_subseq].detach().clone().float()
        sp.vector = torch.nn.Parameter(init)
        sp.term_token_ids = list(term_subseq)
    # Set soft prompt to a distinctive value so we can check substitution
    with torch.no_grad():
        sp.vector.copy_(torch.full_like(sp.vector, 7.5))

    from marker.soft_prompt import find_term_positions

    prompt = "What is a Balance Publisher?"
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    positions = find_term_positions(tokenizer, prompt, "Balance Publisher")
    assert positions, "find_term_positions should locate the term in the prompt"

    handle = install_soft_prompt_hook(model, sp, positions)
    try:
        embed = model.get_input_embeddings()
        with torch.no_grad():
            out = embed(ids)
        # Term positions should equal the soft prompt
        for j, pos in enumerate(positions):
            actual = out[0, pos]
            expected = sp.vector[j]
            assert torch.allclose(actual, expected.to(actual.dtype), atol=1e-5), (
                f"position {pos} not substituted"
            )
    finally:
        handle.remove()


def test_inference_hook_leaves_other_positions_unchanged(tiny_model):
    """The hook must not modify embeddings at non-term positions."""
    from marker.soft_prompt import SoftPrompt, find_term_positions, install_soft_prompt_hook

    model, tokenizer = tiny_model
    sp = SoftPrompt.from_term(model, tokenizer, term="Balance Publisher")

    prompt = "What is a Balance Publisher?"
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    positions = find_term_positions(tokenizer, prompt, "Balance Publisher")
    if len(positions) != sp.vector.shape[0]:
        ids_in_context = tokenizer(prompt, add_special_tokens=False).input_ids
        term_subseq = ids_in_context[positions[0] : positions[-1] + 1]
        embed = model.get_input_embeddings()
        with torch.no_grad():
            init = embed.weight[term_subseq].detach().clone().float()
        sp.vector = torch.nn.Parameter(init)
    with torch.no_grad():
        sp.vector.copy_(torch.full_like(sp.vector, 7.5))

    embed = model.get_input_embeddings()
    with torch.no_grad():
        baseline_out = embed(ids).clone()

    handle = install_soft_prompt_hook(model, sp, positions)
    try:
        with torch.no_grad():
            hooked_out = embed(ids)
        for pos in range(ids.shape[1]):
            if pos in positions:
                continue
            assert torch.allclose(hooked_out[0, pos], baseline_out[0, pos], atol=1e-6), (
                f"non-term position {pos} was modified"
            )
    finally:
        handle.remove()
