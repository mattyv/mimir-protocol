"""Value tokens: promote a novel identifier to a single emittable vocab token.

The single-vector architecture can't emit a multi-token novel string like
``balances.raw`` (no prior, multi-piece). The fix is to register that string
as ONE atomic vocab token whose decode renders the literal identifier, so the
bolt only has to win a single-token prediction.

These tests pin the mechanical invariants (single token, literal decode,
BPE-mean init, and — critically — that registering a seed plus its value
tokens installs ONE grad mask that lets exactly those rows train and nothing
else). They do not assert the experiment's quality.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def model_tok():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:  # pragma: no cover - network/cache dependent
        pytest.skip(f"could not load {name}: {e}")
    return model, tok


def test_value_token_decodes_to_literal_surface(model_tok):
    """The value token renders the bare identifier on decode — no angle
    brackets — so emitting it produces human-readable output."""
    from marker.value_token import register_value_token

    model, tok = model_tok
    vt = register_value_token(model, tok, "balances.raw")

    decoded = tok.decode([vt.token_id])
    assert "balances.raw" in decoded
    assert "<" not in decoded and ">" not in decoded


def test_value_token_embedding_is_bpe_mean(model_tok):
    """The new row inits to the mean of the surface's original BPE pieces —
    a sensible starting point near the literal string's meaning."""
    from marker.value_token import register_value_token

    model, tok = model_tok
    embed = model.get_input_embeddings()
    surface = "warehouse.fluxom_ingested"
    bpe_ids = tok(surface, add_special_tokens=False).input_ids
    expected = embed.weight[bpe_ids].mean(dim=0).detach().clone()

    vt = register_value_token(model, tok, surface)
    got = embed.weight[vt.token_id].detach()

    assert torch.allclose(got, expected, atol=1e-6)


def test_tokenizer_encodes_value_surface_as_single_token(model_tok):
    """After registration the tokenizer encodes the bare surface string as
    exactly one token id — no explicit answer-substitution needed at training
    time; the tokenizer handles it automatically."""
    from marker.value_token import register_axiom_tokens

    model, tok = model_tok
    axiom = register_axiom_tokens(model, tok, "BalancePublisher", ["balances.raw"])
    vt = axiom.values[0]

    # Plain tokenization of a sentence containing the surface should yield the
    # value token id at exactly the right position.
    sentence = "Publishes to the Kafka topic balances.raw every 250ms."
    ids = tok(sentence, add_special_tokens=False).input_ids
    assert vt.token_id in ids, f"value token {vt.token_id} not in {ids}"

    # And it should be a single id, not split.
    count = ids.count(vt.token_id)
    assert count == 1


def test_register_axiom_masks_only_seed_and_values(model_tok):
    """Seed + value tokens registered together must share ONE grad mask:
    exactly those rows receive gradient, every other embedding row stays
    exactly zero. This is the regression guard for the stacking bug where
    chained per-token masks zero everything."""
    from marker.value_token import register_axiom_tokens

    model, tok = model_tok
    axiom = register_axiom_tokens(
        model,
        tok,
        "BalancePublisher",
        ["balances.raw", "warehouse.fluxom_ingested"],
    )
    trainable = axiom.trainable_ids
    assert len(trainable) == 3

    embed_w = model.get_input_embeddings().weight
    ids = torch.tensor([[axiom.seed.token_id] + [v.token_id for v in axiom.values]])
    out = model(ids, labels=ids)
    out.loss.backward()

    assert embed_w.grad is not None
    vocab = embed_w.shape[0]
    keep = torch.zeros(vocab, dtype=torch.bool)
    keep[torch.tensor(trainable)] = True

    # Trainable rows got real gradient; everything else is exactly zero.
    assert embed_w.grad[keep].abs().sum() > 0
    assert embed_w.grad[~keep].abs().max() == 0


@pytest.mark.slow
def test_bolt_emits_value_token_after_training(model_tok):
    """Smoke: train bolt + seed + value token on a tiny QA set for 400 steps
    at a mid-stack layer only. Greedy decode must contain the value token
    (the literal surface 'balances.raw') in the answer span.

    This is the key end-to-end invariant: the bolt learns to route
    'which Kafka topic?' -> the value token, not a hallucinated paraphrase.
    """
    from marker.bolt_selector import install_bolt_hooks, make_bolt_selector, remove_bolt_hooks
    from marker.value_token import register_axiom_tokens, train_axiom_tokens

    model, tok = model_tok
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2

    axiom = register_axiom_tokens(model, tok, "BalancePublisher", ["balances.raw"])
    bolt = make_bolt_selector(model, axiom.seed, r=32, skill_mode=False, layers=[mid])

    qa_pairs = [
        ("<BalancePublisher> publishes to which Kafka topic?", "balances.raw"),
        ("Which topic does <BalancePublisher> emit to?", "The Kafka topic balances.raw"),
        ("Where does <BalancePublisher> land its events?", "On the topic balances.raw"),
    ]

    train_axiom_tokens(model, tok, axiom, bolt, qa_pairs, n_steps=400, lr=5e-4)

    prompt = "Q: <BalancePublisher> publishes to which Kafka topic?\nA:"
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    handles = install_bolt_hooks(model, bolt)
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=10, do_sample=False)
    finally:
        remove_bolt_hooks(handles)

    generated = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    assert "balances.raw" in generated, f"value token not in output: {generated!r}"
