"""Mechanical invariants for SeedToken registration on a Qwen base model.

The seed-token approach adds a single new vocab entry (e.g. `<BalancePublisher>`)
with a learnable embedding. The rest of the model — every other embedding row,
every layer's weights — stays frozen. These tests assert those invariants;
they do not measure whether training the seed produces useful behavior.
"""

from __future__ import annotations

import hashlib

import pytest
import torch


@pytest.fixture
def tiny_model():
    """Smallest Qwen for fast CPU tests. Function-scoped because
    `register_seed_token` mutates both the tokenizer and the model
    (resize_token_embeddings), and tests must not bleed into each other."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load {name}: {e}")
    return model, tokenizer


def _checksum_non_embedding_params(model: torch.nn.Module) -> str:
    """Hash the model's params, excluding the embedding/lm_head rows.
    Used to assert nothing else in the model changed during registration."""
    embed = model.get_input_embeddings()
    embed_id = id(embed.weight)
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if id(p) == embed_id:
            continue
        # lm_head shares storage with embed when tied; skip too
        if name.endswith("lm_head.weight") and p.data_ptr() == embed.weight.data_ptr():
            continue
        h.update(name.encode())
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def test_seed_token_is_single_token(tiny_model):
    """After registration, `<Name>` must encode to exactly one id."""
    from marker.seed_token import register_seed_token

    model, tokenizer = tiny_model
    seed = register_seed_token(model, tokenizer, "BalancePublisher")

    ids = tokenizer.encode("<BalancePublisher>", add_special_tokens=False)
    assert len(ids) == 1, f"expected single-token encoding, got {ids}"
    assert ids[0] == seed.token_id


def test_seed_token_initialized_to_bpe_mean(tiny_model):
    """The new embedding row should equal the mean of the original term's
    BPE-piece embeddings (so the seed starts at a sensible prior)."""
    from marker.seed_token import register_seed_token

    model, tokenizer = tiny_model
    seed = register_seed_token(model, tokenizer, "BalancePublisher")

    embed = model.get_input_embeddings()
    expected = embed.weight[seed.original_bpe_ids].mean(dim=0)
    actual = embed.weight[seed.token_id]
    assert torch.allclose(actual, expected, atol=1e-6), (
        f"new row not at BPE mean: max diff {(actual - expected).abs().max().item()}"
    )


def test_logits_unchanged_for_old_vocab(tiny_model):
    """A prompt that does not use the seed token should produce identical
    logits over the original *logical* vocab range before vs. after
    registration. Qwen's lm_head has padding rows beyond the logical vocab
    that are not addressable as tokens; one of those rows becomes the seed's
    home, and its column in the output logits is allowed to change."""
    from marker.seed_token import register_seed_token

    model, tokenizer = tiny_model

    # Use a short fixed prompt that doesn't contain anything resembling the term.
    prompt_ids = tokenizer("The cat sat on the mat.", add_special_tokens=False).input_ids
    inp = torch.tensor([prompt_ids])

    # Capture the original *logical* vocab size — the largest token id any
    # user could have produced from this tokenizer before we touched it.
    old_logical_vocab = len(tokenizer)

    with torch.no_grad():
        logits_before = model(inp).logits.clone()

    register_seed_token(model, tokenizer, "BalancePublisher")

    with torch.no_grad():
        logits_after = model(inp).logits

    assert torch.equal(
        logits_before[..., :old_logical_vocab],
        logits_after[..., :old_logical_vocab],
    ), "logits over original logical vocab changed after registration"


def test_only_seed_row_receives_gradient(tiny_model):
    """Backprop on a loss involving the seed token must leave gradient ONLY
    on the seed row of the embedding (and on the seed's row of the tied
    lm_head, which is the same tensor). Every other parameter must end up
    with grad=None or all-zero grad."""
    from marker.seed_token import register_seed_token

    model, tokenizer = tiny_model
    seed = register_seed_token(model, tokenizer, "BalancePublisher")

    # Single-token input of just the seed.
    inp = torch.tensor([[seed.token_id]])
    out = model(inp)
    out.logits.sum().backward()

    embed = model.get_input_embeddings()
    grad = embed.weight.grad
    assert grad is not None, "embedding has no grad after backward"

    # Seed row should have nonzero grad
    seed_grad = grad[seed.token_id]
    assert seed_grad.abs().sum().item() > 0, "seed row grad is zero"

    # A handful of other rows must have zero grad
    for other_id in (0, 1, 100, 1000):
        if other_id == seed.token_id or other_id >= grad.shape[0]:
            continue
        other = grad[other_id]
        assert other.abs().sum().item() == 0, (
            f"row {other_id} got nonzero grad (sum={other.abs().sum().item()})"
        )

    # Every other parameter must have no grad signal at all
    embed_ptr = embed.weight.data_ptr()
    for name, p in model.named_parameters():
        if p.data_ptr() == embed_ptr:
            continue
        if p.grad is None:
            continue
        assert p.grad.abs().sum().item() == 0, f"parameter {name} got nonzero grad after backward"


def test_state_dict_checksum_outside_seed(tiny_model):
    """No non-embedding parameter should change as a side effect of
    registering the seed token."""
    from marker.seed_token import register_seed_token

    model, tokenizer = tiny_model
    checksum_before = _checksum_non_embedding_params(model)

    register_seed_token(model, tokenizer, "BalancePublisher")

    checksum_after = _checksum_non_embedding_params(model)
    assert checksum_before == checksum_after, (
        "non-embedding parameters changed during seed-token registration"
    )


def test_tied_embeddings_preserved(tiny_model):
    """Qwen 2.5 ties input embeddings and lm_head. After resize, they must
    still share storage — otherwise the lm_head row for the seed token will
    be uninitialized and predictions of the seed token will be broken."""
    from marker.seed_token import register_seed_token

    model, tokenizer = tiny_model

    # Sanity: model claims tied embeddings
    assert getattr(model.config, "tie_word_embeddings", False), (
        "fixture model does not have tied embeddings; test assumption broken"
    )

    register_seed_token(model, tokenizer, "BalancePublisher")

    in_w = model.get_input_embeddings().weight
    out_w = model.get_output_embeddings().weight
    assert in_w.data_ptr() == out_w.data_ptr(), (
        "lm_head and input embedding no longer share storage after resize"
    )
