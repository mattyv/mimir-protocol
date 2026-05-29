"""Add a single learnable seed-token to a frozen base model.

The seed token (e.g. ``<BalancePublisher>``) is a new vocab entry whose
embedding is the only parameter allowed to receive gradient updates. Every
other model parameter — including every other embedding row — is frozen.

This is Phase 1 of the single-vector axiom architecture: the seed gives the
axiom a stable identity in residual space, and per-layer bolt-on adapters
(Phase 2, separate module) will be keyed to that identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenizers import AddedToken


def _term_token_ids(tokenizer, term: str) -> list[int]:  # noqa: ANN001
    """BPE pieces for ``term``, trying with and without a leading space."""
    ids = tokenizer(term, add_special_tokens=False).input_ids
    if not ids:
        ids = tokenizer(" " + term, add_special_tokens=False).input_ids
    return ids


@dataclass
class SeedToken:
    """Record of a registered seed token.

    The new embedding row IS ``model.get_input_embeddings().weight[token_id]``.
    We don't duplicate-store it; look it up via ``seed_embedding(model, seed)``.
    """

    name: str
    token_id: int
    original_term: str
    original_bpe_ids: list[int]


def seed_embedding(model, seed: SeedToken) -> torch.Tensor:  # noqa: ANN001
    """View into the new embedding row. Writeable; gradient flows into it."""
    return model.get_input_embeddings().weight[seed.token_id]


def _freeze_model_params(model) -> None:  # noqa: ANN001
    for p in model.parameters():
        p.requires_grad = False


def _install_seed_grad_mask(model, token_id: int) -> None:  # noqa: ANN001
    """Re-enable requires_grad on the embedding weight, then mask its grad
    so that only the row at ``token_id`` survives backward. Other rows'
    gradients are zeroed before accumulation.

    With ``tie_word_embeddings=True`` the lm_head shares storage with this
    same tensor, so the same hook covers both directions.
    """
    embed_w = model.get_input_embeddings().weight
    embed_w.requires_grad = True

    def hook(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked[token_id] = grad[token_id]
        return masked

    embed_w.register_hook(hook)


def register_seed_token(model, tokenizer, name: str) -> SeedToken:  # noqa: ANN001
    """Add ``<{name}>`` as a single new vocab entry, resize the model's
    embedding table to include one new row, initialize that row to the mean
    of ``name``'s original BPE-piece embeddings, freeze every other model
    parameter, and install a backward-hook gradient mask so only the new
    row can be trained.

    Returns a ``SeedToken`` record describing what was added.
    """
    original_bpe_ids = _term_token_ids(tokenizer, name)
    if not original_bpe_ids:
        raise ValueError(f"could not tokenize term {name!r}")

    token_str = f"<{name}>"
    added_token = AddedToken(token_str, special=False, normalized=False)
    tokenizer.add_tokens([added_token])
    token_id = tokenizer.convert_tokens_to_ids(token_str)

    # Qwen 2.5 pads its embedding rows past the logical tokenizer vocab for
    # GPU-alignment, so the new id often already falls inside the existing
    # table. Only resize if the new id genuinely overflows. Calling
    # resize_token_embeddings unconditionally would SHRINK the table.
    current_emb_rows = model.get_input_embeddings().weight.shape[0]
    if token_id >= current_emb_rows:
        model.resize_token_embeddings(token_id + 1)

    # Verify single-token encoding (added tokens are matched pre-BPE).
    encoded = tokenizer.encode(token_str, add_special_tokens=False)
    if encoded != [token_id]:
        raise RuntimeError(f"{token_str!r} did not encode as a single token: {encoded}")

    # Initialize the new row to the mean of the original term's BPE pieces.
    embed = model.get_input_embeddings()
    with torch.no_grad():
        init = embed.weight[original_bpe_ids].mean(dim=0)
        embed.weight[token_id] = init

    _freeze_model_params(model)
    _install_seed_grad_mask(model, token_id)

    return SeedToken(
        name=name,
        token_id=token_id,
        original_term=name,
        original_bpe_ids=original_bpe_ids,
    )
