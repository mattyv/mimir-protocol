"""Value tokens: promote novel identifiers to single emittable vocab tokens.

A value token registers a bare string like ``balances.raw`` as one atomic
vocab entry whose decode renders the literal surface (no angle brackets).
Combined with the seed token (subject identity) and the bolt-on selector
(routing), this lets the bolt win a single-token prediction instead of having
to drive an out-of-distribution multi-piece sequence.

The key invariant: seed + all value tokens share ONE grad mask, installed
here. The earlier single-per-token mask approach stacked hooks and zeroed
everything on the second registration. ``register_axiom_tokens`` installs
exactly one hook covering all trainable ids.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from tokenizers import AddedToken

from marker.bolt_selector import (
    BoltSelector,
    bolt_parameters,
    install_bolt_hooks,
    remove_bolt_hooks,
)
from marker.seed_token import SeedToken, _freeze_model_params, _term_token_ids


@dataclass
class ValueToken:
    """A single registered value token."""

    surface: str
    token_id: int
    original_bpe_ids: list[int]


@dataclass
class AxiomTokens:
    """Seed + value tokens for one axiom, with a shared grad mask."""

    seed: SeedToken
    values: list[ValueToken]

    @property
    def trainable_ids(self) -> list[int]:
        return [self.seed.token_id] + [v.token_id for v in self.values]


def register_value_token(model, tokenizer, surface: str) -> ValueToken:  # noqa: ANN001
    """Add ``surface`` as a single new vocab token that decodes to the bare
    string (no angle brackets). Embedding row inited to BPE-mean of the
    original pieces. Does NOT install a grad mask — call
    ``register_axiom_tokens`` to get a properly masked set."""
    original_bpe_ids = _term_token_ids(tokenizer, surface)
    if not original_bpe_ids:
        raise ValueError(f"could not tokenize surface {surface!r}")

    added_token = AddedToken(surface, special=False, normalized=False)
    tokenizer.add_tokens([added_token])
    token_id = tokenizer.convert_tokens_to_ids(surface)

    current_emb_rows = model.get_input_embeddings().weight.shape[0]
    if token_id >= current_emb_rows:
        model.resize_token_embeddings(token_id + 1)

    encoded = tokenizer.encode(surface, add_special_tokens=False)
    if encoded != [token_id]:
        raise RuntimeError(f"{surface!r} did not encode as a single token: {encoded}")

    embed = model.get_input_embeddings()
    with torch.no_grad():
        init = embed.weight[original_bpe_ids].mean(dim=0)
        embed.weight[token_id] = init

    return ValueToken(surface=surface, token_id=token_id, original_bpe_ids=original_bpe_ids)


def _install_multi_grad_mask(model, trainable_ids: list[int]) -> None:  # noqa: ANN001
    """Install ONE grad hook covering all trainable_ids. Replaces the
    per-token approach in seed_token.py that stacked hooks and zeroed grads
    when a second token was registered on the same model."""
    embed_w = model.get_input_embeddings().weight
    embed_w.requires_grad = True
    keep = torch.tensor(trainable_ids, dtype=torch.long)

    def hook(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked[keep] = grad[keep]
        return masked

    embed_w.register_hook(hook)


def register_axiom_tokens(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    name: str,
    value_surfaces: list[str],
) -> AxiomTokens:
    """Register a seed token + value tokens for one axiom.

    Freezes the model, then installs a SINGLE grad mask covering all
    trainable ids (seed + values). Call this instead of calling
    ``register_seed_token`` + ``register_value_token`` individually to avoid
    the stacking bug.
    """
    # Register seed WITHOUT its own grad mask (we'll install one below).
    original_bpe_ids = _term_token_ids(tokenizer, name)
    if not original_bpe_ids:
        raise ValueError(f"could not tokenize name {name!r}")

    token_str = f"<{name}>"
    added_token = AddedToken(token_str, special=False, normalized=False)
    tokenizer.add_tokens([added_token])
    seed_id = tokenizer.convert_tokens_to_ids(token_str)

    current_emb_rows = model.get_input_embeddings().weight.shape[0]
    if seed_id >= current_emb_rows:
        model.resize_token_embeddings(seed_id + 1)

    encoded = tokenizer.encode(token_str, add_special_tokens=False)
    if encoded != [seed_id]:
        raise RuntimeError(f"{token_str!r} did not encode as single token: {encoded}")

    embed = model.get_input_embeddings()
    with torch.no_grad():
        embed.weight[seed_id] = embed.weight[original_bpe_ids].mean(dim=0)

    seed = SeedToken(
        name=name,
        token_id=seed_id,
        original_term=name,
        original_bpe_ids=original_bpe_ids,
    )

    values = [register_value_token(model, tokenizer, s) for s in value_surfaces]

    _freeze_model_params(model)
    _install_multi_grad_mask(model, [seed_id] + [v.token_id for v in values])

    return AxiomTokens(seed=seed, values=values)


TEMPLATE = "Q: {q}\nA:"


def train_axiom_tokens(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom: AxiomTokens,
    bolt: BoltSelector,
    qa_pairs: list[tuple[str, str]],
    n_steps: int = 300,
    lr: float = 1e-3,
    seed_rng: int = 42,
) -> list[float]:
    """Train bolt adapters + all trainable embedding rows (seed + values).

    The tokenizer automatically encodes registered value surfaces as single
    tokens, so CE on the answer span directly trains the bolt to route
    question → value token. No explicit answer substitution needed.
    """
    device = next(model.parameters()).device
    embed_w = model.get_input_embeddings().weight
    params = list(bolt_parameters(bolt)) + [embed_w]
    optim = torch.optim.AdamW(params, lr=lr)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(seed_rng)
    losses: list[float] = []

    handles = install_bolt_hooks(model, bolt)
    try:
        for _ in range(n_steps):
            q, a = rng.choice(qa_pairs)
            q_text = TEMPLATE.format(q=q)
            full_text = q_text + " " + a

            q_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            full_ids = tokenizer(
                full_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)
            if eos_id is not None:
                full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)

            labels = torch.full_like(full_ids, -100)
            labels[0, q_ids.shape[1] :] = full_ids[0, q_ids.shape[1] :]

            optim.zero_grad()
            loss = model(full_ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()
            losses.append(float(loss.item()))
    finally:
        remove_bolt_hooks(handles)

    return losses
