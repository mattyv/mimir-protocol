"""Soft prompt + ghost tokens.

Extends soft_prompt.py with extra trainable "ghost" vectors inserted
immediately after the term's natural sub-tokens. The user's prompt is
expanded in token-space (not text-space) by inserting `n_ghost` padding
tokens after the term; the hook then substitutes BOTH the term's
sub-token positions AND the ghost positions with learned vectors.

Why ghosts: a 3-sub-token term gives the optimizer only 3 trainable
slots × hidden_size. Adding ghosts gives more degrees of freedom for
the optimizer to encode the axiom's facts.

Mechanically:
  - Input prompt tokens are augmented with `n_ghost` pad-token copies
    inserted right after the term.
  - The soft prompt's `vector` has shape (n_term_tokens + n_ghost,
    hidden_size). The first n_term_tokens are initialized from the
    term's natural embeddings; the n_ghost rows are zero-initialized.
  - At forward time, an embedding-layer hook substitutes the vector
    rows into the residual stream at the term + ghost positions.

Frozen model. Only `vector` trains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from marker.soft_prompt import (
    _get_embed_module,
    _term_token_ids,
    find_term_positions,
)


@dataclass
class SoftPromptPlus:
    term: str
    term_token_ids: list[int]
    n_ghost: int
    vector: nn.Parameter  # shape [n_term_tokens + n_ghost, hidden_size]

    @classmethod
    def from_term(
        cls,
        model,  # noqa: ANN001
        tokenizer,
        term: str,
        n_ghost: int = 0,
    ) -> SoftPromptPlus:
        token_ids = _term_token_ids(tokenizer, term)
        if not token_ids:
            raise ValueError(f"could not tokenize term {term!r}")
        embed = _get_embed_module(model)
        with torch.no_grad():
            term_init = embed.weight[token_ids].detach().clone().float()
        ghost_init = torch.zeros(
            n_ghost, term_init.shape[1], dtype=term_init.dtype, device=term_init.device
        )
        full_init = torch.cat([term_init, ghost_init], dim=0)
        return cls(
            term=term,
            term_token_ids=token_ids,
            n_ghost=n_ghost,
            vector=nn.Parameter(full_init),
        )


def prepare_input_with_ghosts(
    tokenizer,
    prompt: str,
    sp: SoftPromptPlus,
    pad_token_id: int | None = None,
) -> tuple[torch.Tensor | None, list[int]]:
    """Tokenize the prompt; if the term appears, insert n_ghost pad tokens
    right after the term's last sub-token. Return (input_ids, positions)
    where positions is the merged list [term_positions ++ ghost_positions].

    If the term is NOT in the prompt, returns (input_ids, []) and no
    insertion occurs.
    """
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if sp.n_ghost == 0:
        positions = find_term_positions(tokenizer, prompt, sp.term)
        return torch.tensor([ids]), positions

    term_positions = find_term_positions(tokenizer, prompt, sp.term)
    if not term_positions:
        return torch.tensor([ids]), []

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    term_end = term_positions[-1]
    new_ids = ids[: term_end + 1] + [pad_token_id] * sp.n_ghost + ids[term_end + 1 :]
    ghost_positions = list(range(term_end + 1, term_end + 1 + sp.n_ghost))
    return torch.tensor([new_ids]), term_positions + ghost_positions


def install_soft_prompt_plus_hook(model, sp: SoftPromptPlus, positions: list[int]):  # noqa: ANN001, ANN201
    """Install an embedding-layer forward hook that replaces the output
    at `positions[i]` with `sp.vector[i]`. Returns a handle."""
    embed = _get_embed_module(model)
    pos_list = list(positions)
    n_vec = sp.vector.shape[0]

    def hook(_module, _inputs, output):
        if not pos_list:
            return output
        seq_len = output.shape[1]
        if seq_len == 1:
            return output  # decode step, KV cache propagates
        out = output.clone()
        for i, pos in enumerate(pos_list):
            if 0 <= pos < seq_len and i < n_vec:
                out[:, pos, :] = sp.vector[i].to(dtype=out.dtype, device=out.device)
        return out

    return embed.register_forward_hook(hook)


def train_soft_prompt_plus_qa(
    model,  # noqa: ANN001
    tokenizer,
    sp: SoftPromptPlus,
    qa_pairs: list[tuple[str, str]],
    n_steps: int = 400,
    lr: float = 0.05,
    template: str = "Q: {q}\nA: {a}",
) -> list[float]:
    """Train sp.vector on Q+A pairs with ghost insertion.

    At each step samples one pair, expands it with ghosts, computes loss
    only on answer tokens. Frozen model.
    """
    import random

    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    device = next(model.parameters()).device
    sp.vector = nn.Parameter(sp.vector.data.to(device=device, dtype=torch.float32).clone())
    optim = torch.optim.AdamW([sp.vector], lr=lr)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    samples: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
    for q, a in qa_pairs:
        question_part = template.split("{a}")[0].format(q=q)
        full_text = template.format(q=q, a=a)
        ids_with_ghosts, positions = prepare_input_with_ghosts(tokenizer, full_text, sp, pad_id)
        if not positions:
            continue
        # Find the boundary between question and answer in TOKEN SPACE, on
        # the ghost-expanded sequence.
        q_ids_with_ghosts, _ = prepare_input_with_ghosts(tokenizer, question_part, sp, pad_id)
        n_q = q_ids_with_ghosts.shape[1]
        input_ids = ids_with_ghosts.to(device)
        labels = torch.full_like(input_ids, -100)
        labels[0, n_q:] = input_ids[0, n_q:]
        samples.append((input_ids, labels, positions))

    if not samples:
        raise RuntimeError(f"no Q+A pairs contained term {sp.term!r}")

    rng = random.Random(0)
    losses: list[float] = []
    for _ in range(n_steps):
        input_ids, labels, positions = samples[rng.randrange(len(samples))]
        handle = install_soft_prompt_plus_hook(model, sp, positions)
        try:
            optim.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu().item()))
        finally:
            handle.remove()
    return losses


@torch.no_grad()
def generate_with_soft_prompt_plus(
    model,  # noqa: ANN001
    tokenizer,
    sp: SoftPromptPlus,
    prompt: str,
    max_new: int = 80,
) -> str:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids, positions = prepare_input_with_ghosts(tokenizer, prompt, sp, pad_id)
    input_ids = input_ids.to(device)
    if not positions:
        # Term not in prompt — generate vanilla
        out_ids = input_ids.clone()
        for _ in range(max_new):
            out = model(out_ids)
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            out_ids = torch.cat([out_ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        return tokenizer.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=True)

    handle = install_soft_prompt_plus_hook(model, sp, positions)
    try:
        out_ids = input_ids.clone()
        for _ in range(max_new):
            out = model(out_ids)
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            out_ids = torch.cat([out_ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        return tokenizer.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
    finally:
        handle.remove()
