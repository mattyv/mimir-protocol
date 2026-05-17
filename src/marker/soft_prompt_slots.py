"""Slot-assigned soft prompts (v9).

A different training paradigm from v5-v8. Each slot position has a
designated role — one Q+A pair (or special role like overview /
boundary). During training, only the assigned slot's gradient is
updated for its Q+A. All other slots stay frozen at their init values
during that step.

The bet: each slot becomes a focused "memory cell" for one fact. At
inference, the model's natural attention picks which slot to read from
based on K-vector similarity to the query.

Architecture per axiom:
  - N term-position trainable vectors (e.g., 2 for "Balance" + "Publisher")
  - N_slots slot positions, one per designated Q+A pair
  - Vector shape: (n_term + n_slots, hidden_size)

Training (one step):
  1. Sample slot index i uniformly at random.
  2. Sample a paraphrase of the Q assigned to slot i.
  3. Forward pass with all slots loaded (and substituted via hook).
  4. Compute loss on the answer A_i's tokens.
  5. Backward → gradient on all slots (non-zero through attention).
  6. MASK the gradient: keep only slot i + term positions.
  7. Optimizer step.

Each slot i thus learns to be the position the model reads when asking
question-shape-i, holding all other slots fixed as context.

Slot init: each slot's vector is initialized from the mean embedding
of the answer's content tokens (stopwords filtered). This gives the
optimizer a meaningful starting point per slot.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from marker.soft_prompt import (
    _get_embed_module,
    _term_token_ids,
)
from marker.soft_prompt_plus import (
    install_soft_prompt_plus_hook,
    prepare_input_with_ghosts,
)

_STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "for", "and", "in", "on", "at", "it",
    "its", "this", "that", "as", "by", "with", "from", "or", "be", "are", "was",
    "were", "has", "have", "had", "but", "not", "no", "do", "does", "did", "if",
    "what", "who", "where", "when", "how", "why", "which", "any", "all", "some",
    "we", "you", "he", "she", "they", "them", "our", "your", "their", "i",
    "into", "out", "up", "down", "over", "under", "than", "then", "so", "also",
    "only", "very", "much", "most", "more", "less", "few", "many", "such",
}  # fmt: skip


def _informative_token_ids(tokenizer, text: str, max_tokens: int = 5) -> list[int]:
    """Tokenize `text`, skip stopwords + 1-char tokens, return token IDs
    of up to `max_tokens` content-bearing words. Each word becomes one
    sub-token id (the leading-space-prefixed first sub-token)."""
    words = re.findall(r"[\w][\w.\-_]+", text)
    out: list[int] = []
    seen_words: set[str] = set()
    for word in words:
        wl = word.lower()
        if wl in _STOPWORDS:
            continue
        if len(word) < 2:
            continue
        if wl in seen_words:
            continue
        seen_words.add(wl)
        ids = tokenizer(" " + word, add_special_tokens=False).input_ids
        if ids:
            out.append(ids[0])
        if len(out) >= max_tokens:
            break
    return out


@dataclass
class SoftPromptSlots:
    """Soft prompt with N slot positions, each assigned to one Q+A pair.

    `slot_qa[i]` = (list of paraphrased questions, single answer).
    """

    term: str
    term_token_ids: list[int]
    slot_qa: list[tuple[list[str], str]]
    vector: nn.Parameter  # shape (n_term + n_slots, hidden)

    @property
    def n_slots(self) -> int:
        return len(self.slot_qa)

    @property
    def n_ghost(self) -> int:
        """Alias so SoftPromptPlus's hooks and prepare-input helpers
        work unchanged (they read .n_ghost)."""
        return self.n_slots


def make_soft_prompt_slots(
    model,  # noqa: ANN001
    tokenizer,
    term: str,
    slot_qa: list[tuple[list[str], str]],
) -> SoftPromptSlots:
    """Build a SoftPromptSlots with content-token-mean init per slot."""
    token_ids = _term_token_ids(tokenizer, term)
    if not token_ids:
        raise ValueError(f"could not tokenize term {term!r}")
    embed = _get_embed_module(model)
    with torch.no_grad():
        term_init = embed.weight[token_ids].detach().clone().float()
        hidden = term_init.shape[1]
        slot_rows: list[torch.Tensor] = []
        for _q_paras, answer in slot_qa:
            info_ids = _informative_token_ids(tokenizer, answer, max_tokens=5)
            if info_ids:
                emb = embed.weight[info_ids].detach().float()
                slot_init = emb.mean(dim=0)
            else:
                slot_init = torch.zeros(hidden, dtype=term_init.dtype)
            slot_rows.append(slot_init)
        slot_tensor = torch.stack(slot_rows, dim=0)
        full_init = torch.cat([term_init, slot_tensor], dim=0)
    return SoftPromptSlots(
        term=term,
        term_token_ids=token_ids,
        slot_qa=slot_qa,
        vector=nn.Parameter(full_init),
    )


def train_soft_prompt_slots(
    model,  # noqa: ANN001
    tokenizer,
    sp: SoftPromptSlots,
    n_steps: int = 2000,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    train_term_positions: bool = True,
    append_eos: bool = True,
    template: str = "Q: {q}\nA: {a}",
) -> list[float]:
    """Slot-masked training. Each step updates ONE slot (the one assigned
    to the sampled Q+A), plus optionally the term-position vectors."""
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    device = next(model.parameters()).device
    sp.vector = nn.Parameter(sp.vector.data.to(device=device, dtype=torch.float32).clone())
    optim = torch.optim.AdamW([sp.vector], lr=lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps, eta_min=lr_end)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id
    n_term = len(sp.term_token_ids)
    n_slots = sp.n_slots

    rng = random.Random(0)
    losses: list[float] = []
    skipped = 0
    for _ in range(n_steps):
        slot_i = rng.randrange(n_slots)
        q_variants, answer = sp.slot_qa[slot_i]
        q = rng.choice(q_variants)

        question_part = template.split("{a}")[0].format(q=q)
        full_text = template.format(q=q, a=answer)
        ids_with_ghosts, positions = prepare_input_with_ghosts(tokenizer, full_text, sp, pad_id)
        if not positions:
            skipped += 1
            continue
        q_ids, _ = prepare_input_with_ghosts(tokenizer, question_part, sp, pad_id)
        n_q = q_ids.shape[1]

        if append_eos and eos_id is not None:
            ids_with_ghosts = torch.cat(
                [ids_with_ghosts, torch.tensor([[eos_id]], dtype=ids_with_ghosts.dtype)],
                dim=1,
            )
        input_ids = ids_with_ghosts.to(device)
        labels = torch.full_like(input_ids, -100)
        labels[0, n_q:] = input_ids[0, n_q:]

        handle = install_soft_prompt_plus_hook(model, sp, positions)
        try:
            optim.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out.loss
            loss.backward()

            # Gradient mask: zero out all positions except (term if enabled) + slot_i
            with torch.no_grad():
                if sp.vector.grad is not None:
                    mask = torch.zeros_like(sp.vector)
                    if train_term_positions:
                        mask[:n_term] = 1.0
                    mask[n_term + slot_i] = 1.0
                    sp.vector.grad *= mask

            optim.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
        finally:
            handle.remove()

    if skipped > 0:
        # Most likely cause: term not appearing in the formatted text — should
        # never happen if slot_qa was built correctly, but surface it loudly.
        import warnings

        warnings.warn(f"train_soft_prompt_slots: skipped {skipped} steps (term not found)")
    return losses


@torch.no_grad()
def generate_with_slots(
    model,  # noqa: ANN001
    tokenizer,
    sp: SoftPromptSlots,
    prompt: str,
    max_new: int = 100,
) -> str:
    """Generate using the slot-trained soft prompt. Reuses the existing
    soft-prompt-plus hook (only depends on sp.vector + sp.term + sp.n_ghost)."""
    from marker.soft_prompt_plus import generate_with_soft_prompt_plus

    return generate_with_soft_prompt_plus(model, tokenizer, sp, prompt, max_new=max_new)


# ---------------------------------------------------------------------------
# Slot allocation helpers for typical axiom structures.
# ---------------------------------------------------------------------------


def build_slot_qa_default(
    axiom: dict,
    include_overview: bool = True,
    boundary_slots: int = 3,
) -> list[tuple[list[str], str]]:
    """Build a default slot_qa list for an axiom with `facts`,
    `boundary_examples` (optional), and `description`.

    Returns a list of (paraphrased Q list, A) per slot:
      - One slot per fact, using its hand-written paraphrases.
      - Optionally one overview slot (multiple "tell me about X" forms).
      - Optionally a few boundary slots.
    """
    name = axiom["name"]
    desc = axiom["description"]
    slots: list[tuple[list[str], str]] = []

    # Fact slots
    for f in axiom["facts"]:
        slots.append((list(f["questions_train"]), f["answer"]))

    # Overview slot
    if include_overview:
        slots.append(
            (
                [f"Tell me about {name}.", f"Describe {name}.", f"What is {name}?"],
                desc,
            )
        )

    # Boundary slots — paraphrased, generic
    boundary_pool: list[tuple[list[str], str]] = [
        (
            [
                f"What programming language is {name} written in?",
                f"What language is {name} in?",
                f"Which programming language does {name} use?",
            ],
            f"The description doesn't specify what programming language {name} is written in.",
        ),
        (
            [
                f"What's the SLA of {name}?",
                f"What SLA does {name} have?",
                f"What is {name}'s availability target?",
            ],
            f"The description doesn't specify an SLA for {name}.",
        ),
        (
            [
                f"How is {name} deployed?",
                f"Where does {name} run?",
                f"What is {name}'s deployment platform?",
            ],
            f"The description doesn't specify how {name} is deployed.",
        ),
        (
            [
                f"What database does {name} use?",
                f"Does {name} have a database?",
                f"What kind of storage does {name} use?",
            ],
            f"The description doesn't mention {name} using a database.",
        ),
    ]
    slots.extend(boundary_pool[:boundary_slots])
    return slots
