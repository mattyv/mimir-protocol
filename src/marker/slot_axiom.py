"""Per-axiom slot injection — Option #1 from the multi-axiom design.

Each axiom owns a contiguous range of dimensions ("slot") in the
residual stream at one chosen layer. Its learnable vector is added
to those dims at every token position during the forward pass.

Different axioms get different (non-overlapping) slots. They compose
by occupying different parts of the hidden state — no interference
because their writes don't overlap.

Frozen LLM; only the slot vector trains per axiom. ~slot_width × fp32
floats per axiom (~1 KB at slot_width=256).

This is a controlled experiment for the question: can the model's
downstream layers READ information that's been written into a
designated slot of the residual stream?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from marker.prefix_tuning import _get_layers


@dataclass
class SlotAxiom:
    name: str
    slot_start: int
    slot_width: int
    target_layer: int
    hidden_size: int
    vector: nn.Parameter = field(init=False)

    @classmethod
    def new(
        cls,
        name: str,
        slot_start: int,
        slot_width: int,
        target_layer: int,
        hidden_size: int,
    ) -> SlotAxiom:
        if slot_start < 0 or slot_start + slot_width > hidden_size:
            raise ValueError(
                f"slot [{slot_start}:{slot_start + slot_width}] out of range for hidden_size={hidden_size}"
            )
        sa = cls.__new__(cls)
        sa.name = name
        sa.slot_start = slot_start
        sa.slot_width = slot_width
        sa.target_layer = target_layer
        sa.hidden_size = hidden_size
        # Init to zero so untrained slot is a strict no-op.
        sa.vector = nn.Parameter(torch.zeros(slot_width, dtype=torch.float32))
        return sa


def _slot_hook_for(sa: SlotAxiom):  # noqa: ANN202
    """Returns a forward_hook function that adds sa.vector to the slot
    dims of the residual stream output by the target layer."""

    def hook(_module, _inputs, output):
        # transformer layers in HF often return (hidden_states, ...) or just
        # hidden_states; normalise.
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        # h shape: (batch, seq, hidden)
        slot = sa.vector.to(dtype=h.dtype, device=h.device)
        h_new = h.clone()
        h_new[..., sa.slot_start : sa.slot_start + sa.slot_width] = (
            h[..., sa.slot_start : sa.slot_start + sa.slot_width] + slot
        )
        if is_tuple:
            return (h_new, *output[1:])
        return h_new

    return hook


def install_slot_hooks(model, slot_axioms: list[SlotAxiom]) -> list:  # noqa: ANN001
    """Install forward hooks on each axiom's target layer that add the
    axiom's slot vector to the residual stream output. Returns hook
    handles; caller must call `.remove()` on each."""
    layers = _get_layers(model)
    handles = []
    for sa in slot_axioms:
        if sa.target_layer < 0 or sa.target_layer >= len(layers):
            raise ValueError(f"target_layer {sa.target_layer} out of range [0, {len(layers)})")
        handles.append(layers[sa.target_layer].register_forward_hook(_slot_hook_for(sa)))
    return handles


@torch.no_grad()
def _build_qa_batch(
    tokenizer,
    qa_pairs: list[tuple[str, str]],
    template: str = "Q: {q}\nA: {a}",
    device: torch.device | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Tokenize each (question, answer) pair into (input_ids, labels) where
    labels is -100 over the question portion and equals input_ids over the
    answer portion (loss is computed only on answer tokens).
    """
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for q, a in qa_pairs:
        question_part = template.split("{a}")[0].format(q=q)  # e.g. "Q: ...\nA: "
        full_text = template.format(q=q, a=a)
        q_ids = tokenizer(question_part, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        full_t = torch.tensor([full_ids])
        labels = torch.full_like(full_t, -100)
        labels[0, len(q_ids) :] = full_t[0, len(q_ids) :]
        if device is not None:
            full_t = full_t.to(device)
            labels = labels.to(device)
        out.append((full_t, labels))
    return out


def train_slot_qa(
    model,  # noqa: ANN001
    tokenizer,
    sa: SlotAxiom,
    qa_pairs: list[tuple[str, str]],
    n_steps: int = 200,
    lr: float = 0.05,
    template: str = "Q: {q}\nA: {a}",
) -> list[float]:
    """Train sa.vector on a set of question/answer pairs so the slot
    becomes useful in QUESTION contexts (not just continuation contexts).

    At each step, samples one (Q, A) pair, feeds the full text to the
    model with the slot active, and computes cross-entropy on the
    ANSWER tokens only (question is masked to -100).

    Only sa.vector trains. Returns per-step loss.
    """
    import random

    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    device = next(model.parameters()).device
    sa.vector = nn.Parameter(sa.vector.data.to(device=device, dtype=torch.float32).clone())
    optim = torch.optim.AdamW([sa.vector], lr=lr)

    batch = _build_qa_batch(tokenizer, qa_pairs, template=template, device=device)
    rng = random.Random(0)

    handles = install_slot_hooks(model, [sa])
    losses: list[float] = []
    try:
        for _ in range(n_steps):
            input_ids, labels = batch[rng.randrange(len(batch))]
            optim.zero_grad()
            out = model(input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu().item()))
    finally:
        for h in handles:
            h.remove()
    return losses


def train_slot(
    model,  # noqa: ANN001
    tokenizer,
    sa: SlotAxiom,
    description: str,
    n_steps: int = 100,
    lr: float = 0.05,
    prompt_prefix: str = "Description: ",
) -> list[float]:
    """Train sa.vector to make the frozen model reproduce `description`
    when conditioned on `prompt_prefix` with the slot injection active.

    Loss = next-token cross-entropy on the description tokens.
    Only sa.vector is trained. Returns the per-step loss list.
    """
    # Freeze model
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    device = next(model.parameters()).device
    sa.vector.data = sa.vector.data.to(device=device, dtype=torch.float32)
    # Re-wrap as Parameter so it tracks grads
    sa.vector = nn.Parameter(sa.vector.data.clone())

    optim = torch.optim.AdamW([sa.vector], lr=lr)

    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False).input_ids
    desc_ids = tokenizer(description, add_special_tokens=False).input_ids
    full_ids = torch.tensor([prefix_ids + desc_ids], device=device)
    # Target: predict desc_ids one-step shifted from full_ids.
    # Labels: -100 for the prefix portion (don't train on it), desc tokens for the rest.
    labels = torch.full_like(full_ids, -100)
    n_prefix = len(prefix_ids)
    labels[0, n_prefix:] = full_ids[0, n_prefix:]

    handles = install_slot_hooks(model, [sa])
    losses: list[float] = []
    try:
        for _ in range(n_steps):
            optim.zero_grad()
            out = model(full_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu().item()))
    finally:
        for h in handles:
            h.remove()
    return losses
