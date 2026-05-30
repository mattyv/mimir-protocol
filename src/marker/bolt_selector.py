"""Per-layer bolt-on selector for single-vector axioms.

A ``BoltSelector`` is a bank of low-rank adapters — one per transformer
block. When the seed token id is detected in the input to a forward pass,
each adapter reads its layer's residual stream at every position and writes
a learned bias back into it. When the seed token is absent the adapters are
invisible.

Combined with the Phase 1 ``SeedToken`` (which provides the new vocab entry
and freezes everything except its embedding row), the bolt-on selector is
the knowledge-carrying half of the single-vector axiom architecture.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from marker.seed_token import SeedToken

# Match the Q+A template used elsewhere in the codebase so the smoke training
# behaves consistently with the existing run_axiom_mlp_demo workflow.
TEMPLATE = "Q: {q}\nA:"


class _Adapter(nn.Module):
    """SmallMLP-style low-rank adapter: down → GELU → up, with up zero-init
    so the whole bolt-on starts as an exact no-op."""

    def __init__(self, hidden_size: int, r: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, r, bias=False)
        self.up = nn.Linear(r, hidden_size, bias=False)
        self.act = nn.GELU()
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


@dataclass
class BoltSelector:
    """Per-layer adapter bank, gated on seed-token presence in the current
    forward pass.

    ``adapters[i]`` fires at ``fire_layers[i]``. Defaults to all layers but
    can be restricted to a subset (e.g. a single mid-stack layer for facts,
    mirroring ROME-style single-site writes).
    """

    seed: SeedToken
    adapters: nn.ModuleList
    fire_layers: list
    r: int
    skill_mode: bool = False
    _fire_this_pass: bool = False


def make_bolt_selector(
    model,  # noqa: ANN001
    seed: SeedToken,
    r: int = 16,
    skill_mode: bool = False,
    layers: list | None = None,
) -> BoltSelector:
    """Allocate one adapter per entry in ``layers`` (defaults to all blocks).

    Pass ``layers=[14]`` to restrict to a single mid-stack layer (ROME-style),
    or ``layers=[12, 13, 14]`` for a narrow band. Facts benefit from a small
    targeted set; skills can stay all-layers.
    """
    n_layers = model.config.num_hidden_layers
    fire_layers = list(layers) if layers is not None else list(range(n_layers))
    hidden = model.config.hidden_size
    device = next(model.parameters()).device
    adapters = nn.ModuleList([_Adapter(hidden, r) for _ in fire_layers])
    adapters = adapters.to(device=device, dtype=torch.float32)
    return BoltSelector(
        seed=seed, adapters=adapters, fire_layers=fire_layers, r=r, skill_mode=skill_mode
    )


def _make_embedding_pre_hook(bolt: BoltSelector):
    """Pre-hook on ``get_input_embeddings()`` that scans the lookup ids for
    the seed token and stashes a per-forward-pass fire flag on ``bolt``."""
    seed_id = bolt.seed.token_id

    def pre_hook(_module: nn.Module, args: tuple) -> None:
        if not args:
            return
        input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor):
            bolt._fire_this_pass = False
            return
        seed_present = bool((input_ids == seed_id).any().item())
        # During autoregressive decoding the per-step input is a single token
        # that won't contain the seed. In skill mode we want the bolt-on to
        # keep steering generation, so we latch the fire flag: once a prefill
        # pass has seen the seed, subsequent single-token decode steps keep
        # firing. A prefill pass (seq len > 1) always re-evaluates from the
        # actual ids, so a fresh generation about something else resets it.
        is_decode_step = input_ids.shape[-1] == 1
        if bolt.skill_mode and is_decode_step:
            bolt._fire_this_pass = bolt._fire_this_pass or seed_present
        else:
            bolt._fire_this_pass = seed_present

    return pre_hook


def _make_layer_hook(bolt: BoltSelector, layer_idx: int):
    """Forward hook on ``model.model.layers[layer_idx]`` that, when the gate
    is set, runs the corresponding adapter at every position and adds the
    result back into the residual stream."""
    adapter = bolt.adapters[layer_idx]

    def hook(_module: nn.Module, _input: tuple, output):  # noqa: ANN001
        if not bolt._fire_this_pass:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        offset = adapter(hidden.float()).to(hidden.dtype)
        new_hidden = hidden + offset
        if is_tuple:
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook


def install_bolt_hooks(model, bolt: BoltSelector) -> list:  # noqa: ANN001
    """Register the embedding pre-hook + one forward hook per transformer
    layer. Returns the list of handles; caller must call
    ``remove_bolt_hooks(handles)`` (or ``handle.remove()``) when done.
    """
    handles: list = []
    embed = model.get_input_embeddings()
    handles.append(embed.register_forward_pre_hook(_make_embedding_pre_hook(bolt)))
    for adapter_idx, layer_idx in enumerate(bolt.fire_layers):
        handles.append(
            model.model.layers[layer_idx].register_forward_hook(_make_layer_hook(bolt, adapter_idx))
        )
    return handles


def remove_bolt_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


def bolt_parameters(bolt: BoltSelector) -> Iterable[nn.Parameter]:
    yield from bolt.adapters.parameters()


def train_seed_and_bolt(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    seed: SeedToken,
    bolt: BoltSelector,
    qa_pairs: list[tuple[str, str]],
    n_steps: int = 300,
    lr: float = 1e-3,
    seed_rng: int = 42,
) -> list[float]:
    """Joint training of the seed embedding row and bolt-on adapters.

    Loss is cross-entropy on answer tokens only (mirroring the existing
    ``run_axiom_mlp_demo.train`` pattern). The base model stays frozen via
    the Phase 1 gradient mask and the optimizer parameter list. Returns the
    per-step loss history.
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
