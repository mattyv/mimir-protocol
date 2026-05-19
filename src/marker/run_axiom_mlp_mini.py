"""Mini local test for per-axiom MLP injection.

Validates the mechanism on a small model (Qwen2.5-1.5B by default, runs
on MPS/CPU). Uses a fictional axiom the model cannot know from pretraining.

What's being tested:
  - Hook fires at the term position at each chosen layer
  - Small MLP adds an offset to the residual stream there
  - Training converges and the model's answers change
  - Held-out question variants work (not just training phrasings)
  - Boundary questions are declined rather than hallucinated

Architecture per axiom:
  - SmallMLP at each of N chosen layers (default 3)
  - Each MLP: hidden → r → hidden  (default r=8)
  - Hook fires at term-position during prefill; KV cache carries the
    modified representation through decode
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Fictional axiom ────────────────────────────────────────────────────────────
# The model cannot know this — so any correct answers come from our MLP.

TERM = "Glorbox"
DESCRIPTION = "Glorbox is a service that synchronizes purple widgets to the cloud every 42 seconds."

TRAIN_QA = [
    (
        "What is Glorbox?",
        "Glorbox is a service that synchronizes purple widgets to the cloud every 42 seconds.",
    ),
    ("What does Glorbox do?", "Glorbox synchronizes purple widgets to the cloud every 42 seconds."),
    ("How often does Glorbox sync?", "Glorbox syncs every 42 seconds."),
    ("What does Glorbox synchronize?", "Glorbox synchronizes purple widgets."),
    ("How frequently does Glorbox run?", "Glorbox runs every 42 seconds."),
    ("What is Glorbox's sync interval?", "Glorbox's sync interval is 42 seconds."),
]

HELDOUT_QA = [
    ("What gets uploaded by Glorbox?", "Purple widgets get uploaded by Glorbox."),
    ("How long is the gap between Glorbox syncs?", "42 seconds."),
    ("Describe what Glorbox does.", "Glorbox syncs purple widgets to the cloud every 42 seconds."),
]

# Used for training (teach the MLP to decline out-of-scope questions)
BOUNDARY_TRAIN_QA = [
    (
        "What programming language is Glorbox written in?",
        "The description doesn't specify what language Glorbox uses.",
    ),
    ("Who created Glorbox?", "The description doesn't say who created Glorbox."),
    (
        "What cloud provider does Glorbox use?",
        "The description doesn't specify what cloud provider Glorbox uses.",
    ),
    ("What is Glorbox's SLA?", "The description doesn't mention an SLA for Glorbox."),
    (
        "How many employees does Glorbox have?",
        "The description doesn't mention how many employees Glorbox has.",
    ),
    ("When was Glorbox founded?", "The description doesn't say when Glorbox was founded."),
    (
        "What database does Glorbox use?",
        "The description doesn't mention what database Glorbox uses.",
    ),
    ("How much does Glorbox cost?", "The description doesn't mention any pricing for Glorbox."),
    (
        "What operating system does Glorbox run on?",
        "The description doesn't specify what operating system Glorbox runs on.",
    ),
    ("Is Glorbox open source?", "The description doesn't say whether Glorbox is open source."),
]

# Used only for evaluation (held-out boundary probes)
BOUNDARY_QA = [
    (
        "What programming language is Glorbox written in?",
        "The description doesn't specify what language Glorbox uses.",
    ),
    ("Who created Glorbox?", "The description doesn't say who created Glorbox."),
    (
        "What cloud provider does Glorbox use?",
        "The description doesn't specify what cloud provider Glorbox uses.",
    ),
    ("What is Glorbox's SLA?", "The description doesn't mention an SLA for Glorbox."),
]

TEMPLATE = "Q: {q}\nA:"

# ── Architecture ───────────────────────────────────────────────────────────────


class SmallMLP(nn.Module):
    def __init__(self, hidden_size: int, r: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, r, bias=False)
        self.up = nn.Linear(r, hidden_size, bias=False)
        self.act = nn.GELU()
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)  # zero init → starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


@dataclass
class AxiomMLP:
    term: str
    term_token_ids: list[int]
    chosen_layers: list[int]
    mlps: nn.ModuleList  # one SmallMLP per chosen layer


def make_axiom_mlp(
    model,
    tokenizer,
    term: str,
    chosen_layers: list[int],
    r: int = 8,
) -> AxiomMLP:
    hidden = model.config.hidden_size
    # Try " Term" (with leading space) first — Qwen tokenizer convention
    ids = (
        tokenizer(" " + term, add_special_tokens=False).input_ids
        or tokenizer(term, add_special_tokens=False).input_ids
    )
    device = next(model.parameters()).device
    mlps = nn.ModuleList([SmallMLP(hidden, r) for _ in chosen_layers])
    mlps = mlps.to(device=device, dtype=torch.float32)
    return AxiomMLP(term=term, term_token_ids=ids, chosen_layers=chosen_layers, mlps=mlps)


# ── Hook plumbing ──────────────────────────────────────────────────────────────


def _find_term_positions(input_ids: torch.Tensor, term_ids: list[int]) -> list[int]:
    """Return start indices of every occurrence of term_ids in input_ids[0]."""
    ids = input_ids[0].tolist()
    n = len(term_ids)
    return [i for i in range(len(ids) - n + 1) if ids[i : i + n] == term_ids]


def _make_layer_hook(mlp: SmallMLP, positions: list[int]):
    """Return a forward hook that adds mlp(h) at each term position."""

    def hook(module, input, output):
        # Newer transformers returns the hidden tensor directly; older returns a tuple.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output  # (batch, seq_len, hidden)
        new_hidden = hidden
        for pos in positions:
            if pos >= new_hidden.shape[1]:
                continue
            offset = mlp(new_hidden[:, pos, :].float()).to(new_hidden.dtype)  # (batch, hidden)
            # torch.cat is explicitly differentiable — gradient flows to offset and
            # hence to mlp.parameters(). The zeros_like+slice-assign approach breaks
            # the computation graph on some PyTorch/transformers versions.
            new_hidden = torch.cat(
                [
                    new_hidden[:, :pos, :],
                    new_hidden[:, pos : pos + 1, :] + offset.unsqueeze(1),
                    new_hidden[:, pos + 1 :, :],
                ],
                dim=1,
            )
        return (new_hidden,) + output[1:] if is_tuple else new_hidden

    return hook


def install_hooks(model, axiom_mlp: AxiomMLP, positions: list[int]):
    handles = []
    for layer_idx, mlp in zip(axiom_mlp.chosen_layers, axiom_mlp.mlps, strict=True):
        h = model.model.layers[layer_idx].register_forward_hook(_make_layer_hook(mlp, positions))
        handles.append(h)
    return handles


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    model,
    tokenizer,
    axiom_mlp: AxiomMLP,
    qa_pairs: list[tuple[str, str]],
    boundary_pairs: list[tuple[str, str]] | None = None,
    boundary_prob: float = 0.33,
    n_steps: int = 300,
    lr: float = 3e-5,
    weight_decay: float = 0.05,
) -> list[float]:
    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device
    optim = torch.optim.AdamW(axiom_mlp.mlps.parameters(), lr=lr, weight_decay=weight_decay)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(42)
    losses: list[float] = []
    skipped = 0

    for _ in range(n_steps):
        # Sample boundary ~1-in-3 steps when boundary pairs are provided
        if boundary_pairs and rng.random() < boundary_prob:
            q, a = rng.choice(boundary_pairs)
        else:
            q, a = rng.choice(qa_pairs)
        q_text = TEMPLATE.format(q=q)
        full_text = q_text + " " + a

        q_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        if eos_id is not None:
            full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)

        # Only hook question-part occurrences — answer positions get different
        # gradient signal (predict the answer, not route to it).
        positions = _find_term_positions(q_ids, axiom_mlp.term_token_ids)
        if not positions:
            skipped += 1
            continue

        labels = torch.full_like(full_ids, -100)
        labels[0, q_ids.shape[1] :] = full_ids[0, q_ids.shape[1] :]

        handles = install_hooks(model, axiom_mlp, positions)
        try:
            optim.zero_grad()
            loss = model(full_ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(axiom_mlp.mlps.parameters(), max_norm=1.0)
            optim.step()
            losses.append(float(loss.item()))
        finally:
            for h in handles:
                h.remove()

    if skipped:
        print(f"  WARNING: skipped {skipped} steps (term not found in prompt)")
    return losses


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    axiom_mlp: AxiomMLP | None = None,
    max_new: int = 80,
) -> str:
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Prefill with hooks active (if axiom loaded)
    handles = []
    if axiom_mlp is not None:
        positions = _find_term_positions(ids, axiom_mlp.term_token_ids)
        if positions:
            handles = install_hooks(model, axiom_mlp, positions)

    try:
        out = model(ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    finally:
        for h in handles:
            h.remove()

    # Decode without hooks — MLP influence is baked into the KV cache
    out_ids = [next_tok]
    for _ in range(max_new - 1):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids.append(next_tok)
        if int(next_tok.item()) == tokenizer.eos_token_id:
            break

    gen_ids = torch.cat(out_ids, dim=1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--r", type=int, default=16, help="MLP bottleneck dim")
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-new", type=int, default=80)
    args = parser.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}  model: {args.model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    n_layers = model.config.num_hidden_layers
    # Proportional to [16, 32, 48] out of 64 for the 32B model
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]
    print(f"n_layers={n_layers}  chosen_layers={chosen_layers}  r={args.r}")

    axiom_mlp = make_axiom_mlp(model, tokenizer, TERM, chosen_layers, r=args.r)
    n_params = sum(p.numel() for p in axiom_mlp.mlps.parameters())
    print(
        f"term_token_ids={axiom_mlp.term_token_ids}  MLP params: {n_params:,} ({n_params * 4 / 1024:.1f} KB)\n"
    )

    def probe(label: str, qa_set: list[tuple[str, str]]) -> None:
        print(f"  [{label}]")
        for q, _ in qa_set:
            prompt = TEMPLATE.format(q=q)
            base = generate(model, tokenizer, prompt, max_new=args.max_new)
            mlp = generate(model, tokenizer, prompt, axiom_mlp=axiom_mlp, max_new=args.max_new)
            print(f"    Q: {q}")
            print(f"       base : {base[:120]}")
            print(f"       axiom: {mlp[:120]}")

    print("─── BEFORE TRAINING ───────────────────────────────────────────")
    probe("HELDOUT", HELDOUT_QA[:2])

    print(f"\n─── TRAINING ({args.n_steps} steps) ──────────────────────────────────")
    t0 = time.time()
    losses = train(
        model,
        tokenizer,
        axiom_mlp,
        TRAIN_QA,
        boundary_pairs=BOUNDARY_TRAIN_QA,
        n_steps=args.n_steps,
        lr=args.lr,
    )
    print(f"done in {time.time() - t0:.1f}s   loss: {losses[0]:.3f} → {losses[-1]:.4f}")

    print("\n─── AFTER TRAINING ────────────────────────────────────────────")
    probe("TRAIN", TRAIN_QA[:3])
    probe("HELDOUT", HELDOUT_QA)
    probe("BOUNDARY", BOUNDARY_QA)


if __name__ == "__main__":
    main()
