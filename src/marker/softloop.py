"""Stage 0: soft-token feedback on a frozen model (latent-thought spec §1).

Instead of sampling a hard token each step, form a temperature-softened,
top-p-truncated distribution p over the vocabulary and feed the
probability-weighted mixture of input embeddings (p @ E) as the next input —
the model reasons over superpositions of tokens ("Soft Thinking"). Every k
steps, snap: take the hard argmax token and continue from its real embedding,
re-anchoring the chain to the token lattice. k=1 is exactly greedy decode
(the mechanical invariant asserted in the runner's smoke); k=None never snaps.

The measured quantity is the DRIFT BUDGET: how many consecutive soft steps
the chain survives before degrading, read from (a) the entropy trajectory of
p (rising = drift onset — the spec's online abort signal) and (b) the argmax
trace, which makes the chain human-readable at every step.

Zero training; inference-only; frozen weights throughout.
"""

from __future__ import annotations

import torch

# ── Pure math (model-free, unit-tested) ──────────────────────────────────────────


def soft_distribution(logits: torch.Tensor, tau: float, top_p: float) -> torch.Tensor:
    """softmax(logits/tau), top-p truncated and renormalized. The argmax always
    survives truncation, so the distribution can never be all-zero (and the
    'nearest lattice point' of the soft step equals the greedy token)."""
    p = torch.softmax(logits / tau, dim=-1)
    sorted_p, idx = torch.sort(p, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep_sorted = (cum - sorted_p) < top_p  # keep tokens whose mass STARTS inside top_p
    keep_sorted[..., 0] = True
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep.scatter_(-1, idx, keep_sorted)
    p = torch.where(keep, p, torch.zeros_like(p))
    return p / p.sum(dim=-1, keepdim=True)


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of a probability vector; 0 for one-hot."""
    return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)


def mix_embedding(p: torch.Tensor, embed_matrix: torch.Tensor) -> torch.Tensor:
    """Probability-weighted mixture of input-embedding rows: p [V] @ E [V,d]."""
    return p.to(embed_matrix.dtype) @ embed_matrix


def is_snap_step(step: int, k: int | None) -> bool:
    """Snap on every k-th step (0-indexed: steps k-1, 2k-1, ...). k=1 snaps
    every step (= greedy decode); k=None never snaps (pure latent chain)."""
    if k is None:
        return False
    return (step + 1) % k == 0


# ── The soft loop ────────────────────────────────────────────────────────────────


@torch.no_grad()
def soft_generate(
    model,  # noqa: ANN001
    ids: torch.Tensor,
    n_steps: int,
    k: int | None,
    tau: float,
    top_p: float,
    eos_id: int | None,
) -> tuple[list[int], list[float], list[bool]]:
    """Run n_steps of soft-token feedback with snap period k.

    Returns (argmax_trace, entropy_trace, snap_flags): the nearest-lattice
    token at each step (readable transcript of the chain), the entropy of the
    soft distribution at each step (drift signal), and whether each step was
    a snap (hard token fed) or soft (mixed embedding fed).

    Note: on tied-embedding checkpoints W_lm == E^T; mixing always uses the
    INPUT embedding matrix (get_input_embeddings), which is correct for both
    tied and untied variants.
    """
    device = ids.device
    embed_matrix = model.get_input_embeddings().weight  # [V, d]

    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]

    argmax_trace: list[int] = []
    entropy_trace: list[float] = []
    snap_flags: list[bool] = []

    for step in range(n_steps):
        p = soft_distribution(logits.float(), tau, top_p)
        tok = int(p.argmax().item())
        argmax_trace.append(tok)
        entropy_trace.append(float(entropy(p)))
        snap = is_snap_step(step, k)
        snap_flags.append(snap)

        if eos_id is not None and tok == eos_id:
            break

        if snap:
            step_ids = torch.tensor([[tok]], device=device)
            out = model(step_ids, past_key_values=past, use_cache=True)
        else:
            e_next = mix_embedding(p.to(device), embed_matrix).unsqueeze(0).unsqueeze(0)
            out = model(inputs_embeds=e_next, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]

    return argmax_trace, entropy_trace, snap_flags


# ── Degeneracy metrics on the argmax trace ───────────────────────────────────────


def distinct_2(tokens: list[int]) -> float:
    """Unique bigrams / total bigrams — collapses toward 0 on repetition loops."""
    if len(tokens) < 2:
        return 1.0
    bigrams = list(zip(tokens, tokens[1:], strict=False))
    return len(set(bigrams)) / len(bigrams)


def longest_run(tokens: list[int]) -> int:
    """Longest run of one repeated token — the crudest degeneration flag."""
    best = cur = 1 if tokens else 0
    for a, b in zip(tokens, tokens[1:], strict=False):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best
