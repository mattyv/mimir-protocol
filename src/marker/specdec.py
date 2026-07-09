"""Speculative draft-and-verify decoding (latent-thought spec, Stage 3b mechanics).

A tiny drafter proposes gamma tokens; the big frozen verifier checks them all
in ONE parallel forward pass (prefill-shaped: weights streamed once per round,
not once per token). At the first mismatch the draft is cut, and the
verifier's own argmax at that position — already computed in the same pass —
is taken as a free correct token. Greedy-only: acceptance rule is exact
argmax match, so the final output is byte-identical to the verifier's own
greedy decode; only the number of verifier passes varies with drafter skill.

This module validates the MECHANISM with an off-the-shelf drafter (no gist
conditioning — that needs Stage-1 training). The acceptance rate measured
here is the baseline a thought-conditioned drafter must beat.

Caveat on identity: exact under exact arithmetic. In floating point, the
verifier's batch (prefill) and incremental (decode) code paths can round
differently and flip a near-tied argmax; the runner reports token-exact
identity per prompt rather than hard-asserting it on GPU.
"""

from __future__ import annotations

import torch

# ── Pure accounting (model-free, unit-tested) ────────────────────────────────────


def accept_prefix(draft: list[int], picks: list[int]) -> tuple[int, int]:
    """Given the drafted tokens and the verifier's argmax picks (one per draft
    position, plus one for the position after the full draft), return
    (n_accepted, free_token). n_accepted is the length of the matching prefix;
    free_token is the verifier's own pick at the first divergence (or the
    bonus pick after the whole draft when everything matched)."""
    n = 0
    for d, p in zip(draft, picks, strict=False):
        if d != p:
            break
        n += 1
    return n, picks[n]


def trim_at_eos(tokens: list[int], eos_id: int | None) -> list[int]:
    """Cut the sequence just after the first EOS (inclusive), if present."""
    if eos_id is None:
        return tokens
    if eos_id in tokens:
        return tokens[: tokens.index(eos_id) + 1]
    return tokens


# ── Decoding loops ───────────────────────────────────────────────────────────────


@torch.no_grad()
def greedy_decode(model, ids: torch.Tensor, max_new: int, eos_id: int | None) -> list[int]:  # noqa: ANN001
    """Vanilla incremental greedy decode (the identity reference): one full
    weight-stream per generated token."""
    out = model(ids, use_cache=True)
    past = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax().item())
    generated = [next_tok]
    while len(generated) < max_new and next_tok != eos_id:
        step = torch.tensor([[next_tok]], device=ids.device)
        out = model(step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        generated.append(next_tok)
    return trim_at_eos(generated, eos_id)


@torch.no_grad()
def _draft_tokens(
    drafter,  # noqa: ANN001
    prefix_ids: torch.Tensor,
    gamma: int,
    eos_id: int | None,
) -> list[int]:
    """Drafter greedily proposes up to gamma continuation tokens (stops early
    at EOS). Re-prefills each round — the drafter is cheap by construction."""
    out = drafter(prefix_ids, use_cache=True)
    past = out.past_key_values
    tok = int(out.logits[0, -1].argmax().item())
    draft = [tok]
    while len(draft) < gamma and tok != eos_id:
        step = torch.tensor([[tok]], device=prefix_ids.device)
        out = drafter(step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = int(out.logits[0, -1].argmax().item())
        draft.append(tok)
    return draft


@torch.no_grad()
def spec_decode(
    verifier,  # noqa: ANN001
    drafter,  # noqa: ANN001
    ids: torch.Tensor,
    max_new: int,
    gamma: int,
    eos_id: int | None,
) -> tuple[list[int], dict]:
    """Draft-and-verify greedy decode. Returns (generated_ids, stats).

    Each round: drafter proposes up to gamma tokens; the verifier runs ONE
    forward pass over prefix+draft; the matching prefix is accepted and the
    verifier's pick at the divergence (or after the full draft) arrives free.
    The verifier is re-prefilled per round — correctness-first; the metric of
    interest (verifier passes per token) is unaffected by cache reuse.
    """
    device = ids.device
    prefix = ids[0].tolist()
    prompt_len = len(prefix)
    generated: list[int] = []
    passes = drafted_total = accepted_total = 0

    while len(generated) < max_new:
        draft = _draft_tokens(drafter, torch.tensor([prefix], device=device), gamma, eos_id)
        full = torch.tensor([prefix + draft], device=device)
        logits = verifier(full).logits[0]
        # picks[i] = verifier's next-token argmax after consuming prefix+draft[:i]
        picks = [int(logits[len(prefix) - 1 + i].argmax().item()) for i in range(len(draft) + 1)]

        n_accepted, free_tok = accept_prefix(draft, picks)
        passes += 1
        drafted_total += len(draft)
        accepted_total += n_accepted

        segment = draft[:n_accepted] + [free_tok]
        prefix += segment
        generated = prefix[prompt_len:]
        if eos_id is not None and eos_id in segment:
            break

    generated = trim_at_eos(generated, eos_id)[:max_new]
    stats = {
        "passes": passes,
        "drafted": drafted_total,
        "accepted": accepted_total,
        "tokens": len(generated),
        "acceptance_rate": accepted_total / drafted_total if drafted_total else 0.0,
        "tokens_per_pass": len(generated) / passes if passes else 0.0,
    }
    return generated, stats
