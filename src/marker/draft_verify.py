"""Stage-3b draft-and-verify: chain thoughts, keep the model honest.

Runtime loop (the speed thesis): predict the next thought cheaply, draft a few
candidate steps from it (frozen model + injected thought), then VERIFY each
against the real reasoning-so-far and keep the best. Drafting is cheap
(short); verify is one full forward — the spec-decode shape. The win is real
only if drafts are good enough that verify accepts them.

This module holds the model-free pieces (advance metric, selection). The
model-coupled draft/verify/roll-out live in run_draft_verify.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def advance_rate(
    drafts: Sequence,
    cur_steps: Sequence,
    next_steps: Sequence,
    sim: Callable,
) -> float:
    """Fraction of drafts closer to the NEXT step than the CURRENT one —
    Fable's forward-motion gate. 3a-i showed decode overlaps the current step
    (0.458) about as much as the next (0.412), so chaining REQUIRES drafts that
    move on rather than restate. sim(a, b) is any similarity (higher = closer);
    a draft advances iff sim(draft, next) > sim(draft, cur)."""
    if not drafts:
        return 0.0
    adv = sum(
        1
        for d, cur, nxt in zip(drafts, cur_steps, next_steps, strict=True)
        if sim(d, nxt) > sim(d, cur)
    )
    return adv / len(drafts)


def pick_by_score(candidates: Sequence, scores: Sequence) -> tuple:
    """Return (best_candidate, index) with the LOWEST score (e.g. NLL under the
    real context — the verify signal). Ties take the first."""
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], best_idx
