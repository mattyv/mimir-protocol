"""Anchored-burst schedule + answer scoring (run_burst.py helpers).

The rollout showed open-loop latent chaining drifts after ~2 steps but that a
real step RESETS the drift. The burst test measures whether *interleaving* cheap
latent steps between real (decoded) anchor steps buys a speed-up at equal
accuracy — end to end, on real generation, scored on the final answer.

These helpers are the model-free parts: which steps are anchors vs latent, and
pulling/comparing the numeric answer. The generation loop lives in run_burst.py.
"""

from __future__ import annotations

import re

_ANS = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def make_schedule(n_steps: int, anchor_every: int) -> list[str]:
    """Label each of n_steps as 'anchor' (decode it as text) or 'latent' (inject
    a predicted thought, skip decoding). Step 0 is always an anchor (generation
    must start from real text), then every `anchor_every`-th step re-anchors.
    anchor_every=1 -> all anchors (== plain generation); large -> mostly latent."""
    if anchor_every < 1:
        raise ValueError("anchor_every must be >= 1")
    return ["anchor" if (i % anchor_every == 0) else "latent" for i in range(n_steps)]


def extract_answer(text: str) -> str | None:
    """The final numeric answer from a solution. Prefer the GSM8K '#### x'
    marker; else fall back to the last number in the text. Commas stripped so
    '1,000' == '1000'. None if no number present."""
    if "####" in text:
        tail = text.split("####")[-1]
        m = _ANS.search(tail)
        if m:
            return m.group(0).replace(",", "")
    nums = _ANS.findall(text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: str | None, gold: str | None) -> bool:
    """Numeric-equality match (tolerant of trailing .0 and thousands commas).
    Both must parse as numbers; a missing prediction is a miss, not a match."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()
