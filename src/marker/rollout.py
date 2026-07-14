"""Latent chain rollout (the real fast-lane test): reason in thought-space for
multiple steps with NO text in between, and measure how fast it drifts.

The chain lives entirely in summary space — the predictor's output IS a
final-layer summary [k, d], the same object it takes as input, so it can eat its
own predictions with no bridge in the loop (the bridge is only for SCORING a
thought by injecting it). Free-running:

    g1..gP        real encoded thoughts (the prefix)
    ĝ_{P+1}   = predict(g1..gP)                     <- all-real history == teacher-forced
    ĝ_{P+2}   = predict(g1..gP, ĝ_{P+1})            <- now conditioning on a PREDICTION
    ĝ_{P+3}   = predict(g1..gP, ĝ_{P+1}, ĝ_{P+2})   <- errors can compound
    ...

Drift is the gap between free-running (feed predictions back) and teacher-forcing
(always predict from the TRUE history) as depth grows. The headline number is the
depth at which an injected free-running thought stops beating the shuffled floor
— i.e. how many latent steps you can take before it's no better than noise.
"""

from __future__ import annotations

import torch

from marker.run_bridge import ladder_gap_closed, predict_step


@torch.no_grad()
def rollout(predictor, prefix: torch.Tensor, depth: int, window: int) -> torch.Tensor:  # noqa: ANN001
    """Free-running latent chain. prefix [P, k, d] = real thoughts; predict the
    next `depth` thoughts, FEEDING EACH PREDICTION BACK as history for the next.
    Returns [depth, k, d]. Each step uses only the last `window` thoughts (real
    or predicted), matching the predictor's training window; a masked dummy slot
    gives the readout a position to predict into (the block-causal mask makes its
    content irrelevant — same trick as predict_step). rollout()[0] therefore
    equals predict_step at position P (all-real history)."""
    seq = list(prefix)  # each [k, d]
    dummy = torch.zeros_like(prefix[0])
    preds = []
    for t in range(len(prefix), len(prefix) + depth):
        a = max(0, t - window + 1)
        hist = torch.stack([*seq[a:t], dummy]).unsqueeze(0)  # [1, t-a+1, k, d]
        g = predictor(hist)[0, -1]  # readout at last-history predicts step t
        preds.append(g)
        seq.append(g)  # <- the chain: prediction becomes history
    return torch.stack(preds)


@torch.no_grad()
def teacher_forced(predictor, summ: torch.Tensor, prefix_len: int, depth: int, window: int):  # noqa: ANN001
    """The control: predict the SAME steps as rollout() but always from the TRUE
    history (summ holds the real thoughts). Returns [depth, k, d]. The gap
    between this and rollout() at each depth IS the drift."""
    out = []
    for t in range(prefix_len, min(prefix_len + depth, len(summ))):
        out.append(predict_step(predictor, summ, t, window))
    return torch.stack(out) if out else summ.new_zeros(0, summ.shape[1], summ.shape[2])


def drift_by_depth(by_depth: dict[int, dict[str, list[float]]]) -> dict[int, dict]:
    """Per rollout-DEPTH, the injection ladder (none/full/gist_true/tf/free/
    shuffled) as gap_closed, plus mean cosines. by_depth[d] is a dict of
    rung -> [nll, ...] with extra keys 'free_cos'/'tf_cos' (lists of cosines).
    Returns depth -> {ladder..., free_cos, tf_cos, n}. This is the drift curve:
    read 'free' falling toward 'shuffled' as depth grows."""
    out = {}
    for d in sorted(by_depth):
        rec = by_depth[d]
        cos = {k: rec.pop(k, []) for k in ("free_cos", "tf_cos")}
        ladder = ladder_gap_closed({r: v for r, v in rec.items() if v})
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        out[d] = {
            **{r: ladder[r]["gap_closed"] for r in ladder},
            "free_cos": mean(cos["free_cos"]),
            "tf_cos": mean(cos["tf_cos"]),
            "n": len(cos["free_cos"]),
        }
    return out
