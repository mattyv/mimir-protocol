"""Fast-lane confidence probe (FASTLANE_PLAN "cheap-first probe").

The fast lane's whole thesis: don't require the predictor to be RIGHT, require
it to know WHEN it is. Accept a predicted next-thought when confidence is high
(cheap latent step, skip the big model), fall back to full generation when it's
low. That only works if some INFERENCE-TIME confidence signal — computable
without knowing the true next thought — actually tracks correctness.

This module is the diagnostic that decides it. For each (thought -> next-thought)
prediction on held-out chains it computes:
  - correctness: is the predictor's top-1 pick the true next thought, among
    SAME-DOCUMENT candidates (the topic-shortcut control, per run_stage2)?
  - confidence signal(s):
      * prediction_norm      — cheap, needs only the prediction
      * dropout_agreement    — MC-dropout stability, needs only the model
      * retrieval_margin     — top1-vs-top2 sharpness against a candidate bank
Then tercile_report / rank_auc measure whether confidence SEPARATES correct
from wrong. All model-free and CPU-testable; the probe run wires them to the
real trained predictor (run_confidence.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

# ── correctness (per-example, within-document pool) ──────────────────────────


@torch.no_grad()
def within_doc_correct(p: torch.Tensor, t: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Per-example top-1 correctness with the candidate pool restricted to
    SAME-DOCUMENT targets — the succession signal stripped of the 'found the
    right document' shortcut (see run_stage2._recall_within_doc). p, t are
    [N, d] pooled predictions/targets; doc is [N] document ids. Returns a bool
    [N]: True iff example i's own target outscores every other same-doc target.
    (A singleton doc is trivially correct — its only candidate is the truth.)"""
    pn = F.normalize(p, dim=-1)
    tn = F.normalize(t, dim=-1)
    sims = pn @ tn.T  # [N, N]
    same = doc.unsqueeze(0) == doc.unsqueeze(1)
    sims = sims.masked_fill(~same, float("-inf"))
    top1 = sims.argmax(dim=-1)
    return top1 == torch.arange(sims.shape[0], device=sims.device)


# ── confidence signals ───────────────────────────────────────────────────────


@torch.no_grad()
def prediction_norm(pred_pool: torch.Tensor) -> torch.Tensor:
    """L2 norm of each pooled prediction [N, d] -> [N]. The cheapest signal:
    the model's output magnitude, no extra passes, no candidate bank."""
    return pred_pool.norm(dim=-1)


@torch.no_grad()
def retrieval_margin(pred_pool: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Top1-vs-top2 cosine margin against a candidate bank [M, d]. A confident
    prediction is a sharp peak (one bank entry clearly closest); a diffuse
    prediction ties several. Returns [N] = (best cosine - second-best cosine).
    Needs a bank of candidate thoughts at inference (a memory of plausible next
    thoughts) — realistic if the fast lane retrieves rather than free-generates."""
    pn = F.normalize(pred_pool, dim=-1)
    bn = F.normalize(bank, dim=-1)
    sims = pn @ bn.T  # [N, M]
    top2 = sims.topk(min(2, sims.shape[1]), dim=-1).values
    if top2.shape[1] < 2:
        return torch.zeros(sims.shape[0], device=sims.device)
    return top2[:, 0] - top2[:, 1]


@torch.no_grad()
def mc_dropout_pool(model, gists: torch.Tensor, n_samples: int = 8) -> torch.Tensor:  # noqa: ANN001
    """Run the predictor n_samples times with DROPOUT ACTIVE (train mode, no
    grad) and return the pooled predictions stacked as [n_samples, N, d_model],
    N = B*(L-1). Dropout makes each pass a slightly different sub-model; if they
    all agree on the next thought, the prediction is robust. Restores the
    caller's train/eval mode."""
    was_training = model.training
    model.train()  # enable dropout (no_grad already blocks grad)
    out = []
    for _ in range(n_samples):
        pred = model(gists)  # [B, L-1, k, d]
        pooled = model.pool(pred).reshape(-1, model.pool_proj.out_features)
        out.append(pooled)
    model.train(was_training)
    return torch.stack(out)


@torch.no_grad()
def dropout_agreement(samples: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine across MC-dropout samples, per example. samples is
    [n_samples, N, d]. High (-> 1) = the dropout ensemble agrees on the next
    thought (confident); low = it scatters (uncertain). Returns [N]."""
    s = F.normalize(samples, dim=-1)  # [S, N, d]
    sims = torch.einsum("snd,tnd->nst", s, s)  # [N, S, S] pairwise cosines
    ns = sims.shape[1]
    if ns < 2:
        return torch.ones(sims.shape[0], device=sims.device)
    off = (sims.sum(dim=(1, 2)) - sims.diagonal(dim1=1, dim2=2).sum(dim=1)) / (ns * (ns - 1))
    return off


# ── separation metrics ───────────────────────────────────────────────────────


@torch.no_grad()
def rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """AUC of `scores` predicting binary `labels` (1 = correct) via the
    Mann-Whitney rank statistic, ties averaged. AUC = P(score of a correct >
    score of a wrong). 1.0 = confidence perfectly ranks correctness; 0.5 =
    useless; <0.5 = anti-correlated. Returns None when labels are all-one or
    all-zero (AUC undefined)."""
    labels = labels.to(torch.bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # average ranks (1-based) with tie handling
    order = scores.argsort()
    ranks = torch.empty_like(scores, dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    # average ranks within tied groups
    uniq, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    rank_sum = torch.zeros(len(uniq), dtype=torch.double).scatter_add_(0, inv, ranks)
    avg = rank_sum / counts
    ranks = avg[inv]
    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def tercile_report(confidence: torch.Tensor, correct: torch.Tensor) -> dict:
    """Bucket examples into low/mid/high thirds by confidence and report the
    correct-rate in each, plus overall AUC. The PASS shape: a high tercile whose
    accuracy is well above the low tercile (confidence is a usable gate). `n`
    counts examples; `base` is the overall correct-rate for reference."""
    correct = correct.to(torch.double)
    n = len(confidence)
    order = confidence.argsort()
    c = correct[order]
    third = n // 3
    lo, mid, hi = c[:third], c[third : 2 * third], c[2 * third :]
    acc = lambda x: round(float(x.mean()), 3) if len(x) else None  # noqa: E731
    auc = rank_auc(confidence, correct.to(torch.bool))
    return {
        "n": n,
        "base": round(float(correct.mean()), 3),
        "low": {"acc": acc(lo), "n": len(lo)},
        "mid": {"acc": acc(mid), "n": len(mid)},
        "high": {"acc": acc(hi), "n": len(hi)},
        "auc": round(auc, 3) if auc is not None else None,
    }
