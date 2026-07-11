"""Stage-2 next-thought predictor (see STAGE2_PLAN.md).

Predicts the NEXT sentence's 8 gist slots from the sequence of prior sentences'
gists — the model that learns thought-succession. Output is 8 slot vectors,
NOT a pooled vector (Fable steer #1): Stage 3 injects 8 KV slots and cannot
unpool, so the predictor must emit them directly.

Layout: a document window is [L sentences x k slots x d]. Slots are projected
to d_model, tagged with slot-index and sentence-position embeddings, flattened
to L*k tokens, and run through a block-causal transformer (a sentence attends
to all slots of sentences <= it). The readout at sentence i's k slot positions
predicts sentence i+1's k slots.

Losses (spec §3.2 killers): regression (1 - cos per slot) + InfoNCE on pooled
projections (the regression-to-the-mean / platitude guard). Metrics:
in-batch recall@k (retrieval signal) and prediction diversity (collapse guard).
All CPU-testable on tiny dims.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

_NEG = float("-inf")


# ── block-causal mask (model-free) ──────────────────────────────────────────────


def build_block_causal_mask(n_sents: int, k: int) -> torch.Tensor:
    """[T, T] additive mask, T = n_sents*k. Token (sent i, slot s) attends to
    token (sent j, slot t) iff j <= i — sentence-level causality, all slots of
    a visible sentence mutually visible. Predicting sentence i+1 from sentences
    0..i is exactly 'read out at sentence i, which saw 0..i'."""
    t = n_sents * k
    idx = torch.arange(t)
    q_sent = (idx // k).unsqueeze(1)
    k_sent = (idx // k).unsqueeze(0)
    allowed = k_sent <= q_sent
    mask = torch.zeros(t, t)
    mask.masked_fill_(~allowed, _NEG)
    return mask


# ── the predictor ───────────────────────────────────────────────────────────────


class NextThoughtPredictor(nn.Module):
    def __init__(
        self,
        d: int,
        k: int = 8,
        d_model: int = 512,
        layers: int = 6,
        heads: int = 8,
        max_sents: int = 256,
    ):
        super().__init__()
        self.k = k
        self.d = d
        self.slot_proj = nn.Linear(d, d_model)
        self.slot_emb = nn.Embedding(k, d_model)
        self.sent_pos_emb = nn.Embedding(max_sents, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, batch_first=True, norm_first=True
        )
        self.trunk = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out_proj = nn.Linear(d_model, d)
        # pooled projection for the contrastive term
        self.pool_proj = nn.Linear(k * d, d_model)

    def forward(self, gists: torch.Tensor) -> torch.Tensor:
        """gists [B, L, k, d] -> predicted next-sentence slots [B, L-1, k, d]
        (position i predicts sentence i+1)."""
        b, length, k, d = gists.shape
        device = gists.device
        x = self.slot_proj(gists)  # [B, L, k, d_model]
        x = x + self.slot_emb(torch.arange(k, device=device))  # broadcast over L
        x = x + self.sent_pos_emb(torch.arange(length, device=device)).unsqueeze(1)
        x = x.reshape(b, length * k, -1)
        mask = build_block_causal_mask(length, k).to(device)
        h = self.trunk(x, mask=mask)
        h = h.reshape(b, length, k, -1)
        out = self.out_proj(h)  # [B, L, k, d]
        return out[:, :-1]  # sentence i predicts i+1

    def pool(self, slots: torch.Tensor) -> torch.Tensor:
        """[..., k, d] -> [..., d_model] pooled projection for InfoNCE."""
        return self.pool_proj(slots.reshape(*slots.shape[:-2], self.k * self.d))


# ── losses (model-free) ─────────────────────────────────────────────────────────


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine, averaged over slots and positions. Both [..., k, d]."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def info_nce_loss(
    pred_pool: torch.Tensor, target_pool: torch.Tensor, temp: float = 0.07
) -> torch.Tensor:
    """InfoNCE over a batch of pooled vectors: each prediction must rank its
    own target above all other targets (in-batch negatives). [N, d_model]."""
    p = F.normalize(pred_pool, dim=-1)
    t = F.normalize(target_pool, dim=-1)
    logits = (p @ t.T) / temp
    labels = torch.arange(p.shape[0], device=p.device)
    return F.cross_entropy(logits, labels)


# ── metrics (model-free) ────────────────────────────────────────────────────────


@torch.no_grad()
def recall_at_k(pred_pool: torch.Tensor, target_pool: torch.Tensor, topk: int = 5) -> float:
    """Fraction of predictions whose true target is in their top-k most-similar
    targets across the batch (random = topk/N)."""
    p = F.normalize(pred_pool, dim=-1)
    t = F.normalize(target_pool, dim=-1)
    sims = p @ t.T  # [N, N]
    n = sims.shape[0]
    top = sims.topk(min(topk, n), dim=-1).indices
    hits = (top == torch.arange(n, device=sims.device).unsqueeze(1)).any(-1)
    return float(hits.float().mean())


@torch.no_grad()
def prediction_diversity(pred_pool: torch.Tensor) -> float:
    """Mean pairwise cosine of predictions across the batch. High (-> 1) means
    predictions collapsed to one vector (platitude failure)."""
    p = F.normalize(pred_pool, dim=-1)
    sims = p @ p.T
    n = sims.shape[0]
    off = (sims.sum() - sims.diag().sum()) / (n * (n - 1))
    return float(off)
