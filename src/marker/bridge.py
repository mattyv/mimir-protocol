"""Stage-3b bridge: a predicted thought (k final-layer vectors) -> injectable
per-layer K/V (see STAGE2_PLAN "Fable post-3a-i review").

The Stage-2 predictor lives in final-layer space; decoding attends to per-layer
K/V (gist_kv). The bridge maps between them, one small head per (layer, K/V):
shared per-slot trunk -> per-layer linear heads. Trained THROUGH the injection
loss — convert, inject, minimize the true next step's NLL — never by
regressing KV tensors from one vector (under-determined; Fable steer). The
injection path itself is logit-parity tested (3a-i), so this loss is exactly
the quantity the decode ceiling measured.
"""

from __future__ import annotations

import torch
from torch import nn


class GistBridge(nn.Module):
    """[k, d] final-layer thought -> AxiomKV over n_layers at k positions.

    trunk: per-slot MLP d -> width; heads: per-layer linear width -> 2*(kv_dim)
    (K and V), kv_dim = n_kv_heads*head_dim. Output reshaped to the
    [1, n_kv_heads, k, head_dim] layout gist_kv produces."""

    def __init__(
        self,
        d: int,
        k: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        width: int = 512,
    ):
        super().__init__()
        self.k, self.n_layers = k, n_layers
        self.n_kv_heads, self.head_dim = n_kv_heads, head_dim
        kv_dim = n_kv_heads * head_dim
        self.trunk = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, width))
        self.heads = nn.ModuleList(nn.Linear(width, 2 * kv_dim) for _ in range(n_layers))

    def forward(self, thought: torch.Tensor):  # noqa: ANN201
        """thought [k, d] -> AxiomKV (batch dim 1, matching gist_kv layout)."""
        from marker.run_axiom_mlp_demo import AxiomKV  # noqa: PLC0415

        h = self.trunk(thought)  # [k, width]
        keys, values = [], []
        for head in self.heads:
            kv = head(h)  # [k, 2*kv_dim]
            kmat, vmat = kv.chunk(2, dim=-1)
            # [k, kv_dim] -> [1, n_kv_heads, k, head_dim]
            keys.append(
                kmat.view(self.k, self.n_kv_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)
            )
            values.append(
                vmat.view(self.k, self.n_kv_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)
            )
        return AxiomKV(n_layers=self.n_layers, keys=keys, values=values)


def bridge_injection_nll(
    peft_model,  # noqa: ANN001
    bridge: GistBridge,
    thought: torch.Tensor,
    cont_ids: list[int],
    cont_start: int,
) -> torch.Tensor:
    """The bridge's training loss: convert the thought, inject it, and return
    the teacher-forced mean NLL of the true next step's tail (same scoring as
    _tail_nll / the 3a-i ceiling — optimize what we measure). Differentiable
    into the bridge: the frozen model attends over the injected cache, so
    gradients flow back through attention into the bridge outputs. NO detach
    anywhere on the bridge path (tested)."""
    import torch.nn.functional as F  # noqa: N812, PLC0415
    from transformers import DynamicCache  # noqa: PLC0415

    if len(cont_ids) < 2:
        raise ValueError("need >= 2 continuation tokens to score a tail")
    device = next(peft_model.parameters()).device
    kv = bridge(thought.to(device))
    cache = DynamicCache()
    for i in range(kv.n_layers):
        cache.update(kv.keys[i], kv.values[i], i)  # bridge outputs keep grad
    m = len(cont_ids) - 1
    pos = torch.arange(cont_start, cont_start + m, device=device).unsqueeze(0)
    out = peft_model(
        torch.tensor([cont_ids[:-1]], device=device),
        past_key_values=cache,
        position_ids=pos,
        use_cache=True,
    )
    return F.cross_entropy(out.logits[0], torch.tensor(cont_ids[1:], device=device))
