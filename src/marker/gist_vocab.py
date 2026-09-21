"""G7 vocabulary wiring (Stage 2+3 design v3, see GIST_LM_PLAN.md "STAGES 2+3
DESIGN v3"): extends the frozen 7B's vocabulary with 8 slots x K gist-dictionary
ids plus <think>/<commit>, WITHOUT `resize_token_embeddings` and WITHOUT any
in-place row write into the base embedding/lm_head weight (both are invisible
to autograd -- the vocabulary side would silently never train).

Wiring (option A'): `GistEmbedWrapper`/`GistHeadWrapper` are installed via
`set_input_embeddings`/`set_output_embeddings` BEFORE `get_peft_model`; ids
below base_vocab still read the frozen base row; ids at/above base_vocab are
computed ON THE FLY, differentiably, from a small parameterization of the
frozen dictionary means (`GistVocab.mu`, the per-slot `mu_readout` loaded from
`dict_kv_K4096.pt`).

Flat id layout: flat_id(s, j) = base_vocab + s*K + j (s = NATURAL slot index,
0..7; j = dictionary id, 0..K-1); <think> = base_vocab + 8*K; <commit> =
<think> + 1. base_vocab is always read from the model/tokenizer at runtime
(config.vocab_size) -- never hardcoded, so the same code runs on the tiny
smoke model (vocab_size=200) and the real 7B (152,064).

GistVocab per-id row parameterization (design doc "STAGES 2+3 DESIGN v3"):
  input_row(s, j)  = A'_s . mu[s,j] + U_s . c_in[s,j]     (A'_s = A + P_s Q_s)
  output_row(s, j) = B'_s . mu[s,j] + V_s . c_out[s,j] + bias[s,j]
                                                      (B'_s = B + Pout_s Qout_s)
A, B are SHARED across slots (init as a scaled identity so a fresh mu starts
roughly embedding-scaled); P_s/Q_s and Pout_s/Qout_s are per-slot
rank-`r_slot` corrections on the input resp. output side (Q/Qout zero-init,
so each starts as a no-op -- the design doc's formula names BOTH A'_s and
B'_s, and the ~37M param budget only closes with both); U_s/V_s are per-slot
rank-`r_id` bases turning a small per-id code (c_in/c_out, zero-init) into a
full-width delta, UNTIED between input and output (separate matrices,
separate codes) per "STAGE 1b RESULT"'s slot-7/slot-3 blur finding. "Output
rows keep μ" holds throughout: μ is a frozen buffer, only the projections
around it train.
"""

from __future__ import annotations

import torch
from torch import nn

N_SLOTS = 8
# Emission/layout order, descending informativeness (GIST_LM_PLAN.md STAGE 1
# RESULT: "op lives in slot 7"). One constant, shared by training layout,
# loss and the generation-time logits processor -- see slot_for_position.
SLOT_ORDER = [7, 1, 0, 3, 6, 2, 4, 5]
assert sorted(SLOT_ORDER) == list(range(N_SLOTS))  # a fixed permutation of 0..7


def slot_for_position(t_since_think: int) -> int:
    """Which NATURAL dictionary slot owns the gist token `t_since_think`
    positions after <think> (0-indexed, wraps every 8). The ONE function
    shared by build_sequence/two_gather_loss (training) and
    GistLogitsProcessor (generation) -- if they ever disagreed here, training
    and inference would silently score/mask different slots."""
    if t_since_think < 0:
        raise ValueError(f"t_since_think must be >= 0, got {t_since_think}")
    return SLOT_ORDER[t_since_think % N_SLOTS]


def gammas(
    embed_weight: torch.Tensor, head_weight: torch.Tensor, mu: torch.Tensor
) -> tuple[float, float]:
    """(emb_std/mu_std, head_std/mu_std) -- the scale factors that make A's
    and B's identity init land the FIRST forward pass' gist rows in the same
    ballpark as the base model's own embedding/lm_head rows, rather than
    swamping (or being swamped by) attention at step 0."""
    mu_std = mu.detach().float().std().item()
    if mu_std == 0:
        raise ValueError("mu has zero std; cannot compute a scale-matched gamma")
    emb_std = embed_weight.detach().float().std().item()
    head_std = head_weight.detach().float().std().item()
    return emb_std / mu_std, head_std / mu_std


class GistVocab(nn.Module):
    """The trainable parameterization of the 8*K gist ids + <think>/<commit>.
    All math fp32 (`self.mu` and the parameters below); callers cast the
    OUTPUT rows to the base model's compute dtype (bf16) at the point of use.
    Frozen: `mu` (the dictionary's per-slot mean readout, a buffer, not a
    Parameter -- it never trains). Trainable: everything else.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        d_model: int,
        d_out: int,
        base_vocab: int,
        r_slot: int = 64,
        r_id: int = 32,
        gamma_in: float = 1.0,
        gamma_out: float = 1.0,
    ) -> None:
        super().__init__()
        if mu.ndim != 3 or mu.shape[0] != N_SLOTS:
            raise ValueError(f"mu must be [{N_SLOTS}, K, d_mu], got {tuple(mu.shape)}")
        n_slots, K, d_mu = mu.shape
        self.register_buffer("mu", mu.detach().float())
        self.n_slots = n_slots
        self.K = K
        self.d_mu = d_mu
        self.d_model = d_model
        self.d_out = d_out
        self.base_vocab = base_vocab
        self.think_id = base_vocab + n_slots * K
        self.commit_id = self.think_id + 1

        self.A = nn.Parameter(gamma_in * torch.eye(d_model, d_mu))
        self.B = nn.Parameter(gamma_out * torch.eye(d_out, d_mu))
        self.P = nn.Parameter(torch.randn(n_slots, d_model, r_slot) * 0.01)
        self.Q = nn.Parameter(torch.zeros(n_slots, r_slot, d_mu))
        self.P_out = nn.Parameter(torch.randn(n_slots, d_out, r_slot) * 0.01)
        self.Q_out = nn.Parameter(torch.zeros(n_slots, r_slot, d_mu))
        self.U = nn.Parameter(torch.randn(n_slots, d_model, r_id) * 0.01)
        self.V = nn.Parameter(torch.randn(n_slots, d_out, r_id) * 0.01)
        self.c_in = nn.Parameter(torch.zeros(n_slots, K, r_id))
        self.c_out = nn.Parameter(torch.zeros(n_slots, K, r_id))
        self.bias = nn.Parameter(torch.zeros(n_slots, K))
        self.think_in = nn.Parameter(torch.randn(d_model) * 0.01)
        self.commit_in = nn.Parameter(torch.randn(d_model) * 0.01)
        self.think_out = nn.Parameter(torch.randn(d_out) * 0.01)
        self.commit_out = nn.Parameter(torch.randn(d_out) * 0.01)
        self.think_commit_bias = nn.Parameter(torch.zeros(2))

    def flat_id(self, s: int, j: int) -> int:
        return self.base_vocab + s * self.K + j

    def input_rows(self, flat_ids: torch.Tensor) -> torch.Tensor:
        """flat_ids [n] (long, values in [base_vocab, base_vocab+8K+2)) ->
        [n, d_model] fp32, differentiable w.r.t. every trainable parameter
        above. Gist ids: A'_s . mu + U_s . c_in (per-slot low-rank
        correction + per-id low-rank delta). <think>/<commit>: their own
        full-width rows."""
        if flat_ids.numel() == 0:
            return flat_ids.new_zeros((0, self.d_model), dtype=torch.float32)
        if (flat_ids < self.base_vocab).any() or (flat_ids > self.commit_id).any():
            raise ValueError("input_rows got an id outside the gist/<think>/<commit> range")
        is_think = flat_ids == self.think_id
        is_commit = flat_ids == self.commit_id
        rel = (flat_ids - self.base_vocab).clamp(min=0, max=self.n_slots * self.K - 1)
        slot = torch.div(rel, self.K, rounding_mode="floor")
        idx = rel % self.K

        mu_vecs = self.mu[slot, idx]  # [n, d_mu]
        base = mu_vecs @ self.A.T  # [n, d_model]
        p_sel = self.P[slot]  # [n, d_model, r_slot]
        q_sel = self.Q[slot]  # [n, r_slot, d_mu]
        tmp = torch.einsum("nrd,nd->nr", q_sel, mu_vecs)
        corr = torch.einsum("ndr,nr->nd", p_sel, tmp)
        u_sel = self.U[slot]  # [n, d_model, r_id]
        c_in_sel = self.c_in[slot, idx]  # [n, r_id]
        delta = torch.einsum("ndr,nr->nd", u_sel, c_in_sel)
        rows = base + corr + delta

        rows = torch.where(is_think.unsqueeze(-1), self.think_in.unsqueeze(0).expand_as(rows), rows)
        rows = torch.where(
            is_commit.unsqueeze(-1), self.commit_in.unsqueeze(0).expand_as(rows), rows
        )
        return rows

    def output_rows_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All 8*K+2 output rows + biases at once (the lm_head needs logits
        for every id every forward pass). ([8K+2, d_out], [8K+2])."""
        mu_flat = self.mu.reshape(self.n_slots * self.K, self.d_mu)
        base = mu_flat @ self.B.T  # [n_slots*K, d_out]
        # per-slot rank-r_slot correction on B (B'_s = B + Pout_s Qout_s)
        tmp = torch.einsum("srd,skd->skr", self.Q_out, self.mu)  # [n_slots, K, r_slot]
        corr = torch.einsum("sdr,skr->skd", self.P_out, tmp)  # [n_slots, K, d_out]
        delta = torch.einsum("sdr,skr->skd", self.V, self.c_out)  # [n_slots, K, d_out]
        rows = base + (corr + delta).reshape(self.n_slots * self.K, self.d_out)
        bias = self.bias.reshape(-1)
        rows = torch.cat([rows, self.think_out.unsqueeze(0), self.commit_out.unsqueeze(0)], dim=0)
        bias = torch.cat([bias, self.think_commit_bias], dim=0)
        return rows, bias


class GistEmbedWrapper(nn.Module):
    """Replaces the base model's input embedding. ids < base_vocab -> the
    frozen base row (normal embedding lookup); ids >= base_vocab ->
    `vocab.input_rows`, cast to the base embedding's dtype. Assembled with
    (non-in-place) `index_put` into a fresh tensor -- writing into
    `base_embed.weight` in place would be invisible to autograd and would
    corrupt the frozen text rows besides."""

    def __init__(self, base_embed: nn.Module, vocab: GistVocab) -> None:
        super().__init__()
        self.base_embed = base_embed
        self.vocab = vocab

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        orig_shape = ids.shape
        flat = ids.reshape(-1)
        dtype = self.base_embed.weight.dtype
        out = torch.zeros(flat.shape[0], self.vocab.d_model, dtype=dtype, device=flat.device)
        base_pos = (flat < self.vocab.base_vocab).nonzero(as_tuple=True)[0]
        gist_pos = (flat >= self.vocab.base_vocab).nonzero(as_tuple=True)[0]
        if base_pos.numel():
            out = out.index_put((base_pos,), self.base_embed(flat[base_pos]).to(dtype))
        if gist_pos.numel():
            rows = self.vocab.input_rows(flat[gist_pos]).to(dtype)
            out = out.index_put((gist_pos,), rows)
        return out.reshape(*orig_shape, self.vocab.d_model)


class GistHeadWrapper(nn.Module):
    """Replaces the base model's lm_head. logits = cat([base_head(h), h @
    gist_rows.T + gist_bias], dim=-1) -- one wider softmax over base_vocab +
    8K + 2 classes; `.weight` exposes the BASE head's weight (unchanged
    shape) for any code that inspects it (e.g. tied-weight bookkeeping)."""

    def __init__(self, base_head: nn.Module, vocab: GistVocab) -> None:
        super().__init__()
        self.base_head = base_head
        self.vocab = vocab

    @property
    def weight(self) -> torch.Tensor:
        return self.base_head.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_head(hidden)
        rows, bias = self.vocab.output_rows_all()
        rows = rows.to(hidden.dtype)
        bias = bias.to(hidden.dtype)
        gist_logits = hidden @ rows.T + bias
        return torch.cat([base_logits, gist_logits], dim=-1)


def attach_g7(
    base_model,  # noqa: ANN001
    mu: torch.Tensor,
    r_lora: int = 16,
    targets: list[str] | None = None,
    r_slot: int = 64,
    r_id: int = 32,
) -> tuple[object, GistVocab]:
    """Install the gist vocabulary (embed + head wrappers), extend
    config.vocab_size, then wrap in a LoRA adapter named "g7". Order matters:
    the wrappers go in BEFORE `get_peft_model` so PEFT's module walk sees
    them as part of the base tree (not adapter-only). `get_peft_model`
    freezes everything not named `lora_*` -- including the freshly-attached
    GistVocab, which lives in that same tree -- so its parameters are
    explicitly re-enabled for grad afterward."""
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    base_embed = base_model.get_input_embeddings()
    base_head = base_model.get_output_embeddings()
    base_vocab = base_model.config.vocab_size  # read at runtime, never hardcoded
    d_model = base_embed.weight.shape[1]
    d_out = base_head.weight.shape[1]  # lm_head input dim == hidden size
    gamma_in, gamma_out = gammas(base_embed.weight, base_head.weight, mu)
    vocab = GistVocab(
        mu,
        d_model,
        d_out,
        base_vocab,
        r_slot=r_slot,
        r_id=r_id,
        gamma_in=gamma_in,
        gamma_out=gamma_out,
    )
    # The vocab must live where the base embedding lives (cuda:0 under
    # device_map={"": 0}): its masters stay fp32, but indexing self.mu with a
    # cuda `slot` tensor -- or matmul-ing cuda hidden states against cpu rows
    # -- is a device-mismatch crash the all-CPU tiny-model tests can never
    # see. ~37M fp32 params + the mu buffer on GPU is well under 1 GB.
    vocab = vocab.to(base_embed.weight.device)
    base_model.set_input_embeddings(GistEmbedWrapper(base_embed, vocab))
    base_model.set_output_embeddings(GistHeadWrapper(base_head, vocab))
    base_model.config.vocab_size = base_vocab + vocab.n_slots * vocab.K + 2

    cfg = LoraConfig(
        r=r_lora,
        lora_alpha=r_lora * 2,
        target_modules=targets
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, cfg, adapter_name="g7")
    for p in vocab.parameters():
        p.requires_grad_(True)
    return peft_model, vocab
