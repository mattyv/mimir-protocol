"""Stage-1 gist model: LoRA + learned gist embeddings on a frozen base, the
batched training forward, and the 3-PPL eval (see GIST_PILOT_PLAN.md).

Trainables = LoRA adapters + the k gist embeddings ONLY; the base is frozen.
The 4-bit quantization for the real 7B run lives in the runner — this module
takes an already-loaded base model so it stays CPU-testable on a tiny model.

Fable build-notes honored here:
 #1 4D masks need an explicit attention path — assert_attn_impl() refuses
    flash-attention-2 (which would silently ignore the mask).
 #2 batched per-sample masks (marker.gist.build_batch_mask), use_cache=False.
 #3 the three eval PPLs keep C at identical positions, varying only cont_sees.
 #4 resume without optimizer state is acceptable for the pilot (runner concern).
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model

from marker.gist import build_batch_labels, build_batch_mask

QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def assert_attn_impl(model) -> None:  # noqa: ANN001
    """Refuse flash-attention-2: it ignores the 4D additive mask, which would
    silently break the gist bottleneck and fake a 'gist works' result. sdpa and
    eager both honor a custom 4D mask."""
    impl = getattr(model.config, "_attn_implementation", "eager")
    if impl not in ("sdpa", "eager"):
        raise ValueError(
            f"attn_implementation={impl!r} ignores 4D masks; load with "
            "attn_implementation='sdpa' (or 'eager')."
        )


def attach_gist(
    base_model,  # noqa: ANN001
    gist_k: int,
    r: int = 16,
    alpha: int = 32,
    targets: list[str] | None = None,
) -> tuple[object, torch.nn.Parameter]:
    """Wrap base_model with LoRA and create k learned gist embeddings. Returns
    (peft_model, gist_param). Base frozen; only LoRA + gist require grad."""
    assert_attn_impl(base_model)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets or QWEN_TARGETS,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, cfg)

    hidden = base_model.get_input_embeddings().weight.shape[1]
    # Stat-matched small init (the embedding std is a reasonable scale so the
    # gist slots start in-distribution rather than swamping attention).
    emb_std = base_model.get_input_embeddings().weight.std().item()
    gist_param = torch.nn.Parameter(torch.randn(gist_k, hidden, dtype=torch.float32) * emb_std)
    return peft_model, gist_param


def to_leaf_param(param: torch.nn.Parameter, device) -> torch.nn.Parameter:  # noqa: ANN001
    """Move a Parameter to `device` and keep it a LEAF (optimizable). A plain
    `param.to(cuda)` returns a NON-leaf tensor (result of the move op), which
    AdamW rejects with 'can't optimize a non-leaf Tensor' — and CPU `.to(cpu)`
    is a no-op that hides the bug in tests. Re-wrap the moved data as a fresh
    Parameter so it's a leaf again."""
    return torch.nn.Parameter(param.detach().to(device))


def trainable_param_names(peft_model) -> list[str]:  # noqa: ANN001
    """Names of base/LoRA params that require grad (the gist param is separate
    and always trainable — tested directly)."""
    return [n for n, p in peft_model.named_parameters() if p.requires_grad]


def _pad(seqs: list[list[int]], max_len: int, pad_id: int, device) -> torch.Tensor:  # noqa: ANN001
    return torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in seqs], dtype=torch.long, device=device
    )


def gist_forward(
    peft_model,  # noqa: ANN001
    gist_param: torch.nn.Parameter,
    spans: list[list[int]],
    conts: list[list[int]],
    *,
    cont_sees: frozenset[str] = frozenset({"gist"}),
    pad_id: int = 0,
) -> torch.Tensor:
    """One batched forward: splice [span | gist | cont] embeddings, apply the
    4D mask for `cont_sees`, CE on continuation tokens only. Returns the loss
    (mean CE over real continuation positions)."""
    device = next(peft_model.parameters()).device
    k = gist_param.shape[0]
    span_lens = [len(s) for s in spans]
    cont_lens = [len(c) for c in conts]
    max_s, max_c = max(span_lens), max(cont_lens)

    embed = peft_model.get_input_embeddings()
    span_e = embed(_pad(spans, max_s, pad_id, device))
    cont_e = embed(_pad(conts, max_c, pad_id, device))
    gist_e = gist_param.to(span_e.dtype).unsqueeze(0).expand(len(spans), -1, -1)
    inputs_embeds = torch.cat([span_e, gist_e, cont_e], dim=1)

    mask = build_batch_mask(
        span_lens, cont_lens, k, max_s, max_c, cont_sees=cont_sees, dtype=inputs_embeds.dtype
    ).to(device)
    labels = build_batch_labels(conts, max_s, k, max_c).to(device)
    t = inputs_embeds.shape[1]
    position_ids = torch.arange(t, device=device).unsqueeze(0).expand(len(spans), -1)

    out = peft_model(
        inputs_embeds=inputs_embeds,
        attention_mask=mask,
        position_ids=position_ids,
        labels=labels,
        use_cache=False,
    )
    return out.loss


@torch.no_grad()
def three_ppls(
    peft_model,  # noqa: ANN001
    gist_param: torch.nn.Parameter,
    spans: list[list[int]],
    conts: list[list[int]],
    pad_id: int = 0,
) -> dict[str, float]:
    """Perplexity of the continuation under three conditions at IDENTICAL
    positions: gist (C sees only the k gist slots), full (C sees the raw span
    — upper bound), none (C sees neither — lower bound). Returns
    {'gist','full','none'} PPLs. Sanity direction: full <= none."""
    out = {}
    for name, sees in [
        ("gist", frozenset({"gist"})),
        ("full", frozenset({"gist", "span"})),
        ("none", frozenset()),
    ]:
        loss = gist_forward(peft_model, gist_param, spans, conts, cont_sees=sees, pad_id=pad_id)
        out[name] = float(torch.exp(loss))
    return out


def cross_doc_spans(docs: list[list[tuple[list[int], list[int]]]]) -> list[list[int]]:
    """The TRUE null for the shuffled-gist control (gate-review correction):
    for the flattened pair list of doc-grouped heldout, return a donor span
    from a DIFFERENT document per pair (doc j borrows from doc j+1, index-
    matched modulo donor length). The within-batch roll_spans control turned
    out to donate a same-document NEIGHBOR — topically overlapping the gold
    continuation — so it measures 'nearby context helps', not slot-presence.
    Cross-document donors share no document context; if their gists still
    close the gap, THAT would be the artifact signal."""
    donor_spans = [[p[0] for p in d] for d in docs]
    out: list[list[int]] = []
    for j, d in enumerate(docs):
        donor = donor_spans[(j + 1) % len(docs)]
        for i in range(len(d)):
            out.append(donor[i % len(donor)])
    return out


def roll_spans(spans: list[list[int]]) -> list[list[int]]:
    """Roll the spans by one within a batch, keeping continuations in place —
    the SHUFFLED-GIST control (Fable gate review): each continuation now sees a
    gist computed from a DIFFERENT sentence's span. If shuffled ~= none, the
    gist genuinely carries the span's content; if shuffled ~= gist, the headline
    was just 'having a warm prefix helps'. Needs >= 2 spans to be a real
    permutation (single-item batches are dropped by the caller)."""
    return spans[-1:] + spans[:-1]


@torch.no_grad()
def encode_gist(
    peft_model,  # noqa: ANN001
    gist_param: torch.nn.Parameter,
    spans: list[list[int]],
    pad_id: int = 0,
) -> torch.Tensor:
    """The Stage-2 encode: run [span | gist] through the Stage-1 model and read
    the final-layer hidden states at the k gist positions — a sentence's gist
    'thought' vector [B, k, hidden]. No continuation (encoding only); the gist
    attends to the span + causal-gist exactly as in training."""
    device = next(peft_model.parameters()).device
    k = gist_param.shape[0]
    span_lens = [len(s) for s in spans]
    max_s = max(span_lens)
    embed = peft_model.get_input_embeddings()
    span_e = embed(_pad(spans, max_s, pad_id, device))
    gist_e = gist_param.to(span_e.dtype).unsqueeze(0).expand(len(spans), -1, -1)
    inputs_embeds = torch.cat([span_e, gist_e], dim=1)  # [B, max_s+k, hidden]

    mask = build_batch_mask(span_lens, [0] * len(spans), k, max_s, 0, dtype=inputs_embeds.dtype).to(
        device
    )
    pos = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0).expand(len(spans), -1)
    out = peft_model(
        inputs_embeds=inputs_embeds,
        attention_mask=mask,
        position_ids=pos,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[-1][:, max_s : max_s + k, :]  # [B, k, hidden]


@torch.no_grad()
def gist_kv(
    peft_model,  # noqa: ANN001
    gist_param: torch.nn.Parameter,
    span: list[int],
    pad_id: int = 0,
):
    """The Stage-3 decode substrate: the FULL per-layer K/V at the k gist
    positions — what the continuation actually attends to during training.

    encode_gist() returns only the final-layer readout (one vector per slot);
    that is what the Stage-2 predictor is trained on, but it is NOT enough to
    decode from — the continuation reads the gist at EVERY layer's K/V. This
    runs [span | gist] with use_cache and slices the k gist positions out of
    each layer's cache. Returns (AxiomKV, cont_start, first_logits):
    - cont_start = span_len + k. The gist keys carry RoPE rotations at their
      TRAINING positions [span_len, span_len+k) — the continuation must decode
      from explicit position cont_start onward, or the relative geometry is
      off by span_len (cache-length position inference would put the query
      BEFORE the keys — Fable 3a-i review, blocking bug).
    - first_logits = the last gist position's next-token logits — in training
      the FIRST continuation token is predicted from exactly this position
      (next-token shift), so decode starts from argmax(first_logits): no
      out-of-distribution prime token needed.
    (Predicting a decodable thought still needs a final-layer -> per-layer-KV
    bridge, or re-targeting the predictor onto this object — the Stage-3 fork.)"""
    from marker.run_axiom_mlp_demo import AxiomKV  # noqa: PLC0415

    device = next(peft_model.parameters()).device
    k = gist_param.shape[0]
    max_s = len(span)
    embed = peft_model.get_input_embeddings()
    span_e = embed(torch.tensor([span], device=device))
    gist_e = gist_param.to(span_e.dtype).unsqueeze(0)
    inputs_embeds = torch.cat([span_e, gist_e], dim=1)  # [1, max_s+k, hidden]
    mask = build_batch_mask([max_s], [0], k, max_s, 0, dtype=inputs_embeds.dtype).to(device)
    pos = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0)
    out = peft_model(
        inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=pos, use_cache=True
    )
    cache = out.past_key_values
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    # slice the k gist positions [max_s : max_s+k] along the sequence dim (2)
    keys = [layer_kv[0][:, :, max_s : max_s + k, :].detach() for layer_kv in legacy]
    values = [layer_kv[1][:, :, max_s : max_s + k, :].detach() for layer_kv in legacy]
    kv = AxiomKV(n_layers=len(keys), keys=keys, values=values)
    return kv, max_s + k, out.logits[0, -1].detach()


@torch.no_grad()
def decode_from_gist_kv(
    peft_model,  # noqa: ANN001
    gist_kv_obj,  # noqa: ANN001
    cont_start: int,
    first_logits: torch.Tensor,
    *,
    max_new: int = 32,
    eos_id: int | None = None,
    stop_ids: set[int] | None = None,
):
    """Stage-3 ceiling: greedily decode from ONLY an injected per-layer gist KV
    — the span is gone; the thought alone must carry what-comes-next.

    Faithful to training geometry (logit-parity tested): the gist keys are
    RoPE'd at [span_len, span_len+k), so generated tokens take EXPLICIT
    positions cont_start, cont_start+1, ... (never cache-length inference),
    and the first token comes from argmax(first_logits) — the last gist
    position's prediction, exactly as in training. stop_ids (e.g. newline for
    step-per-line corpora) halt generation inclusively."""
    from marker.run_axiom_mlp_demo import _build_dynamic_cache  # noqa: PLC0415

    device = next(peft_model.parameters()).device
    halt = set(stop_ids or ())
    if eos_id is not None:
        halt.add(eos_id)
    past = _build_dynamic_cache(gist_kv_obj, device)
    nxt = int(first_logits.argmax().item())
    gen = [nxt]
    for j in range(max_new - 1):
        if nxt in halt:
            break
        out = peft_model(
            torch.tensor([[nxt]], device=device),
            past_key_values=past,
            position_ids=torch.tensor([[cont_start + j]], device=device),
            use_cache=True,
        )
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        gen.append(nxt)
    return gen


@torch.no_grad()
def generate_from_gist(
    peft_model,  # noqa: ANN001
    gist_param: torch.nn.Parameter,
    span: list[int],
    *,
    cont_sees: frozenset[str] = frozenset({"gist"}),
    max_new: int = 40,
    pad_id: int = 0,
    eos_id: int | None = None,
) -> list[int]:
    """Greedily decode a continuation for a single span, where generated tokens
    attend per `cont_sees` (gist-only vs gist+span). Re-prefills each step
    (eval-only, small). Returns the generated token ids. Compare gist-only vs
    full decode for the topical-agreement sanity check + the ilp_for probe."""
    device = next(peft_model.parameters()).device
    k = gist_param.shape[0]
    embed = peft_model.get_input_embeddings()
    gen: list[int] = []
    for _ in range(max_new):
        cont = gen if gen else [pad_id]  # need >=1 cont slot to read a logit
        span_e = embed(torch.tensor([span], device=device))
        cont_e = embed(torch.tensor([cont], device=device))
        gist_e = gist_param.to(span_e.dtype).unsqueeze(0).expand(1, -1, -1)
        inp = torch.cat([span_e, gist_e, cont_e], dim=1)
        mask = build_batch_mask(
            [len(span)], [len(cont)], k, len(span), len(cont), cont_sees=cont_sees, dtype=inp.dtype
        ).to(device)
        pos = torch.arange(inp.shape[1], device=device).unsqueeze(0)
        logits = peft_model(inputs_embeds=inp, attention_mask=mask, position_ids=pos).logits
        nxt = int(logits[0, -1].argmax().item())
        gen.append(nxt)
        if nxt == eos_id:
            break
    return gen


def gap_closed(ppls: dict[str, float]) -> float:
    """Fraction of the none->full PPL gap that gist closes. 1.0 = gist matches
    full context; 0.0 = gist no better than nothing; <0 = worse than nothing."""
    denom = ppls["none"] - ppls["full"]
    if denom <= 0:
        return 0.0
    return (ppls["none"] - ppls["gist"]) / denom
