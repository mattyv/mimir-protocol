# Implementation Plan 03 — Fix 3+ Prefix Chains via CacheBlend Selective Recompute

**Open problem (from `README.md`):** With three prose prefixes
concatenated, the RoPE-correction fix that solves 2-prefix sometimes
regresses. The model gets confused by "three independent documents
stacked back-to-back" because that's not a configuration the pretrained
model encountered. Path 2 (per-query joint encoding — re-tokenize all
relevant descriptions, run one prefill) is mechanically guaranteed to
work but costs a prefill per query.

**Chosen technique:** Yao et al 2024,
[CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/pdf/2405.16444).
At layer 1 of inference, identify the ~10–15% of cached prefix tokens
whose K vectors deviate most from a joint-context K, and selectively
recompute only those positions through all layers. Cheaper than full
joint encoding (Path 2), more robust than naive concat-with-RoPE-fix
(current 3-prefix regression).

**Why this technique** (full reasoning in
`docs/related-work-and-open-problems.md`): the literature reports up to
−35% accuracy from naive concat (KVLink paper), with selective
recompute closing most of that gap at ~15% the cost of full prefill.
Belt-and-braces — keep Path 2 as the always-correct fallback; ship
selective recompute as the production-default.

**Estimated effort:** 3–4 days. Hardest of the three plans because the
algorithm needs custom forward-pass plumbing through the model (not just
hooks). CacheBlend's reference impl is deeply integrated into vLLM and
not directly liftable; the algorithm is simple but the integration work
is real.

---

## Design decisions to make before coding

### 1. When to invoke selective recompute

| Option | Description | Verdict |
|---|---|---|
| **(R) Always for `n_prefixes >= 3`** | 2-prefix already works without; only kick in selective recompute for chains where naive concat regresses. | Surgical. Doesn't touch the validated 2-prefix path. |
| For all `n_prefixes >= 2` | Uniform code path. Simpler. | Risks regressing the validated 2-prefix case. Skip. |
| Optional, opt-in via flag | Default off; user enables explicitly. | Doesn't move us forward. Skip. |

### 2. Deviation metric

CacheBlend uses K-vector L2 distance between cached K and joint-context
K at layer 1. Two refinements possible:

| Option | Description | Verdict |
|---|---|---|
| **(R) L2 norm of (K_joint − K_cached) at layer 1, top 15%** | Matches CacheBlend exactly. Gives the safest baseline. | Start here. |
| Weighted by V-vector L2 too | Capture both K and V deviation. | Extra complexity, unproven gain. Defer. |
| Per-prefix (top 15% of *each* prefix) vs global (top 15% of all positions) | Per-prefix guarantees every prefix gets some recompute attention. Global may starve a stale prefix. | Try global first; switch to per-prefix if any prefix is consistently un-recomputed. |

### 3. Recompute scope

Once high-deviation positions are flagged, two ways to "recompute":

| Option | Description | Verdict |
|---|---|---|
| **(R) Re-prefill those positions through all layers with full cross-prefix attention** | Most faithful to CacheBlend. Each flagged position attends to all preceding positions in the joint cache, not just its own prefix. | This is the actual fix. |
| Just re-rotate K with corrected RoPE, leave V alone | Cheaper but doesn't solve the cross-attention problem. | Already covered by our existing RoPE fix. Skip. |

### 4. Integration with Path 2 (joint encoding)

| Option | Description | Verdict |
|---|---|---|
| **(R) Selective recompute as default; Path 2 as opt-in fallback flag** | Selective recompute is the new default for 3+ prefixes. If a query needs maximum fidelity (e.g. evaluation runs), opt into Path 2 via flag. | Production-friendly. |
| Path 2 default; selective recompute as a performance flag | Conservative; maintains "always correct" guarantee. | Higher per-query cost without strong evidence selective recompute regresses. |
| Auto-fallback: try selective; if output looks wrong (some heuristic), redo with Path 2 | Brittle; "looks wrong" is hard to detect. | Skip. |

---

## File-by-file implementation plan

### New file: `src/marker/selective_recompute.py` (~250 lines)

This is the bulk of the new code. Encapsulates the layer-1 deviation
analysis and the selective re-prefill.

```python
def find_high_deviation_positions(
    model,
    cached_k_layer1: torch.Tensor,    # shape (1, n_kv_heads, total_prefix_len, head_dim)
    joint_input_ids: torch.Tensor,    # full concatenated prefix tokens
    top_k_pct: float = 0.15,
) -> torch.Tensor:
    """
    Run a single forward pass at layer 1 using joint_input_ids with no
    cache (treat as fresh prefill). Compute K at layer 1.
    L2-diff against cached_k_layer1. Return indices of top top_k_pct
    most-deviant positions.
    """

def selective_recompute_prefix_cache(
    model,
    base_cache: DynamicCache,         # naive-concat cache with RoPE fix
    joint_input_ids: torch.Tensor,    # full concatenated prefix tokens
    high_deviation_positions: torch.Tensor,
    target_layers: list[int],
) -> DynamicCache:
    """
    For every position p in high_deviation_positions, recompute its K/V
    through all layers using full cross-prefix attention (i.e. attending
    to all preceding positions in joint_input_ids).
    Replace those positions in base_cache; leave other positions alone.

    Implementation: a custom forward loop that runs the model layer-by-
    layer, where each layer's attention reads from the PARTIALLY-PATCHED
    cache for queries from `high_deviation_positions` and writes back
    new K/V at those positions only. Other positions' K/V stay frozen.
    """

def blend_prefixes(
    model,
    prefixes: list[Prefix],
    rope_corrected: bool = True,      # use existing RoPE fix as base
    selective_recompute: bool = True, # NEW
    top_k_pct: float = 0.15,
) -> DynamicCache:
    """
    Top-level entry. Caller passes a list of Prefixes; gets back a
    populated DynamicCache. Internally:
      1. Build naive concat with RoPE correction (existing code)
      2. If selective_recompute and len(prefixes) >= 3:
         a. find_high_deviation_positions(...)
         b. selective_recompute_prefix_cache(...)
      3. Return final cache
    """
```

**Hard implementation note:** the layer-by-layer custom forward loop is
the hard part. HF transformers exposes layers via
`model.model.layers[i]` (or equivalent for Gemma/Mistral). For each
layer, you need to:

1. Take the residual-stream input at that layer for positions in
   `high_deviation_positions`
2. Project to Q/K/V (using the layer's attention module weights)
3. Apply RoPE to Q and K at the correct absolute positions
4. Compute attention over the FULL joint cache (all preceding positions
   regardless of high-deviation status)
5. Write the new K/V back into the cache at the high-deviation positions
6. Continue to MLP, then to the next layer

This is essentially a custom attention forward path that operates on a
sparse set of query positions while reading from a dense cache. The
closest reference impl is in
`related_work/CacheBlend/vllm_blend/vllm/` but it's vLLM-specific.

**Reference for the algorithm at the math level:**
[`CacheBlend paper`](https://arxiv.org/pdf/2405.16444) section 3
("Selective Recomputation"). Read this before coding.

### Modify: `src/marker/prefix_tuning.py`

- `combined_cache()` (line 224) — currently builds the naive concat cache
  with RoPE correction. Add a `selective_recompute: bool = False` kwarg;
  when True and `len(prefixes) >= 3`, route through
  `blend_prefixes()` from the new module instead.
- `generate_with_prefixes()` (line 285) — accept and pass through the
  `selective_recompute` kwarg.

Keep the existing 2-prefix path unchanged; the new path only activates
for `len(prefixes) >= 3`.

### New file: `src/marker/run_chain_selective_recompute_demo.py` (~100 lines)

Mirror `run_chain_demo.py`. Run the dependency-chain test
(BalancePublisher → TradingRiskEngine → ?) but extended to 3+ axioms.
A/B/C/D conditions:

| Condition | Description |
|---|---|
| A | No prefix (baseline) |
| B | Naive concat (no RoPE fix) — historical regression baseline |
| C | Naive concat + RoPE fix (current 2-prefix winner; **expected to regress at 3 prefixes**) |
| D | Selective recompute on top of C — the new fix |
| E | Path 2 (per-query joint encoding) — the always-correct upper bound |

Track:
- Output correctness on dependency-chain prompts (the existing
  `run_chain_demo.py` rubric — does the model use facts from each
  axiom in the chain?)
- Wall-clock latency per query
- For D: which positions got recomputed (sanity-check the deviation
  metric is selecting reasonable positions, not all the start tokens
  or all the end tokens)

### Modify: `modal_blends.py`

Add a `run_chain_selective_recompute` entrypoint matching the existing
`run_blends_big` pattern (line 38):

```python
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_chain_selective_recompute(
    model_name: str = "Qwen/Qwen2.5-32B",
    n_prefixes: int = 3,
) -> str:
    from marker.run_chain_selective_recompute_demo import main
    return main(model_name=model_name, n_prefixes=n_prefixes)

@app.local_entrypoint()
def chain_selective(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefixes: int = 3,
) -> None:
    print(run_chain_selective_recompute.remote(
        model_name=model, n_prefixes=n_prefixes,
    ))
```

### New file: `tests/test_selective_recompute.py` (~150 lines)

Per `CLAUDE.md`: mechanical invariants, not numerical experiment
outcomes. Match `tests/test_soft_prompt.py` style.

1. **Deviation detection shape.** Given a synthetic
   `cached_k_layer1` of shape `(1, 8, 32, 64)` and a synthetic joint-K
   of the same shape, `find_high_deviation_positions(top_k_pct=0.25)`
   returns a tensor of 8 indices (25% of 32) all in `[0, 32)`.
2. **High-deviation positions are correct.** Construct cached K and
   joint K differing in known positions (e.g. positions 5, 10, 15);
   assert `find_high_deviation_positions(top_k_pct=0.10)` returns
   `{5, 10, 15}` (or contains them as the top 3 of 32).
3. **Recompute writes only flagged positions.** Save K/V at all
   positions before `selective_recompute_prefix_cache`; after, assert
   K/V values are unchanged at non-flagged positions (within fp tolerance)
   and changed at flagged positions.
4. **Cache shape preserved.** Output cache has same per-layer shapes as
   input cache.
5. **Sanity rail.** Tiny model + 3 prefixes + selective recompute →
   "What's the capital of France?" → "Paris".
6. **Selective recompute is no-op when nothing deviates.** If
   `cached_k_layer1 == joint_k_layer1`, output cache equals input
   cache exactly.

---

## Validation criteria

Run `run_chain_selective_recompute_demo.py` on Qwen 2.5-32B base:

| Condition | Description | Required |
|---|---|---|
| C (RoPE fix only) | Current 3-prefix regression | **Reproduce documented regression** (some prefix's facts are missing from the answer) |
| D (selective recompute) | New fix | **Match Path 2's correctness** within 1 axiom-fact difference |
| E (Path 2) | Joint encoding upper bound | Always-correct reference |

Wall-clock budget:
- D should be **≤ 1.5×** baseline (no-prefix) per-query latency
- E is typically 2-4× baseline depending on prefix count

If D matches E in correctness at 1.5× cost, ship. If D matches E only
sometimes, raise `top_k_pct` (more positions recomputed → closer to
Path 2 but slower) and re-evaluate.

---

## Risks / what could kill it

1. **The custom layer-by-layer forward path doesn't faithfully match
   the model's native forward.** A subtle off-by-one in RoPE position
   indices, a missing layernorm, a wrong residual-stream update — any
   of these silently corrupts the cache. Mitigation: write a strict
   test that runs the custom forward path with `high_deviation_positions
   = all positions` and asserts the output cache matches a vanilla
   `model(...)` forward pass within fp tolerance. If the no-op-on-all
   case doesn't match vanilla, the implementation is broken.
2. **Layer-1 K is the wrong signal.** CacheBlend uses layer 1 because
   it's cheap and predictive of downstream divergence. On Qwen 32B
   (64 layers), layer 1 may be too generic — early layers carry token
   identity, not composition state. Mitigation: try layer 1, layer 4,
   layer 8, layer N/4 and compare which best predicts which positions
   need recompute. Add a config flag.
3. **15% recompute ratio is wrong for our prefixes.** CacheBlend tunes
   on RAG documents; our prose-description prefixes are different.
   Mitigation: sweep `top_k_pct ∈ {0.05, 0.10, 0.15, 0.25, 0.50}` in
   the demo, plot correctness vs latency, pick the knee.
4. **Selective recompute regresses 2-prefix.** If the fix is gated on
   `n_prefixes >= 3`, this can't happen. If you decide to apply it
   uniformly (Decision #1 alternative), test the 2-prefix gauntlet
   regression explicitly before shipping.
5. **It still doesn't beat Path 2's correctness, just costs less.**
   That's fine — ship it as the default, keep Path 2 as the
   `--high-fidelity` flag.

---

## Order of execution

1. Read CacheBlend paper section 3 (Selective Recomputation) end-to-end.
   Sketch the algorithm in pseudocode (45 min).
2. Implement `find_high_deviation_positions()` (1 hr).
3. Write the deviation-detection tests; pass them (1 hr).
4. Implement `selective_recompute_prefix_cache()` against a 0.5B Qwen
   model. The custom forward loop is the hard part — use HF
   `Qwen2DecoderLayer.forward()` source as the template, swap in the
   sparse-query attention (1 day).
5. Write the "no-op-on-all" matching test against vanilla forward.
   This is the make-or-break test (2 hr).
6. Pass the recompute writes-only-flagged test (30 min).
7. Wire into `combined_cache()` and `generate_with_prefixes()` behind
   the gating flag (1 hr).
8. Sanity-rail test on tiny model with 3 prefixes (1 hr).
9. `run_chain_selective_recompute_demo.py` + Modal entrypoint (2 hr).
10. Sweep `top_k_pct` on the demo (~3 hr GPU).
11. Compare D vs E correctness on the chain prompts (~2 hr GPU).
12. Update `CONCLUSIONS.md` and `README.md` "What's still hard" section.

Total: 3–4 days assuming the layer-by-layer forward path doesn't have a
hidden show-stopper specific to Qwen's attention impl.

---

## Citations

- Yao et al, *CacheBlend: Fast LLM Serving for RAG with Cached
  Knowledge Fusion*, 2024.
  [arxiv:2405.16444](https://arxiv.org/pdf/2405.16444) ·
  [code](https://github.com/YaoJiayi/CacheBlend) (cloned to
  `related_work/CacheBlend/`)
- Yang et al, *KVLink: Accelerating LLMs via Efficient KV Cache
  Reuse*, NeurIPS 2025.
  [arxiv:2502.16002](https://arxiv.org/abs/2502.16002) — alternative
  approach (trainable cross-segment linker tokens). Read for context;
  not the primary technique for this plan.
- Hu et al, *EPIC: Efficient Position-Independent Caching for Serving
  LLMs*, ICML 2025. [arxiv:2410.15332](https://arxiv.org/abs/2410.15332)
  — same family; "LegoLink" is functionally equivalent to CacheBlend's
  selective recompute.
- See also: `docs/related-work-and-open-problems.md` for the broader
  survey and why selective recompute beat KVLink/EPIC for our use case
  (we don't want to train linker tokens or modify the model).
- See also: `README.md` "What's still hard" — the original 3+ prefix
  regression problem statement.
