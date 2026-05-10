# Implementation Plan 02 — Make Prefix Tuning Work on Gemma 4 (Sliding-Window Attention)

**Open problem (from `README.md`):** On Gemma 4-31B-IT (hybrid 5:1
local:global attention), prefix tuning produces **null effect across all
10 axioms** — outputs identical to baseline. Existing `THINGS_TO_TRY.md`
diagnoses likely cause: most layers use sliding-window attention; when
prefix sits at positions 0-31 and user prompt starts later, local layers
can't reach the prefix.

**Chosen technique:** combine two complementary fixes:

1. **Inject only at global-attention layers** (every 6th layer in Gemma 4
   per the architecture). Already on the queue in `THINGS_TO_TRY.md`.
2. **Pin prefix tokens as attention sinks** so even local-attention
   layers attend to them regardless of sliding-window distance. Adapted
   from Xiao et al 2024,
   [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453).

**Why both:** global-only injection alone leaves 5/6 of the model's
layers ignorant of the prefix. Sink-pinning alone keeps the prefix
visible to local layers, but the model still wasn't trained to attend to
"out-of-window" content there — it might ignore it. Together: prefix is
mechanically reachable everywhere *and* the layers most likely to use it
(globals) get the strongest injection.

**Why this technique** (full reasoning in
`docs/related-work-and-open-problems.md`): StreamingLLM proved that
pinning the first few tokens as always-attended sinks is enough to
stabilize long-context generation across architectures. We apply the same
mask-modification trick to mark our prefix range, not the input prefix.

**Estimated effort:** 2–3 days. Harder than Plan 01 because Gemma's
attention implementation is nonstandard and HF transformers' attention
mask plumbing varies by attention impl (eager / sdpa / flash-attn 2).

---

## Design decisions to make before coding

### 1. Which Gemma variant to validate against

| Option | Description | Verdict |
|---|---|---|
| **(R) Gemma 4-31B-IT (the original null-result target)** | Reproduces the exact failure case from `README.md`. | Closes the loop on the documented problem. |
| Gemma 4-31B base (non-IT) | Isolates sliding-window failure from RLHF refusal. Already in `THINGS_TO_TRY.md` as a suggested diagnostic. | Run as a control alongside IT. If IT works, base trivially works; if base works but IT doesn't, you've also got a refusal problem (see Plan 01). |
| Gemma 4-9B / smaller | Cheaper iteration. | Use as smoke-test, not final validation. |
| Mistral 7B (also has sliding window) | Different sliding-window impl; cross-check generality. | Stretch; not needed for first land. |

Run on Gemma 4-31B base **and** IT together — base tells you if the fix
works; IT tells you if Plan 01 (refusal-direction) is also needed.

### 2. Where to detect "global" layers

| Option | Description | Verdict |
|---|---|---|
| **(R) Read from `model.config`** | Gemma 4's `Gemma3TextConfig` exposes `sliding_window_pattern` (e.g. 6 = every 6th layer is global). Programmatic. | Robust across model sizes. |
| Hard-code `[5, 11, 17, ...]` for Gemma 4-31B | Fast and explicit. | Fragile across model sizes; redo for every variant. |
| Probe attention impl at runtime | Inspect each layer's attention module class. | Overkill. |

### 3. How to pin sinks

HF transformers' attention mask is constructed per-call. Two ways to
intervene:

| Option | Description | Verdict |
|---|---|---|
| **(R) Custom 4D attention mask passed to `model(...)`** | Build a mask of shape `(batch, 1, q_len, kv_len)` where positions in our prefix range are unmasked regardless of sliding-window distance. Pass via `attention_mask` kwarg (transformers 4.43+ accepts 4D masks). | Cleanest. No model surgery. Composes with our existing `past_key_values` injection. |
| Monkey-patch attention forward | Modify Gemma's attention to extend the local window for prefix positions. | More invasive; couples us to Gemma's attention impl version. Reserve as fallback if 4D mask path fails. |
| Use StreamingLLM's `enable_streaming_llm()` directly | They support llama / mpt / gpt_neox / falcon (see `related_work/streaming-llm/streaming_llm/enable_streaming_llm.py:4`). Gemma not supported. | Would need to extend their `pos_shift/` logic to a new `modify_gemma.py`. |

### 4. Whether to keep the top-half-layer rule

Currently we inject at top-half layers (32-63 of 64 on Qwen). On Gemma
4-31B with 6:1 sparsity, top-half = layers 30-59. Of those, only
~5 are global. Two options:

- **(R) Inject at all global layers (top half OR bottom half)** that
  fall in or above the existing top-half band's start. Keep the spirit
  of "no early generic-token layers" but allow more global layers in.
- Keep strict top-half. Effectively reduces injection to 5 layers.
  Likely too few. Try only if option above causes looping (the original
  reason for the top-half rule).

---

## File-by-file implementation plan

### New file: `src/marker/sliding_window_support.py` (~150 lines)

Two responsibilities: detect global layers, build the sink-mask.

```python
def detect_global_attention_layers(model) -> list[int]:
    """
    Returns indices of layers that use full (global) attention.
    For Gemma 4: reads model.config.sliding_window_pattern.
    For other architectures: returns range(num_hidden_layers) (no-op
    for dense models — used as a safety fallback).
    """

def build_sink_attention_mask(
    base_mask: torch.Tensor,
    prefix_range: tuple[int, int],
    sliding_window: int,
) -> torch.Tensor:
    """
    Take a 2D causal+padding mask, expand to 4D, then mark the prefix
    token range as always-visible across all positions regardless of
    the sliding-window cutoff.

    Args:
      base_mask: (batch, kv_len) padding mask
      prefix_range: (start, end) absolute positions of prefix tokens
      sliding_window: window size (read from model.config.sliding_window)

    Returns:
      mask of shape (batch, 1, q_len, kv_len), additive
      (0.0 = visible, -inf = masked).
    """

def select_injection_layers_for_sliding_window(
    model,
    user_target_layers: list[int],
) -> list[int]:
    """
    Intersect user_target_layers with detect_global_attention_layers().
    If empty, fall back to all global layers.
    """
```

**Reference for sink semantics:** the StreamingLLM code (cloned at
`related_work/streaming-llm/streaming_llm/kv_cache.py:23`) shows the
**eviction-policy** form of sinks (keep first N keys forever). We need
the **mask-policy** form: don't evict (we don't decode that long), but
unconditionally allow attention to the prefix range. The math is the
same — additive 0 for sink positions, additive -inf for out-of-window
non-sink positions.

### Modify: `src/marker/prefix_tuning.py`

Three changes:

1. **`Prefix.from_description()` (line 84):** when target model has
   sliding-window attention, intersect requested target_layers with
   global layers and emit a warning if the user's spec doesn't include
   any global layers.

2. **`combined_cache()` (line 224) and `to_cache()` (line 145):**
   no change needed — the cache structure is the same.

3. **`generate_with_prefix()` (line 473) and `generate_with_prefixes()`
   (line 285):** when the model's `model.config` has a `sliding_window`
   attribute, build a sink-aware 4D attention mask via
   `build_sink_attention_mask()` and pass via `attention_mask=` kwarg
   to `model(...)`. For dense-attention models, behavior is unchanged.

Detection guard:

```python
has_sliding_window = (
    hasattr(model.config, "sliding_window")
    and model.config.sliding_window is not None
)
```

### Modify: `src/marker/run_prefix_demo.py`

Add `--sink-attention` flag (default off for backward compat). When set,
plumb through to `generate_with_prefix(es)` so the existing 10-axiom
gauntlet runs with sink-aware masks on sliding-window models.

### New file: `src/marker/run_gemma_prefix_demo.py` (~120 lines)

Mirror `run_prefix_demo.py`'s structure but pre-configured for Gemma 4:
- Calls `select_injection_layers_for_sliding_window(model, [...])` to
  pick layers
- Always enables sink attention
- Compares **A: baseline / B: prefix injection naive (current null
  result) / C: prefix + sink-aware mask** on the 10-axiom gauntlet

The naive-B condition is essential — without it, you can't show the
fix changed anything from the documented null state.

### Modify: `modal_blends.py`

There's already a `run_gemma_probe` (line 148) and `gemma_probe` local
entrypoint (line 168). Add parallel:

```python
@app.function(
    image=image,
    gpu="H100",  # Gemma 4-31B benefits from H100 over A100
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_gemma_prefix(
    model_name: str = "google/gemma-3-27b-it",  # confirm exact id
    sink_attention: bool = True,
) -> str:
    from marker.run_gemma_prefix_demo import main
    return main(model_name=model_name, sink_attention=sink_attention)

@app.local_entrypoint()
def gemma_prefix(model: str = "google/gemma-3-27b-it") -> None:
    print(run_gemma_prefix.remote(model_name=model))
```

(Gemma 4 vs 3 model id depends on what's actually released by ship date —
check HF Hub before running. The existing `docs/deployment-gemma4-31b.md`
runbook may need updating.)

### New file: `tests/test_sliding_window_support.py` (~120 lines)

Per `CLAUDE.md`: mechanical invariants only.

1. **Layer detection.** On a dummy config with `sliding_window_pattern=6`
   and `num_hidden_layers=64`, `detect_global_attention_layers()`
   returns `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]` (every 6th).
2. **Mask correctness — sink positions visible.** Given prefix_range
   `(0, 32)` and a query at position 5000, the sink-mask sets positions
   0..31 to `0.0` (visible) and positions 32..(5000-1024) to `-inf`
   (masked by sliding window).
3. **Mask correctness — recent window preserved.** Same query at
   position 5000 with window 1024: positions 3976..4999 visible.
4. **Causal mask preserved.** Positions ≥ 5000 (future tokens) remain
   masked from query at position 5000.
5. **No-op on dense models.** For a model whose config has no
   `sliding_window`, `build_sink_attention_mask` returns the input mask
   unchanged (or is bypassed by the call-site detection).
6. **Sanity rail.** Tiny model + sink mask + prefix → "What's the
   capital of France?" → "Paris". (Requires a sliding-window tiny
   model — use Mistral-7B if available, else mock the config.)

### New file: `data/gemma_axioms_baseline.json` (auto-generated, optional)

Caches the documented null-result baselines for B (naive prefix on
Gemma) so the C-vs-B delta in the new demo is reproducible without
re-running B every time.

---

## Validation criteria

A/B/C eval in `run_gemma_prefix_demo.py`:

| Condition | Description | Required |
|---|---|---|
| A | Gemma 4-31B base, no prefix | Baseline reference |
| B | Gemma 4-31B base, naive prefix injection (current behavior) | **Must reproduce documented null result** (≈ 0/10 axioms produce axiom-specific facts) |
| C | Gemma 4-31B base, prefix + sink mask + global-layer-only injection | **Target ≥ 6/10 axioms** (matches Qwen 32B Instruct baseline) |

If C reaches ≥ 6/10 base but ≤ 6/10 IT, that confirms the IT failure is
the refusal problem from Plan 01 — composing both fixes should clear it.

If C is still ≤ 1/10 even on base, the diagnosis was wrong: it's not
sliding window. Likely candidates:
- Gemma uses GQA with grouped K/V heads in a way our prefix capture
  shape doesn't match (check `n_kv_heads` vs `n_heads` on Gemma)
- Position embeddings interaction (Gemma 4 uses RoPE per head_dim group)
- Top-half rule wrong for Gemma's depth distribution

---

## Risks / what could kill it

1. **Gemma's attention impl rejects 4D masks.** HF transformers added
   4D mask support in 4.43, but only for `attn_implementation="eager"`.
   If Gemma defaults to SDPA or flash-attn 2 and silently ignores the
   mask, the fix is invisible. Mitigation: force
   `attn_implementation="eager"` on the model load. Slow but correct.
2. **Sink positions still aren't useful.** Even if mechanically visible,
   the model wasn't trained to put information into out-of-window-but-
   visible positions. The injection might still produce no behavior
   change. Mitigation: try the alternative — re-render the prefix at
   inference time as part of the input prompt (prefix-as-text) on Gemma
   only, sacrificing the "no text in user prompt" claim for
   architectures where K/V splicing fundamentally doesn't fit.
3. **GQA shape mismatch.** Gemma 4 uses grouped-query attention. Our
   `Prefix.per_layer_shapes` (line 60 of `prefix_tuning.py`) records
   `(n_kv_heads, n_tokens, head_dim)` already. Verify the capture path
   handles `n_kv_heads != n_heads` correctly — it should, since HF's
   `past_key_values` already reflects GQA shapes, but worth a unit test
   on a GQA-only tiny model first.
4. **Top-half rule is wrong for Gemma.** Our top-half-layer empirical
   finding was on Qwen. Gemma's mid layers may carry the relevant
   composed state. Mitigation: add per-layer ablation in the demo
   (inject at one global layer at a time, find which layers actually
   move the output) before the full A/B/C.

---

## Order of execution

1. Confirm Gemma 4 model id on HF Hub. Update
   `docs/deployment-gemma4-31b.md` if needed.
2. Write the layer-detection test against a fake config. Pass it (15 min).
3. Write the mask-correctness tests against a synthetic input. Pass them
   (1 hr).
4. Implement `build_sink_attention_mask()` and
   `detect_global_attention_layers()` (1 hr).
5. Sanity-check: load Gemma 4-9B locally if possible, build a mask,
   inspect via `print(mask[0, 0, -1, :].tolist())` — eyeball that prefix
   positions show 0.0 and out-of-window non-sink positions show -inf
   (30 min).
6. Wire `attention_mask` kwarg through `generate_with_prefix(es)`
   (1 hr).
7. Smoke test on a tiny sliding-window model (Mistral-7B locally if
   memory permits) (1 hr).
8. Modal entrypoint + `run_gemma_prefix_demo.py` (2 hr).
9. Full Modal run on Gemma 4-31B base + IT, A/B/C (~2 hr GPU each).
10. Per-layer ablation if C ≤ 1/10 (~half day debug).
11. Update `CONCLUSIONS.md` and `THINGS_TO_TRY.md` (move out of TODO).

Total: 2–3 days assuming Gemma is available and the H100 is provisioned.

---

## Citations

- Xiao et al, *Efficient Streaming Language Models with Attention
  Sinks*, ICLR 2024.
  [arxiv:2309.17453](https://arxiv.org/abs/2309.17453) ·
  [code](https://github.com/mit-han-lab/streaming-llm) (cloned to
  `related_work/streaming-llm/`)
- See also `THINGS_TO_TRY.md` "Prefix tuning on Gemma 4 (sliding-window
  attention)" — original problem statement and parked diagnostics.
- See also `docs/deployment-gemma4-31b.md` — Gemma 4 31B deployment
  runbook.
- See also `docs/related-work-and-open-problems.md` for the broader
  survey.
