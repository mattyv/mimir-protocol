# POC plan: true per-axiom prefix tuning (trained virtual KV tokens)

## Hypothesis

N trained virtual K/V tokens per layer — learned **from scratch** against Q+A
loss, never derived from text — can carry one axiom's facts, matching
FACTS-text-prefill accuracy at ~10x fewer cache positions.

This is the untested variant: CONCLUSIONS.md showed *text-init + gradient*
drifts, and init-only text KV works. From-scratch trained tokens are the gap.

## Pre-registered success / kill criteria

- **PASS**: some N ≤ 16 where PREFIX heldout-phrasing accuracy ≥ FACTS accuracy
  minus one question, on most axioms. Deliverable: accuracy-vs-N capacity curve.
- **KILL**: N=16 still well below FACTS on heldout phrasings, or
  trained-phrasing ≈ perfect while heldout ≈ ZERO (pure QA-string memorization).
  Record in FAILED_IDEAS.md and stop this line.
- **Watch**: gibberish/looping (off-manifold KV) — print sample outputs, not
  just scores.

## Design

### Parameterization
Per axiom: learnable tensors `K, V` of shape
`(n_layers, n_kv_heads, N, head_dim)`, stored fp32, cast to model dtype at
injection (cast, NOT detach — gradients must flow through the cache).
No MLP, no hooks, no term-position logic. All layers.

The prefix occupies cache slots `0..N-1`. Learned K values are post-RoPE by
construction (they're free parameters), so no RoPE correction for
single-axiom use. Multi-axiom composition is explicitly out of scope.

### Init (two arms, one flag)
1. `--init random` (primary): per-layer scaled random — compute the axiom
   description's real KV once, match its per-layer/per-head mean+std. Wrong
   scale breaks attention; this must be stat-matched, not unit normal.
2. `--init subsample` (secondary): N token positions sampled from the
   description KV, then trained.

### Training
- Reuse the backprop-through-`_build_dynamic_cache` path proven in
  `kv_compression.train_compressor` (build a fresh DynamicCache from the param
  tensors each step; the cache is stateful).
- Loss: CE on answer tokens only (labels -100 elsewhere). Template `Q: {q}\nA:`.
- AdamW, lr 5e-3 cosine → 5e-4, grad-clip 1.0, n_steps 800 per (axiom, N).
- Sample uniformly from the axiom's expanded train_qa.

### Data — the part that decides whether results mean anything
- Base: `ABLATION_AXIOMS` (8 axioms, 6 domains) from `run_ablation_demo.py`.
- **Expand train_qa to 6–10 paraphrase pairs per axiom** (currently 2 — a
  prefix would just memorize two strings). Hand-written, distinct phrasings.
- Eval stays the existing gold-substring `eval` sets (unseen phrasings).
- **Leakage guard**: assert no normalized train question equals an eval
  question (reuse the normalizer from `tests/test_heldout_leakage.py`).

### Conditions (auto-scored, per axiom, trained vs heldout phrasing split)
| condition | what | cache positions |
|---|---|---|
| ZERO | no injection | 0 (floor) |
| FACTS | text prefill of fact_text | ~35 (champion to match) |
| PREFIX-N, N ∈ {2,4,8,16} | trained virtual tokens | N |

8 axioms × 4 N values × 800 steps ≈ 25–35 min on A100 (~$0.30).

### Output
Per-axiom sample outputs for every condition + a final summary table:
rows = condition, cols = TRAINED-phrasing acc, HELDOUT-phrasing acc, positions.
Print the capacity curve (N vs heldout acc) explicitly.

## Files
- `src/marker/prefix_poc.py` — `AxiomPrefix` (params + save/load),
  `train_prefix()`, `build_prefix_cache()`, stat-matched init helpers.
- `src/marker/run_prefix_poc.py` — runner: axiom data (expanded train sets
  live here; import eval/fact_text from `run_ablation_demo`), N sweep,
  scoring, summary. Args: `--model-name --n-steps --lr --n-list --init
  --max-new --smoke`.
- `tests/test_prefix_poc.py` — mechanical: param shapes; cache build keeps
  grad_fn (no silent detach); stat-matched init matches target std within
  tolerance; save/load round-trip; train/eval leakage guard on the POC data.
- `--smoke` mode: Qwen2.5-0.5B, 1 axiom, N=2, 10 steps — **must run clean
  locally before any Vast launch** (a dtype bug and a data bug each cost a
  paid run in earlier iterations; the smoke gate is non-negotiable).

## Vast procedure (unchanged from previous runs)
onstart clones `claude/project-review-6rx97z`, pip installs
`transformers>=4.45,<5 accelerate sentencepiece`, runs the module with
`python -u`, echoes `=== RUN COMPLETE rc=$? ===`; poll logs over HTTPS;
destroy node immediately after log capture.

## Known risks
- Backprop through DynamicCache may hit a transformers version that detaches
  internally — the smoke test catches this (loss.backward() with no grad on
  params = fail fast).
- lr 5e-3 may be hot for V at some layers → watch for NaN; clip + cosine
  usually suffices; `--lr` is a flag for a reason.
- Small train sets risk memorization even at 6–10 pairs — that's precisely
  what the trained/heldout phrasing split measures; don't "fix" it by adding
  eval phrasings to train.
