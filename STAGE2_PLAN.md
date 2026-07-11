# Plan: Stage-2 next-thought predictor (gated build)

Authored 2026-07-11 after Stage-1 gist PASS (gap_closed 0.887, xdoc control
-0.83 — see LATENT_PLAN.md). Fable gate-review steers baked in. Build is
model-free/CPU until the corpus encode; SPEND (encode + train) is the gate.

## What Stage 1 settled that shapes this (record before it's forgotten)

- **The gist carries span-specific content, not just slot-presence.** xdoc
  (gist from an UNRELATED doc) = -0.83: a wrong gist is WORSE than none.
- **Stage-3 design consequence — wrong thoughts actively mislead.** Because
  xdoc < none, the frozen decoder cannot ignore a bad injected gist. So when
  the predictor is wrong, the failure is MISLEADING context, not missing
  context. The snap/verify machinery (and Stage-3b draft-verify) must assume
  a bad predicted gist degrades output below no-injection — verification is
  not optional politeness, it's load-bearing.
- **Decomposition:** neighbor(same-doc)=0.632 vs gist=0.887 — ~2/3 of a
  sentence's predictive value is document-topic, the rest span-specific. Both
  live in the gist. Quote both together; don't over-claim the headline.

## Architecture (Fable steer #1: 8 slots, NOT a pooled vector)

Stage 3 injects 8 KV slots and cannot unpool a single vector, so the
predictor's OUTPUT must be 8 slot vectors.
- **Input:** per sentence, its 8 gist slots as 8 tokens + a sentence-position
  encoding (which sentence in the doc) + a slot-index encoding (which of 8).
- **Trunk:** small transformer (4-8 layers, d_model 512-768, ~30-80M params),
  causal over the flattened (sentence x slot) sequence.
- **Head (phase A):** read out the next sentence's 8 slot vectors jointly
  (regression). InfoNCE on a pooled projection for the contrastive term.
- **Head (phase B, later):** diffusion/flow sampler over the 8-slot target —
  ONLY after phase A shows retrieval signal. Never debug trunk+diffusion
  together (Fable steer #2).
- **NO unpooling expander** — a pooled-vector predictor + expander is a second
  lossy stage nobody asked for.

## Whitening (Fable steer #3 + the streaming to-do)

- **Per-slot-index whiteners (8 of them):** slot 1 and slot 8 are
  distributionally different by construction. Default to 8 ZCA whiteners.
- **Streaming fit:** at real scale (>=300k gists x slot-d) a double-precision
  materialization is tens of GB. Accumulate running mean + outer-product sums
  in chunks; never hold the full matrix. (whiten.py fit() is the in-memory
  version for tests; add fit_streaming for the corpus.)
- **Fit on TRAIN gists only, never heldout** (same leak discipline as Stage 1).

## Loss (spec §3.2 killers)

- Regression: 1 - cos(pred_slots, true_slots), averaged over 8 slots, whitened.
- Contrastive: InfoNCE, in-batch negatives on pooled projections — the
  regression-to-the-mean/platitude guard. L = lam_reg*reg + lam_nce*nce,
  start lam_reg=0.1, lam_nce=1.0.
- **Hard negatives for free (Fable steer #4):** doc-clustered batches make
  same-doc gists genuinely-hard decoys (neighbor=0.632 proves it). Keep some
  cross-doc shuffling for easy negatives too.

## Data flow (Fable steer #5: one node, one run)

Encode -> fit whitener -> train, all sequential on ONE node. 300k slot-level
gists ~= 15+ GB — do NOT push through HF. Push only artifacts (predictor +
8 whiteners + manifest). ~$2-3 total.
1. Load frozen 7B + the Stage-1 gist adapter (from mattyvee/mimir-artifacts).
2. Stream corpus; per document, gist every sentence -> [n_sents, 8, d] gists.
3. Streaming-fit the 8 whiteners on train gists.
4. Train the predictor on gist sequences (next-8-slot prediction).
5. Eval + push artifacts.

## Pre-registered gates (Fable steer #6 — WRITE BEFORE LAUNCH, project law)

Predictor must beat these on held-out (document-disjoint) gist sequences:
- **Retrieval:** in-batch recall@1 > 0.15 and recall@5 > 0.40 (batch>=128).
  A predictor that ranks the true next-gist top-5 out of 128 decoys 40% of
  the time is finding real structure (random = 5/128 ~= 4%).
- **Platitude guard:** mean pairwise cosine of predictions across DIFFERENT
  contexts must be < corpus mean pairwise similarity (i.e. predictions are
  context-varying, not collapsed to one vector). If predictions are more
  self-similar than the corpus, it collapsed — KILL and rethink the loss.
- **KILL:** recall@5 <= random (~4%) after the token budget ⇒ the predictor
  learned nothing; the raw-text signal may be too weak (⇒ the parked CoT
  distillation variant).

## Build order (model-free first, no spend)

1. whiten.py — DONE (in-memory fit + tests). Add fit_streaming + per-slot.
2. predictor.py — the transformer trunk + regression/InfoNCE + recall@k /
   diversity metrics. All CPU-testable (tiny dims, overfit a toy sequence).
3. run_stage2.py — encode/fit/train/eval runner (smoke on tiny model+corpus).
4. Launcher. Then: write final gate numbers here, launch.

## Non-goals (defer)

Diffusion head (phase B); the CoT-distillation data variant (parked in
LATENT_PLAN, only if raw-text predictor hits the KILL gate); 32B; Stage-3
runtime.
