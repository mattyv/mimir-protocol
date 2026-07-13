# Fast lane (latent reasoning) — design, 2026-07-13

The unproven pillar: reason in thought-space for SPEED, skipping the big
model's expensive per-token generation on PREDICTABLE steps. Design-first —
the naive versions already failed (3b-i single-thought draft-verify, 3b chain);
no code until the cheap-first probe below clears.

## What's proven vs open

- Encoder makes good thoughts (gap_closed 0.88). ✓
- Render turns a thought back into faithful text (F1 0.87-0.93). ✓
- Predictor predicts the next thought ~2x within-doc chance; top-5 ~0.89 but
  top-1 ~0.30 (B). ✓ real, modest.
- Wrong thoughts actively mislead (xdoc < none). ✓ the constraint.
- OPEN: can you CHAIN predicted thoughts without errors compounding?

## Why naive latent chaining fails

Predict g2 from g1 -> inject -> predict g3 from [g1,g2] -> ... At top-1 ~0.30,
most predicted thoughts are "wrong", and wrong thoughts mislead, so a multi-
step rollout drifts off the reasoning manifold within a few steps. Draft-and-
verify IN TEXT failed to rescue it (the verifier couldn't select good drafts).

## Reframe: don't require the predictor to be RIGHT, require it to know WHEN it is

Adaptive skipping (this is the speed thesis, correctly stated):
- Each step, the predictor proposes the next thought + a CONFIDENCE.
- High confidence -> accept the predicted thought (cheap latent step; skip the
  big model's generation for that step).
- Low confidence -> fall back to full generation (correct, expensive).
- Speed win ≈ (fraction skippable) x (per-step speedup). The predictor handles
  the easy/predictable steps; the big model handles the hard ones. top-5 0.89
  says the true thought is usually among the predictor's candidates — so a
  usable confidence signal plausibly exists.

## Confidence — the crux (candidates to test)

1. Retrieval sharpness: is the predicted thought a confident peak (high top-1
   vs top-2 margin) or diffuse?
2. Cheap one-token agreement: inject the predicted thought, does the frozen
   model's FIRST continuation token agree with proceeding?
3. Predictor ensemble/dropout variance.

## Cheap-first probe (NO bridge, NO rollout, ~$0.3) — the gate

CONFIDENCE CALIBRATION. Using the EXISTING trained predictor + real encoded
thoughts on held-out chains: for each step, record (a) the predictor's
confidence signal(s), (b) whether its top-1 prediction is correct
(within-doc). Measure: does confidence SEPARATE correct from wrong predictions
(AUC / correct-rate in the high-confidence bin)?
- PASS (high-confidence bin is reliably correct) => the gate is viable =>
  build the bridge + gated rollout. Pre-register a skip-rate/accuracy target.
- FAIL (confidence doesn't track correctness) => confidence-gating is dead;
  rethink (e.g. always-verify spec-decode-of-thoughts, or accept the render
  lane as the deliverable and shelve the fast lane).

This probe is decisive and cheap — it reuses the predictor we already trained,
needs no new training, and answers "is adaptive skipping even possible" before
any bridge/rollout spend.

## Build order (only past the probe)

1. Confidence-calibration probe (above). GATE.
2. Train the bridge (predicted final-layer thought -> injectable KV;
   bridge_injection_nll already built + tested). ~$2.
3. Gated latent rollout: predict -> (confident?) bridge+inject : full-gen ->
   repeat; render to text at the end. Measure skip-rate + final-answer
   accuracy vs the true chain, and wall-clock vs plain generation.

## RESULT (2026-07-13): probe FAILED — confidence does not track correctness

Ran the cheap-first probe on the trained `stage2_cot_openr1` predictor, held-out
OpenR1 chains (skip first 2000 docs = its train+eval range), window=4, ~$0.30.
Teacher-forced block: 2004 (thought→next-thought) pairs, base within-doc top-1
= 0.275. Pre-registered gate (top-20% skip-bin accuracy ≥ 0.60, lift ≥ 0.15 over
base, on a by-document confirm split) — **FAIL on every signal:**

| signal | AUC | top-10% acc (lift) | top-20% acc (lift) |
|---|---|---|---|
| prediction_norm | 0.558 | 0.44 (+0.165) | 0.357 (+0.082) |
| dropout_agreement | 0.500 | 0.345 (+0.07) | 0.327 (+0.052) |
| retrieval_margin (diag) | 0.516 | 0.37 (+0.095) | 0.299 (+0.024) |

One-step-drift block agreed (all signals ~0.5 AUC; its base 0.56 is inflated by
small within-doc pools, so it's only a sanity echo, not the number).

Read: `prediction_norm` has a faint pulse — the most-confident 10% of
predictions are right 44% vs 27.5% base — but that's nowhere near a usable gate
(you'd still be wrong on 56% of the steps you skipped, and wrong thoughts
actively mislead). `dropout_agreement`, the signal we'd have bet on, is dead flat
at AUC 0.500 — the predictor's uncertainty is **not calibrated**. Absolute
prediction quality is moderate (mean slot-cosine to truth 0.665), but the model
cannot tell its good predictions from its bad ones.

**Verdict per the gate: the fast lane (adaptive skipping for speed) is DEAD as
designed.** Do NOT build the bridge or the gated rollout — they were scaffolding
for a skip decision that has no reliable signal to stand on. The render lane
(thought→faithful text, validated) remains the deliverable. The one unexplored
swing at reviving thought-*prediction* is a diffusion/sampling head that attacks
regression-to-the-mean directly (below) — a bigger build, not a gate.

## REVERSAL (2026-07-13, bridge ladder): predicted thoughts ARE usable injected

The confidence GATE stayed dead, but the "dig" (user call: don't stop at the
gate verdict) built the bridge anyway to ask the never-asked question: what does
a predicted thought DO when injected? Four runs (dtype crash -> undertrained ->
unstable optimizer -> overfit/hash-table probe) converged on a working recipe —
2000 docs, jittered training inputs (noise 0.5, anti-hashing/denoising),
val-gated checkpoint, width 512 — and the ladder came back (334 eval docs,
held-out, doc-disjoint):

| rung | eval gap_closed | read |
|---|---|---|
| gist_true | 0.804 | encoder ceiling (harness sanity — matches 0.88-era) |
| **bridge_true** | **0.817** | conversion is LOSSLESS — the k=8 final-layer summary IS a faithful handle; bridge(summary) even edges out raw gist KV (denoising training) |
| **bridge_pred** | **0.619** | a PREDICTED thought — no text ever existed — closes 62% of the gap when injected |
| shuffled | 0.284 | generic-math-context floor; ALSO net-positive through this bridge |
| none | 0.0 | |

Three upgrades to the world-model:
1. The final-layer summary is NOT lossy (kills the run-3 fear for good).
2. Predicted thoughts carry real step-specific signal end-to-end:
   0.62 sits far above the 0.28 generic-context floor.
3. The noise-trained bridge SOFTENED the misleading-injection constraint: even a
   wrong (cross-doc) thought is now net-positive (+0.28), where raw xdoc
   injection used to be worse-than-nothing. Denoising made injection robust —
   which weakens the case that skipping NEEDS a confidence gate at all.

Open next: (a) latent chain rollout — predict->bridge->inject->predict again,
measure drift over multiple latent steps (the real fast-lane test, now with a
substrate that works); (b) render a predicted thought to text and READ it;
(c) end-to-end task accuracy with always-inject (no gate).

## Non-goals / open

Diffusion thought-sampler (only if regression head plateaus — it has: top-1 0.30
+ flat confidence); 32B; the still-owed gate-3 Mimir confirmations (separate
track).
