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

## Non-goals / open

Diffusion thought-sampler (only if regression head plateaus); 32B; the
still-owed gate-3 Mimir confirmations (separate track).
