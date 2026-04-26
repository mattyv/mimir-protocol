# Marker-extraction + contrastive injection — first WISE-style positive

**Date:** 2026-04-26 (same session as the LoRA work, late-evening pivot
back to activation methods).
**Verdict:** First real positive in the activation-injection track.
Concept-selective binding works on Qwen 2.5 1.5B at layer 20 with
marker-anchored extraction and contrastive (self − other) keys.

This is the result the GPT-2 POC failed to produce. It's small in
magnitude but unambiguous in shape.

## The mechanism

**Extraction (offline, per axiom):**
1. Take 30 paraphrases that mention the axiom term
2. Wrap each occurrence of the term in `[[...]]` markers
3. Forward-pass through frozen Qwen 2.5 1.5B
4. Capture the residual at the **last `]]` token position** at layer 20
5. Average across paraphrases, L2-normalise → `k_concept`
6. Compute contrastive: `k_concept_contr = normalise(k_concept − k_other)`
   (or mean of others if N > 2)

**Inference:**
1. User asks question; Mimir detects axiom name → retrieves
   `k_concept_contr`
2. Wrap the term in `[[...]]` in the question prompt
3. Forward-pass, with a hook at layer 20 that adds
   `α · k_concept_contr` to the residual at the closing marker position
4. Read the model's next-token distribution; selective shifts toward
   axiom-aligned tokens

No training. No new vocab tokens. No fine-tuning. Just frozen base
model + marker-anchored capture + a vector add at the right position.

## Diagnostics that mattered

### `cos(k, k_neg)` at the closing marker

| Layer | Eiffel | JOTP |
|---|---|---|
| 10 | 0.60 | 0.58 |
| 14 | 0.61 | 0.58 |
| 18 | 0.64 | 0.64 |
| 22 | 0.70 | 0.74 |

Compare to GPT-2 baseline of **+0.97 across all variants tested**. The
extracted direction at marker positions in Qwen 1.5B is
qualitatively different from random prose.

### `cos(k_jotp, k_eiffel)` per layer

| Layer | cos | concept-specific % |
|---|---|---|
| 4 | 0.929 | 37% |
| 8 | 0.889 | 46% |
| 12 | 0.890 | 46% |
| 16 | 0.904 | 43% |
| **20** | **0.854** | **52%** ← peak |
| 24 | 0.879 | 48% |
| 27 | 0.940 | 34% |

The two concepts share a large `axiom-anchored term in prose` direction
(0.85+ across most layers), with **46–52% of each key's magnitude in
concept-specific directions**. Layer 20 has the cleanest separation.

The shared direction explains why direct injection of `k_concept` did
nothing — the concept-specific signal exists but is dominated by
shared structure. The fix: **subtract the shared direction by using
contrastive vectors** (`k_jotp − k_eiffel`).

## Selectivity result

T1 prompts with markers wrapped on the inference side:

- JOTP: `[[JOTP]] is a technique used to`
- Eiffel: `The [[Eiffel Tower]] is located in`

Aligned / distractor target sets per the GPT-2 POC convention.
Selectivity gap = mean(aligned log-prob shift) − mean(distractor log-prob shift).

### Eiffel (clear positive)

| α | self_gap | cross_gap | rand_gap |
|---|---|---|---|
| 5 | **+0.023** | **−0.029** | −0.008 |
| 10 | **+0.055** | **−0.057** | +0.005 |
| 20 | **+0.102** | **−0.109** | +0.018 |

**Self positive, cross negative, mirror-symmetric. Random null.**
This is the textbook signature of concept-specific binding.

### JOTP (smaller but same shape)

| α | self_gap | cross_gap | rand_gap |
|---|---|---|---|
| 10 | +0.006 | −0.014 | −0.002 |
| 20 | +0.011 | −0.025 | −0.002 |

Smaller magnitude because JOTP-aligned targets (`appear`, `look`,
`fake`) are higher-baseline-probability than Eiffel-aligned targets
(specific geographies). Less room to shift. But the *sign pattern* is
identical: self positive, cross negative, random null.

## What this rules in

- **Marker-anchored extraction extracts genuinely concept-specific
  information** at mid-late layers of an instruction-tuned 1.5B model.
- **Contrastive isolation works.** Subtracting one concept's key from
  another isolates the perpendicular component, which is the
  concept-specific axis.
- **The injection-at-marker mechanism delivers the signal** through
  the rest of the forward pass to the prediction position.
- **The architectural bet is recoverable.** GPT-2's failure was a
  combination of (a) too-small model, (b) wrong capture position
  (last-token, output-aligned), (c) no contrastive isolation.
  Addressing all three together gives a working primitive.

## What's still to test

- **N-axiom version.** "Self minus other" only works pairwise. For a
  real Mimir register with many axioms, need to subtract the *mean of
  all other axioms*, or learn a shared-baseline vector. Test: does the
  selectivity pattern hold when N=3 or N=10?
- **Bigger magnitude.** ~0.1 nats max is enough for selectivity gates,
  not for rewriting model output substantially. Larger model (Gemma 4
  31B) should produce both larger gaps and stronger separations.
- **Composition (multi-axiom).** Can we inject `k_axiom_A + k_axiom_B`
  and get joint reasoning? This is the WISE compositional test from
  the original Mimir-Axiom spec.
- **Selectivity under distractors.** T4-style: a sentence with
  marker-anchored term plus contradictory ambient context. Does the
  marker-injected vector dominate?
- **Component decomposition.** Each axiom decomposes into typed
  components in Mimir. Can we extract per-component vectors and
  compose? This is the original Mimir-Axiom architecture.

## What this means for Mimir

The pipeline becomes:

```
Mimir (axiom register)
  ↓ get_axiom_for(term)
Vector key bank (key_bank: dict[axiom_id, np.ndarray])
  ↓ retrieve k for each detected axiom
Inference runtime
  ↓ wrap term in [[...]] in user prompt
  ↓ forward pass with hook at layer 20
  ↓ inject α · k at closing marker position
Model produces output biased toward axiom content
```

Pre-extraction is per-axiom and one-time. Inference is one extra
matrix-add per detected axiom. Adding the 1001st axiom to Mimir = one
more entry in the key bank + Mimir's symbolic register update.

This is the architecture the Mimir-Axiom design rationale described
and the original POC was supposed to test. We now have a primitive
that works, on a model small enough to iterate on locally.

## Repo state

- `src/marker/markers.py` — wrap-with-markers, find-marker-position
- `src/marker/run_extraction.py` — single-concept extraction + cos diagnostic
- `src/marker/run_contrastive.py` — pairwise contrastive diagnostic across layers
- `src/marker/run_injection.py` — full extract + inject + selectivity test
- `artifacts/marker_extraction_jotp.json`, `marker_extraction_eiffel.json`
- `artifacts/marker_contrastive.json`
- `artifacts/marker_injection_layer10.json` (the negative result before contrastive)
- 99/99 tests pass, ruff clean
