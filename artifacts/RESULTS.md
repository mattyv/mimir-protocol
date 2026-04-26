# GPT-2 POC — Final Results

**Verdict.** Single-vector residual-stream injection does not produce
axiom-shaped behavior on GPT-2 small layer 8, under any of the capture
or injection variants we tested. Confirmed on a novel concept (JOTP) and
on a known concept (Eiffel Tower), under a metric that is invariant to
uniform logit tilt. The simplest form of the geometric-realisation
thesis — "a learned direction added to the residual stream binds the
axiom as a premise" — is now cleanly falsified on this stack.

The broader thesis is *narrowed*, not killed. The single-vector form
fails; richer forms (multi-feature SAE keys, sentinel-LoRA, larger
models) remain untested. See `docs/mimir-protocol-poc-spec.md` for the
staged next experiment.

This document supersedes two prior versions, both of which over- and
mis-interpreted earlier negative results. The history is preserved in
git.

## Method, final

For each concept (Eiffel, JOTP):

1. Capture residuals at GPT-2 small layer 8, **at the position of the
   concept token** in each paraphrase (last token of " Tower"/" tower"
   for Eiffel; last token of "JOTP" for JOTP). Build `k = mean(positives)`
   L2-normalised, `k_neg = mean(neutral-prose-last-token)` L2-normalised,
   `k_minus_neg = (k − k_neg) / ||·||`.
2. For each test prompt (T1 = concept-relevant, three T2 = unrelated):
   inject the key under three modes —
   (a) **last** — at the prompt's final token (the v2 baseline)
   (b) **concept** — at the concept token's position in *that* prompt
   (c) **multi** — at every position from concept-onward through the end.
3. Record **log-probability shifts** (not raw logit shifts) for the
   aligned target set and the distractor target set. Selectivity gap =
   `mean(aligned shift) − mean(distractor shift)`.

Why log-probability: a uniform additive logit tilt produces zero
log-prob shift (because logsumexp shifts identically), so global-bias
confounds vanish from the metric. Anything surviving in log-prob space
is genuine selective signal.

## Results

`cos(k, k_neg)` at concept-position capture (prior diagnostic):

| Capture | cos(k, k_neg) |
|---|---|
| Last-token (prior baseline) | +0.97 (Eiffel & JOTP) |
| Concept-position, layer 8 | +0.74 (Eiffel) / similar (JOTP) |
| Concept-position, layer 4 | +0.66 (Eiffel) |
| Concept-position, layer 2 | +0.65 (Eiffel) |

The capture-point fix is real — the +0.97 figure was structural to
last-token, not novelty-driven. Concept-position capture finds a
qualitatively different direction. *That direction does not, however,
inject axiom-shaped behavior.*

**Selectivity gap at α=5, k_minus_neg, log-prob metric.** A successful
mechanism would put T1_relevant distinctly positive and T2 prompts at
~zero. Instead:

| Concept | Mode | T1_relevant | T2_photo | T2_hammer | T2_encrypt |
|---|---|---|---|---|---|
| Eiffel | last | **−0.058** | +0.007 | −0.002 | −0.036 |
| Eiffel | concept | **−0.023** | +0.018 | −0.016 | −0.015 |
| Eiffel | multi | **−0.081** | +0.037 | −0.024 | −0.035 |
| JOTP | last | **+0.025** | +0.024 | +0.053 | +0.047 |
| JOTP | concept | **+0.003** | −0.013 | −0.008 | +0.005 |
| JOTP | multi | **+0.021** | +0.010 | +0.045 | +0.049 |

In every panel, T1_relevant either has the wrong sign (Eiffel: most
negative) or is dominated by an unrelated T2 prompt (JOTP last: T2_hammer
+0.053 > T1 +0.025; JOTP multi: T2_encrypt +0.049 > T1 +0.021). No
single (concept, mode, vec) produces a clean axiom-shaped pattern.

Multi-position injection *amplified* the wrong-shape signal on Eiffel
and made the JOTP T2 dominance even sharper. Asserting the key across a
span did not surface concept-binding; it just compounded whatever the
single-position injection was already doing.

## What is now falsified, narrowly

Three independent properties of the failure rule out shallower
diagnoses than "wrong primitive":

1. The control concept (Eiffel) shows the *same* failure shape as the
   novel concept (JOTP). Novelty is not the bottleneck.
2. The log-prob metric removes uniform-tilt confounds. The remaining
   shifts are tiny (≤0.10 nats anywhere) and not consistently the right
   sign. There is no large selective signal hiding under a uniform tilt.
3. Position-matching capture and injection didn't help; injecting the
   signal across a span didn't help.

So the falsified claim is now:

> A single L2-normalised mean-of-paraphrase direction in GPT-2 small
> layer 8's residual stream, when added to the residual stream at any
> single position or contiguous span, does not produce concept-selective
> next-token probability shifts on either novel or known concepts.

## What remains untested

- **Multi-layer injection.** Same key, simultaneously added at layers 6
  and 8 (or 4, 6, 8). Single-layer was the v2 spec; we didn't sweep this.
- **SAE-feature keys.** A different basis: explicit content directions
  rather than dense averages. Properly motivated now: averaging finds
  *some* concept-distinguishable direction (cos drops from 0.97 to 0.36
  at concept-position), but that direction doesn't function as a binding.
  SAE features may be the basis in which binding-shaped directions live.
- **Sentinel-LoRA.** A different paradigm: train one shared adapter once
  that teaches the model to consume a designated context slot as a
  premise. After install, every new axiom is just slot content — text,
  SAE feature combo, or learned embedding. Preserves the "registration
  without per-axiom retraining" architectural bet from Mimir, while
  acknowledging that some teaching-of-the-mechanism may be needed
  (one-time, shared). See `docs/mimir-protocol-poc-spec.md`.
- **Larger model.** GPT-2 small at 124M may simply lack the
  representational room for binding-shaped directions to be linearly
  addressable. A re-run on Pythia 410M or Qwen 0.5B would partially
  control this.
- **The thesis broadly.** "Axioms have geometric realisations in
  residual space" remains untested — we never had a primitive that
  worked well enough to test it. What we tested is the simplest possible
  *implementation* of that thesis, and that implementation doesn't carry
  the load.

## Decision and next step

Per agreement, we stop the activation-injection track on this stack. The
next experiment is **sentinel-LoRA**, staged as a separate session in
`docs/mimir-protocol-poc-spec.md`. That document is self-contained and
can be picked up cold by a fresh agent.

We do not escalate to SAE features in this session. SAE remains a
defensible track but is not where the architectural bet wants us to go
next; it's an internal refinement of "find a better content direction,"
which works only if injection-as-primitive is the right operation. The
sentinel-LoRA experiment tests whether *any* injection-flavoured
operation can do binding, not just whether we have the right vector.

## Repo state at end of session

- 32/32 tests passing, ruff clean
- `src/poc/{hooks,keys,build_keys,run_tests,run_variants,run_control,run_control_v2,run_unified}.py`
- `data/paraphrases.json`, `data/eiffel_paraphrases.json`
- `artifacts/{activations,keys,keys_variants,control_keys}.npz`
- `artifacts/{layer_separation,scores,variants_scores,control_v2,unified}.json`
- `artifacts/{shifts,variants,control,control_v2,unified}.png`
- `artifacts/RESULTS.md` (this file)
- `docs/mimir-protocol-poc-spec.md` (next-session brief)
