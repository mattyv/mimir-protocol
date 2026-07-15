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

## ROLLOUT RESULT (2026-07-14): open-loop latent chaining drifts after ~2 steps

The real fast-lane test. Free-run the predictor on its OWN outputs from a 2-step
real prefix; at each rollout depth inject the predicted thought and score
gap_closed on the true next step, vs a teacher-forced control (predict from TRUE
history). 508 held-out docs, n=441 (depth 1) tapering to 85 (depth 12).

| depth | free (chained) | teacher-forced | shuffled floor | free_cos |
|---|---|---|---|---|
| 1 | 0.56 | 0.56 | 0.19 | 0.67 |
| 2 | 0.48 | 0.62 | 0.37 | 0.65 |
| 3 | 0.19 | 0.60 | 0.19 | 0.62 |
| 4 | 0.13 | 0.63 | 0.21 | 0.58 |
| 6 | −0.06 | 0.64 | 0.27 | 0.56 |
| 9 | −0.43 | 0.74 | 0.50 | 0.50 |
| 12 | −3.37 | 0.68 | 0.47 | 0.47 |

Read:
- **Teacher-forced holds flat at ~0.6-0.7 across ALL depths.** The
  predictor+bridge substrate is solid at every step — repeated single-step
  prediction never degrades. So the failure below is the CHAINING, not the
  pieces.
- **Free-running collapses fast.** d1 = 0.56 (== tf by construction, all-real
  history), still 0.48 at d2, but by **d3 it's 0.19 — down at the shuffled
  floor** (0.19), i.e. a chained thought is already no better than a random one.
  By **d6 it goes NEGATIVE** (−0.06): injecting the drifted thought is worse than
  injecting nothing. free_cos decays smoothly (0.67→0.47) — the drift is gradual
  in geometry but injection amplifies "slightly wrong" into "harmful".

**Verdict: open-loop latent reasoning survives ~2 steps, then drifts off the
manifold.** This is the "errors compound" failure this plan predicted for naive
chaining — now measured precisely: usable depth ≈ 2, harmful by ≈ 6. The
confidence gate was meant to catch exactly this and is dead, so open-loop
chaining is not viable.

(Fable review caveat — two causes conflated: the bridge was trained to tolerate
inputs at cos ~0.89 from clean (noise 0.5); by d3 the chained thought sits at
cos ~0.6, far outside that ball, so part of the cliff may be BRIDGE brittleness
on out-of-ball inputs rather than prediction drift alone. A heavier-noise bridge
might chain somewhat deeper; doesn't rescue open-loop, but don't cite
"prediction drift" as the sole cause.)

NOT purely negative — the reframe the numbers point to:
- Teacher-forced staying at 0.65 means a real step RESETS the drift. So latent
  reasoning works in SHORT BURSTS between real anchors: take 1-2 cheap latent
  steps, then decode+re-encode one real step to re-anchor, repeat. The speed
  story shrinks from "reason entirely in latent space" to "skip 1-2 big-model
  steps at a time" — still a real (if modest) win, and it needs no confidence
  gate.
- Next test if pursued: the anchored-burst schedule (k latent, 1 real, k latent,
  …) end-to-end vs plain generation — measure wall-clock and final-answer
  accuracy. Otherwise the render lane stays the headline deliverable and the
  latent-prediction thread closes with: substrate works single-step (0.62),
  chaining drifts (~2 steps), speed upside is bounded to short bursts.

## FRONTLOAD RESULT (2026-07-14): thoughts do NOT transfer to free generation

The clean instrument (after the burst harness proved unusable — its forced
line-per-step decode broke the baseline itself, plain 13%): give the model the
first half of a solution as context, then let it GENERATE FREELY. GSM8K test,
86 problems x 5 arms, healthy baseline this time:

| context given | accuracy | mean gen tokens |
|---|---|---|
| nothing (solve alone) | 0.698 | 89 |
| first half as TEXT | 0.826 | 56 |
| first half as TRUE thoughts | **0.326** | 46 |
| thoughts minus the last | 0.465 | 69 |
| thoughts + last PREDICTED | 0.488 | 86 |

Reads:
1. Instrument is sane: text context helps (+13 pts over none). And per the
   pre-registered rule, gist_pred vs gist_minus = +0.02 — a TIE: the predicted
   thought adds nothing measurable. Do NOT launch predictor-v2 on this basis.
2. The headline: TRUE thoughts as context HALVE accuracy vs no context at all
   (0.33 vs 0.70), and more injected thoughts = worse (minus-one arm 0.47 >
   full 0.33). Teacher-forced NLL gains (gap_closed 0.62-0.82) do NOT
   transfer to free generation.
3. Mechanism (from the generation dumps): injected arms are NOT derailed —
   text is fluent, on-topic, structurally identical to the no-context arm.
   But gist_true generations are HALF the length (46 vs 89 tokens): the
   injected KV convinces the model that solution content already exists (so it
   wraps up early) without the model being able to fully READ that content —
   premature conclusions from partially-legible context. Text context also
   shortens generation (56 toks) but is fully legible -> 0.83.
4. Caveat: none=0.70 means GSM8K gives context little headroom; a harder task
   (none low) is the one configuration where generation-time injection could
   still show value. The premature-wrap-up mechanism would likely persist.

### CHASE — hard problems (2026-07-14): caveat closed, AGAINST the thesis

Same instrument, GSM8K filtered to >=7 reference steps (mean 7.4, harder):

| context | acc (hard) | acc (easy, for ref) |
|---|---|---|
| none | 0.492 | 0.698 |
| text | **0.730** | 0.826 |
| gist_true | **0.111** | 0.326 |
| gist_minus | 0.222 | 0.465 |
| gist_pred | 0.222 | 0.488 |

The headroom hypothesis was RIGHT — and it damns the thoughts. none dropped to
0.49 (harder, as intended), and TEXT context opened up to +0.24 over none (vs
+0.13 on easy) — more headroom, exactly predicted. But thoughts didn't
capitalize; they got WORSE (gist_true 0.11, half the easy-problem number).
gist_pred == gist_minus again (predicted thought adds nothing). So: the model
uses legible context MORE on hard problems, and compressed-thought context LESS.
The premature-wrap-up failure amplifies with more content to (mis)read.

Bearing on the base-vs-instruct question: text context works and IMPROVES with
difficulty, so the failure is NOT the base model's instruction-following — it's
the frozen model's inability to fully READ injected compressed KV during
generation (the same limit the render lane needed a trained decoder to overcome).
An instruct model injects KV identically and would hit the same wall; an
instruct swap is therefore unlikely to rescue generation-time injection. (Worth
one confirmation someday, not a priority.)

**For the generation-time thesis: RAW-injected compressed thoughts do NOT work
as generation-time context, at any difficulty. Validated capabilities stand
(encoder, render+ledger, the likelihood ladder). The fast lane — in every form
tried (draft-verify, confidence gate, open-loop chain, anchored burst,
front-loaded raw injection) — is closed.**

CAVEAT (user BS-call, upheld in part): the "capacity is fine" evidence (k-sweep
saturating at 8) was measured in likelihood-space, which does not transfer to
generation — so "the model doesn't get enough OUT of the gist while generating"
remained live. The render receipt says the data is IN the gist (F1 0.99); what
the raw model lacks is a READER — and the render adapter (a LoRA on this same
frozen model) IS a trained gist-reader that no generation test ever used. The
missing experiment: RECONSTITUTE-THEN-SOLVE — inject thoughts, transcribe them
to text with the render adapter (its trained job), then solve from the
transcription with the adapter off. Prediction: recovers to ~= the text arm.

### RECONSTITUTE RESULT (2026-07-14, hard problems): split verdict

| context | acc |
|---|---|
| none | 0.492 |
| text | 0.730 |
| gist_true (raw injection) | 0.111 |
| **gist_render (reader transcribes, then solve)** | **0.492** |

1. **The reader interface CURES the harm completely**: 0.11 -> 0.49 (+38 pts).
   Making the content legible eliminates the premature-wrap-up poisoning. Raw
   injection was indeed the wrong interface — that part of the prediction held.
2. **No DETECTABLE value at n=63** (0.492 == none 0.492 by coincidence of
   31/63; the honest CI on gist_render−none is ~±12-17pts, so anywhere from
   mild help to mild harm). "Zero net value" was an overclaim on a point
   estimate — corrected (Fable reconstitute review).

CORRECTION (Fable review of the interpretation, 2026-07-14): claims 3-4 below
were NOT supported and are retracted.
- The "hard steps are LONGER, so the gist can't hold them" premise is FALSE:
  the 252 hard-run context steps are the SAME token length as render's training
  steps (mean 22.1 vs 23.2; p90 36 vs 34; 2/252 hit the 64-cap). No length gap.
- The "right numbers, wrong structure" mechanism rests on <=3 dumped problems
  (n~1 anecdote) — the harness scores the free generation, never the
  reconstructions themselves.
- token-F1 (the 0.93-0.99 receipt) + the ledger (digits given as a visible
  prefix) + true-first-token priming certify LITERALS, not RELATIONS. "right
  numbers, wrong operator" scores HIGH F1. So the reader's fidelity on relations
  was never measured, on easy OR hard steps.
- Compounding math: F1~0.9 => ~10-14% per-step structure error; (1-q)^4 over 4
  reconstructed steps/problem nets ~0.49 with NO hard-step-specific fidelity
  drop needed. The result may be exactly what render's KNOWN fidelity predicts
  on ANY steps once stacked 4-deep and the ledger stops flattering the metric.

Honest status: gist_render CURES the raw-injection poisoning (0.11->0.49, real).
Whether it fails to ADD value because (a) the 8-slot gist lacks relational
structure, (b) the render reader is a ledger-crutched decoder that never learned
structure, or (c) plain per-step-error compounding with no hard-specific drop —
is UNMEASURED. The three point to different fixes (k=16 encoder / relation-aware
reader / nothing).

Next: the CHEAP discriminator (~$1-2, pure eval, no generation) — reconstruct
the hard AND easy context steps and measure (i) a relation score (extracted
arithmetic ops vs gold), (ii) a WRONG-gist control (true ledger+first-token but
another step's gist — does the reader even read the gist?), (iii) an NLL
contrast on STRUCTURE tokens only, true-gist vs wrong-gist (is the relational
info IN the 8-slot gist at all — answered WITHOUT the decoder having to generate
it). This separates gist-capacity from reader-capacity before any retrain spend;
the retrain is NOT a clean discriminator in either direction. Also patch
run_frontload to log per-problem {pid, arm, correct} for paired stats.

### GISTPROBE RESULT (2026-07-14): READER-limited. The gist holds the structure.

397 context steps (256 hard / 141 easy), true-gist vs wrong-gist:

| | hard true | hard wrong | easy true | easy wrong |
|---|---|---|---|---|
| token F1 | 0.894 | 0.536 | 0.891 | 0.535 |
| number recall | 0.958 | 0.801 | 0.950 | 0.795 |
| relations exact | 0.636 | 0.159 | 0.697 | 0.066 |
| op-sequence match | 0.807 | 0.250 | 0.816 | 0.158 |
| struct-NLL (non-digit tokens) | **0.394** | **4.482** | 0.319 | 4.811 |

Verdict per the pre-registered grid:
1. **The relational structure IS in the 8-slot gist** — the 11x struct-NLL
   contrast (0.39 vs 4.48) is the direct, generation-free measurement. The
   WEB-trained encoder packs math relations into 8 slots recoverably. The
   "not enough data in the gist" hypothesis is REFUTED (as is my earlier
   "gist fidelity is the binding constraint").
2. **The READER is the wall** — it *recognizes* the correct structure (NLL
   0.39) but greedily *writes* only ~64-70% of relations exactly. Per Fable's
   compounding math, rel~0.64 over m~4 context steps predicts almost exactly
   the observed reconstitute tie (0.49 vs 0.73 text).
3. **No hard-easy cliff** (0.64 vs 0.70; NLL 0.39 vs 0.32) — the failure is
   difficulty-independent reader imprecision, not hard-step anything.
4. Wrong-gist F1 0.54 confirms Fable's metric critique: the ledger+first-token
   crutch alone buys half the F1 — token-F1 was never a structure metric.

Fix indicated: READER retrain (scale: 800->4000 docs, 2000->6000 steps — the
reader saw only ~2000 training examples). Pre-registered targets: relations
exact 0.64 -> >=0.85 on this probe, then rerun reconstitute expecting
gist_render 0.49 -> ~0.65+. The k=16 / CoT-encoder retrains are NOT justified
by this evidence.

### READER-V2 RESULT (2026-07-15): bar CLEARED — reader was the bottleneck

Reader retrained on 4000 docs / 6000 steps (5x data, 3x steps). Same probe:

| metric | v1 | **v2** | bar |
|---|---|---|---|
| relations exact (hard) | 0.636 | **0.921** | 0.85 ✓ |
| op-sequence match (hard) | 0.807 | **0.966** | |
| relations exact (easy) | 0.697 | **0.921** | |
| token F1 (hard) | 0.894 | 0.900 | — |

The bar (0.85) is beaten by +0.07; relations 0.64 -> 0.92 (+0.28). Token-F1
barely moved (0.894 -> 0.900) — the entire gain is in STRUCTURE fidelity, the
thing F1 was blind to and the probe exists to see. More data fixed the reader's
precision-writing; the "more data alone won't teach precision" prior was wrong.
Wrong-gist relations stay low (0.27) — v2 reads the gist more, not the prior.

Compounding forecast: 0.92 relations over m~4 context steps => 0.92^4 = 0.72 of
problems fully clean (vs v1's 0.64^4 = 0.17). Reconstitute gist_render should
climb from 0.49 toward the text ceiling 0.73.

Next (RUNNING): rerun the reconstitute solve-test with reader-v2. If gist_render
0.49 -> ~0.65+, the memory story validates end-to-end: store 8 vectors/step,
reconstitute on demand, solve at near-text accuracy.

### READING-WHILE-SOLVING RESULT (2026-07-15): null proxy, thread CLOSED (Fable call C)

The last live idea: attend injected gist KV DIRECTLY during solving (reader
adapter active), consuming the 3x KV-residency saving without transcribing.
Cheap proxy = a `gist_read` arm (render adapter active during the solve decode)
vs a `none_read` control (render active, zero gists). Hard GSM8K n=63, paired,
4090 (66 min, $~1.1; the 3090 attempt RC=124 timed out — run_frontload pushes
only at the end, fix owed):

| arm | acc |
|---|---|
| none | 0.524 |
| text | 0.730 |
| gist_true (raw inject, default) | 0.159 |
| gist_render (reconstitute->solve) | 0.619 |
| none_read (render on, no gists) | 0.000 |
| gist_read (render on, gists) | 0.016 |

Paired gist_read − none_read = +0.016, McNemar p~1.0 → NULL. But none_read=0.000
shows the render adapter active during solving destroys solving ENTIRELY (its
policy is transcribe-then-stop-loop), so the null is "wrong tool," not "gists
unreadable." Generations confirm gist_read READS the gists — it emits step-1
content given only as a gist — then loops; weak POSITIVE evidence gists survive
KV injection.

Decision C (close), reasoning = the ceiling, not a training-failure prior: a
dedicated solve-time read-head (B', ~$30-60) is capped above by gist_render 0.619
(perfect gist-reading's downstream value), which LOSES to text 0.730 and barely
beats none 0.524. Fable priors: ~35% B' beats none, ~5% it reaches the text bar
the user cares about (render F1~0.9 bounds what 8 vectors carry). Even success
buys a latency optimization of an already-dominated pathway. Reopen ONLY if a
deployment has a BINDING KV budget (text can't stay resident) AND accepts ~0.62.

## THREAD STATUS: generation-time use of thoughts is CLOSED.
Every lane tried and killed: draft-verify, confidence gate, open-loop chain,
anchored burst, front-loaded raw injection, reconstitute-then-solve (works but
dominated), reading-while-solving (null proxy + ceiling). Validated wins that
STAND: encoder (text->8-vec, gap_closed 0.88), render+ledger (8-vec->text F1
0.99 / numbers 100%), and the 8-slot structure-retention result (struct-NLL
0.39 vs 4.48). The honest identity: a validated compression+reconstruction
system for reasoning steps, not a generation accelerator.

### RECONSTITUTE-V2 RESULT (2026-07-15): net-positive — the memory lane works

Reconstitute solve-test rerun with reader-v2, same hard config (n=63, mean 7.4
steps, m~4 context steps reconstituted per problem):

| context | v1 reader | **v2 reader** |
|---|---|---|
| none (solve alone) | 0.492 | 0.492 |
| text (real words) | 0.730 | 0.730 |
| gist_true (raw injection) | 0.111 | 0.111 |
| **gist_render (reconstitute→solve)** | 0.492 | **0.603** |

The lane is now NET-POSITIVE: gist_render 0.603 > none 0.492 (+0.11), where v1
was a dead tie. It closes ~46% of the compression penalty (the none→text gap),
sits far above raw injection (0.11), and the per-problem dumps show it landing
correct exactly where raw injection fails. The reader fix (relations 0.64→0.92)
propagated to end-to-end accuracy, as the compounding math predicted (forecast
~0.65; got 0.60 — a touch under, because a clean reconstruction still requires
the model to reason).

**End-to-end validation of the memory story:** store ~8 vectors/step instead of
~25 tokens of KV, reconstitute on demand via the render reader, and solve at
0.60 vs 0.49 alone — a real, net-positive capability. Compression is USABLE, not
free: it still trails full text 0.73, so you trade ~13 pts of accuracy for the
~3x KV-memory saving.

Calibration: +0.11 at n=63 is real-looking but near the resolution limit
(paired McNemar ~±0.10-0.12); the mechanistic chain (relation fidelity 0.92,
per-example dumps, dose-response vs gist_minus/pred) makes it coherent, but a
confirm run at n~200 would bank it. The k=16 / CoT-encoder retrains remain
UNjustified — the gist held the structure all along; the reader was the wall.

**Standing conclusion for the thread: compressed thoughts are validated for
LIKELIHOOD (memory/compression: the ladder) and for RECONSTRUCTION (render
lane), but not as generation-time context in this configuration. Predictor
improvements (v2: question-conditioning + hard negatives, built + tested,
launch held) currently have no generation-side customer.**

Pre-registered design for the burst test (Fable review — without these arms it
proves nothing). Two untested assumptions it must cover: MIXED real/latent
history (rollout only measured all-real vs all-predicted) and free GENERATION
from injected predicted thoughts (everything so far is teacher-forced NLL):
1. plain full generation (baseline to beat on wall-clock at equal accuracy)
2. burst with TRUE thoughts injected (ceiling — does the schedule work at all)
3. burst with PREDICTED thoughts (the test)
4. burst with NO injection, steps just skipped (killer control: if ~= arm 3,
   the predictor contributes nothing and the win is model robustness)
Metric: final-answer accuracy + wall-clock. NOT NLL. One run, then close the
thread either way. (Validated 0.62 bridge frozen at HF bridge_validated/ so
burst work can't clobber it.)

## Non-goals / open

Diffusion thought-sampler (only if regression head plateaus — it has: top-1 0.30
+ flat confidence); 32B; the still-owed gate-3 Mimir confirmations (separate
track).
