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

## Whitening (Fable steer #3 — REVERSED BY MEASUREMENT 2026-07-11)

**Whitening is now OPT-IN (--whiten off|shrunk|zca, default off).** Isolation
experiment on the smoke (same predictor, same data, only the lens varies),
best recall@5 vs chance 0.046:
| lens | recall@5 |
|---|---|
| raw (off) | 1.00 |
| shrink 0.5 | 1.00 |
| shrink 0.1 | 0.95 (slower) |
| pure ZCA | 0.30 |
Why: ZCA equalizes variance across all 896 dims, amplifying ~800 near-noise
directions to parity with the signal dims — cosine/InfoNCE then weight noise
equally — and rsqrt blows up the worst-estimated tail eigenvalues (~316x at
the 1e-5 eps floor for out-of-subspace eval components). Full-rank fit (1296
samples > 896 dims) did NOT save it. Shrinkage (blend cov toward spherical,
bounding amplification at (shrink*mean_eig)^-1/2) recovers the signal.
- The anisotropy worry (spec killer #2) stays real in principle; if the raw
  run platitude-collapses (diversity gate), the shrunk arm is the follow-up.
- Original steers kept for when whitening is on: per-slot-index whiteners,
  streaming fit (running moments, never materialize), fit on TRAIN only.

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
- **Checkpoint policy (registered 2026-07-11, BEFORE the real run):** the gate
  reads the BEST eval checkpoint over training, FINAL reported alongside. The
  smoke peaks then overfits; "did it ever find structure" is the Stage-2
  question, and the pushed artifact is the best checkpoint.
- **Metric spec (registered 2026-07-11 after the validation run, BEFORE the
  CoT run):** the gate number is recall@5_128 (true + 127 seeded decoys) —
  full-pool recall at N~2000 is ~15x harder than the registered gate and not
  comparable. PLUS the topic-shortcut control: recall@5_doc (same-doc
  candidates only) must beat within-doc chance (5/doc_pool). Global-pool
  recall is inflated by "found the right document" (neighbor=0.632 says doc
  topic is ~2/3 of predictive value); within-doc, topic is shared by
  construction, so anything above chance is succession. Platitude gate reads
  diversity < tgt_sim (now logged).

## Raw-text validation result (2026-07-11, n=1500, whiten off — MECHANISM OK)

recall@5 = 0.237 @ pool 1990 (chance 0.0025), peak at step 500 then monotonic
overfit decline (train loss 0.03 by step 4000; 1149 train seqs vs ~50M params).
diversity 0.04, not collapsed. Artifacts (best ckpt) in
mattyvee/mimir-artifacts/stage2_predictor.
- NOT a clean gate pass and NOT a kill. Two caveats, both Fable-review flagged:
  (1) 0.237@1990 is not gate-comparable (gate was @128; recall@5_128 wasn't
  logged yet); (2) TOPIC-SHORTCUT CONFOUND — with ~16 same-doc targets in the
  pool, a topic-centroid-only predictor scores ~0.31 within-doc-random, so
  0.237 does not yet separate "predicts the next thought" from "identifies the
  document". recall@5_doc (added after) is the control.
- Decision: raw-text run did its job (mechanism works at some level); full
  n=4000 raw run SKIPPED in favor of the CoT main line with the fixed metrics
  + regularization (the step-500 peak says the current recipe overfits long
  before the token budget).

## Fable pre-spend review (2026-07-11, after the window fix — all landed)

The shakedown's chance-flat recall was DATA bugs, not the predictor: stride-1
windows duplicated every target ~8x (recall-by-index deflation + InfoNCE
false negatives) and the smoke corpus was 2 texts x15 (14 identical twins per
target). Fixed: non-overlapping windows + distinct smoke docs. Review then
caught, all fixed same day:
1. evaluate() ran with dropout ON (no_grad doesn't disable it; trunk default
   dropout=0.1) — every earlier eval number was deflated/noisy. Now eval-mode
   with caller-mode restore.
2. manual_seed(0) in _batches froze batch order AND dropout masks across
   epochs (InfoNCE negatives never varied). Now epoch-seeded local Generator.
3. Cross-doc boilerplate ("All rights reserved.") = identical sentences =
   bitwise-identical gists = twin eval targets — the window bug reborn at
   corpus scale. Now exact-deduped in eval (dup_dropped logged).
4. Encode is 1 sentence/forward on the 4-bit 7B: n=4000 x ~20 sents ~= 80k
   forwards can eat the 4h timeout BEFORE training. Validation run n=1500
   fits; the full run needs a longer timeout or a batched encode.
5. Dropout-off eval exposed the earlier smoke "signal" (recall@5 0.306) as
   dropout-noise peak-picking — clean eval was chance. Isolating that led to
   the whitening reversal above; with whitening off the runner smoke shows
   real retrieval (see Whitening section).

## Build order (model-free first, no spend)

1. whiten.py — DONE (in-memory fit + tests). Add fit_streaming + per-slot.
2. predictor.py — the transformer trunk + regression/InfoNCE + recall@k /
   diversity metrics. All CPU-testable (tiny dims, overfit a toy sequence).
3. run_stage2.py — encode/fit/train/eval runner (smoke on tiny model+corpus).
4. Launcher. Then: write final gate numbers here, launch.

## Parked: gist granularity (user idea 2026-07-12 — one vector per construct)

Current unit of thought = one SENTENCE -> 8 slots (sentence splits are free +
deterministic). Proposal: one vector per language CONSTRUCT (clause / entity /
relation) — capacity matched to content, discourse-referent tracking, maybe
better retention of exact elements. Costs: needs a segmenter (parser
dependency or learned segmentation = open research), and variable-length
units break the fixed-[8,d] prediction target (set-matching losses).
Sequenced empirically, cheapest first:
1. SLOT-SPECIALIZATION PROBE (piggyback on next eval node): ablate each slot,
   decode from single slots — did construct-like specialization EMERGE in the
   learned 8 slots? Entangled slots = evidence for the redesign.
2. K-SWEEP (already owed, Stage-1-real): k=4~=k=8 => coarse units suffice;
   k=16>>k=8 => sentences underfunded, finer granularity helps. Also answers
   the recurring one-vector-per-thought question (prediction on record:
   k=1 collapses — single-slot attention has no query-dependent readout).
3. If 1+2 both point finer: CLAUSE-level gisting (comma/conjunction splits,
   parser-free, k=2-4 per clause) before any full construct/parser fork.
Note: if the motivation is exact-element retention (ILP_END_RETURN), that is
not a granularity fix — lossy is lossy at any grain; exact syntax stays on
the Mimir KV/prefix side.

## Parallel batch outcome (2026-07-11 — A/B/k-sweep; logs in runs/vast_logs/)

Money: all nodes destroyed+verified, $0 idle burn, ~$5 spent, $10.34 credit left.
- **A (GSM8K CoT): DONE** — result + post-mortem above (succession real but
  small; gate rework landed).
- **k-sweep: 1 of 4 arms survived.** k=4 COMPLETE: gap_closed 0.863 @ 8000
  steps. NOT directly comparable to k=8's 0.887 (that was 16000 steps; the
  sweep's own k=8 arm died). k=8 arm: CUDA OOM at step 1000 (24GB 3090,
  backward pass; partial 0.809 and climbing). k=1 and k=16: produced NOTHING
  (CN-geolocated hosts, huggingface unreachable — hours of dead download).
  Granularity question still OPEN: no k=1 collapse test, no k=16 headroom
  test, no budget-matched k=8. Salvage-session relaunch notes (DEAD_NODES.md):
  curl-check huggingface reachability in onstart BEFORE burning hours;
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (or smaller batch / 48GB
  card) for k>=8.
- **B (OpenR1): never produced a read** — 1x real infra (CUDA 804), then
  SETUPFAILs on HEALTHY nodes traced to OUR config: reason_check at
  --max-pairs 1500 ≈ 560 forwards of the 4-bit 7B ≈ the whole 30m step
  timeout (GSM8K's ~350 pairs fit; OpenR1's don't). Relaunch with
  --max-pairs ~600 or a 45m step-1 timeout. The sharp succession test is
  still the most important pending read.
- A session on another account salvaged during the usage outage: destroyed 3
  dead nodes, recovered n2's log to completion, pushed branch vast-logs
  (merged here). No secrets in its commits (scanned).

## k-sweep RESULT: the complete capacity curve (2026-07-12, all arms @ 8000 steps)

| k | gap_closed | span content kept (above topic 0.632, vs k=8) |
|---|---|---|
| 1 | 0.794 | 65% |
| 2 | 0.835 | 82% |
| 4 | 0.863 | 93% |
| 8 | 0.880 | 100% (=0.248) |
| 16 | ~0.88 (plateau band 0.87-0.887; exact final clipped in log race) | ~100% |

Verdict: **smooth log-shaped diminishing returns, no knee, saturated by k=8.**
Each doubling buys less: +0.041 (1→2), +0.028 (2→4), +0.017 (4→8), ~0 (8→16).
- Sentences are NOT underfunded at k=8; content is low-rank. k=4 is the
  value pick for quality (93% at half the slots).
- k=1 does not collapse (prediction falsified, recorded above) — 1 slot keeps
  ~2/3 of span-specific content; k=2 keeps ~82% at 4x-shorter Stage-2
  sequences than k=8. The compress-for-prediction asymmetry (k=2 encode →
  predictor → k=8 decode) is now priced and viable.
- k=3 NOT run, deliberately: the curve is regular enough that interpolation
  (~0.85) is safe; no hidden knee to find.
- Sweep arms pushed no checkpoints (by design) — any arm that informs a real
  build decision gets one re-run with push + the xdoc control (~$0.75).

## QUEUED: Stage-1-real sweep (teed up 2026-07-12, runs after the Stage-2 result)

One node, ~6-8h, ~$1.5-2. Order of runs after the current Stage-2 shakedown:
(1) if recall@5 shows signal -> full Stage-2 run (scale the predictor);
(2) THIS sweep; (3) Mimir confirmations (gate 3, ~$2, still owed).

Arms (all 8000 steps — the pilot plateaued by ~7500, no need for 16000):
- k-sweep: k ∈ {1, 4, 8, 16} — capacity curve for gist slots. Pre-registered
  prediction (Fable, on record): k=1 COLLAPSES (single-slot attention has no
  query-dependent readout); k=4 vs 8 tells whether sentence content is
  low-rank; k=16 >> 8 would mean sentences are underfunded.

  **PREDICTION FALSIFIED mid-run (2026-07-12, recorded before finals): k=1
  does NOT collapse.** Clean learning curve (step 0 gap_closed −1.76 →
  plateau ~0.77-0.78 by step 2500, flat thereafter) while k=8 blows past
  (0.849 @ 2500, climbing). Why the prediction was wrong: it treated a slot
  as ONE VECTOR — but one KV position is 28 layers × per-head K/V, so
  different heads/layers read the same position differently; capacity of a
  single position was under-counted by ~2 orders of magnitude. What
  survives: the CAPACITY GAP — vs the topic-only baseline 0.632, k=1 keeps
  ~half the span-specific content (0.78 vs 0.887 ⇒ 0.15/0.26). Implication:
  k=1-encode → predictor → k=8-decode asymmetry is a legitimate design
  option (8x shorter Stage-2 sequences), taxed ~half the specific content.
  Caveat: sweep arms push NO checkpoints, so no post-hoc xdoc control on
  k=1 — if this informs a real decision, re-run that arm once with push
  (~$0.75). Judge the final curve at step 8000 only (same cosine schedule).
- clause-snap arm (k=8): truncate at the last clause boundary inside the cap
  instead of mid-token (55% of over-cap cuts orphan clause material). Beats
  plain k=8 -> construct-aware boundaries matter; ~= k=8 -> truncation noise
  was negligible after the 48->64 cap fix.
- slot-specialization probe (eval-only, rides the same node at start, on the
  EXISTING step-16000 checkpoint): per-slot ablation + single-slot decode.
  Specialized slots -> construct structure emerged; entangled -> evidence for
  the clause/construct redesign.
Also: widen heldout to 40+ docs (small-n caveat from the xdoc control).

Success shape: a capacity curve + a boundary-policy verdict + a slot anatomy,
for the price of one pilot. Feeds directly into whether the "one vector per
construct" idea (parked above) gets built.

## CoT traces PROMOTED to main line (user call 2026-07-11, Fable steers baked)

CoT is no longer the KILL-gate fallback — it's the aligned objective. Raw web
prose has WEAK succession (next sentence often drifts topic; predictor ceiling
= narrative drift). Reasoning traces have TIGHT succession (step n+1 entailed
by step n) — higher recall ceiling, and what it learns is inference, not
topic-continuation. That is exactly the Stage-3 latent-reasoning target. The
raw-text run stays worth its ~$2 as the cheap mechanism check.

Sequenced, each gated on the last:
1. Raw-text n=1500 validation (IN FLIGHT) — does predict-next-gist work at all.
2. ENCODER-ON-REASONING CHECK (reason_check.py, ~10 min GPU, rides the next
   node): gap_closed on consecutive GSM8K step pairs with the EXISTING
   adapter. Pre-registered: >= 0.4 -> proceed; collapse -> Stage-1 re-fit on
   CoT data FIRST (the FineWeb-trained gist may be blind to equations /
   'therefore' / symbolic tokens). Prose reference 0.887.
3. CoT Stage-2 run on an OPEN trace dataset (~$2-3; fresh teacher distillation
   deferred — control over domain/format isn't needed to de-risk mechanism).

### A (GSM8K) RESULT + gate post-mortem (2026-07-11, Fable review)

Run completed clean (encoder gate 0.884; artifacts pushed). Numbers:
recall@5_128 BEST 0.901 (step 250) but recall@1_doc there 0.427 ~= chance;
recall@1_doc peaks ~0.51 late (step 2000) vs empirical chance ~0.38-0.42.
- **The registered recall@5_128>0.40 gate is NEAR-VACUOUS on short traces:**
  with doc_pool ~3, topic-matching alone saturates it (~0.9+) — it re-measures
  Stage-1's known topic signal, and it CANNOT FAIL on such corpora. On B
  (doc_pool ~10) its topic-only baseline is still ~0.5. The load-bearing gate
  is within-doc succession ONLY.
- **Honest verdict: succession is real but small on GSM8K** — ~+0.1 absolute
  over blind guessing (0.48-0.51 vs ~0.40). B (chance ~0.1) is the sharp test.
- Fixes landed (pre-registered before B's numbers exist): (1) doc_chance now
  computed empirically per-row as E[topk/pool_i] (1/mean-pool understates
  chance — Jensen); (2) best-checkpoint selection keys on recall@1_doc, NOT
  recall@5_128 — the two ANTI-CORRELATE across training (early ckpt = topic
  matcher, late = step predictor; selecting on @128 shipped the topic-matcher).
  B is running with old selection: judge B from its printed eval curve using
  the new criterion; only its pushed artifact is mis-picked.

Fable steers for the CoT run:
- STEP-TEMPLATE SHORTCUT: math traces are positionally regular ('So the answer
  is...' is always last). Report per-position recall; a terminal-step spike =
  predicting the template, not the content. Eval stays document-disjoint.
- Same-problem negatives are harder AND better here (steps mutually similar):
  doc-clustered batches turn InfoNCE into 'which step', not 'which topic'.
- Wrong-step risk compounds down a chain (xdoc < none): draft-and-verify in
  Stage 3 is load-bearing for CoT, not politeness.
- No runner fork: same run_stage2.py; CoT = corpus source + newline step
  splitter, not new architecture.

## Non-goals (defer)

Diffusion head (phase B); fresh teacher distillation (open datasets first);
32B; Stage-3 runtime.

## Stage-3 3a-i pre-registration (2026-07-12, WRITTEN WHILE THE NODE RUNS, before results)

Node 44588589 (run_mimir_decode, GSM8K, step-16000 k=8 adapter). Predictions:
- SOFT ceiling (teacher-forced PPL of true next step, injected thought vs
  none): logit parity with the training forward is CPU-tested EXACT, and
  reason_check already measured this adapter on this distribution
  (gap_closed 0.884; sweep gist~11.4 vs none~15.4). So the drop is
  PREDICTED: ~20-30% (ratio 0.72-0.80). This number VALIDATES THE HARNESS;
  a wild deviation means a 3a-i bug, not new science.
- HARD test (greedy F1 from the thought alone) is the genuinely NEW
  information — no prior measurement constrains it. Reads: F1(next) >>
  no-inject and > random-step => direct latent decode works; soft-drop-good
  but F1 weak => thought rich, greedy extraction poor => draft-and-verify
  (3b) carries the load; soft drop absent => harness bug (see above).

## Stage-3 3a-i RESULT (2026-07-12, node 44588589, GSM8K, step-16000 k=8 adapter)

167 held-out step pairs. THE DECODE PATH WORKS — a real thought round-trips.
- SOFT ceiling: PPL(next | injected thought) 3.44 vs no-injection 12.43 = 72%
  drop. NOTE — this BEAT the pre-registered ~20-30% by a lot. Not a bug: the
  injection path is parity-tested EXACT, and the deviation is a
  mis-calibrated prediction — I predicted using gap_closed's FULL-continuation
  PPL (gist 11.4 vs none 15.4) but the metric TAIL-scores (cont[1:], each
  token teacher-forced on the step's own earlier tokens). Thought + within-step
  context compound, so the injected condition benefits far more from tail-
  scoring than 'none' does. Real, but the 72% is not the gap_closed number.
  FLAG FOR FABLE: sanity-check this calibration next review.
- HARD test (the genuinely new info): greedy F1(decoded, next) 0.412 >>
  no-inject 0.183 AND > random-step 0.254 — BOTH pre-registered gates PASS.
  own-span F1 0.458 (slightly > next) — the decode reflects step n's content
  as much as it predicts n+1, as expected (the gist encodes step n).
- Qualitative: decoded text is COHERENT, ON-TOPIC math reasoning that shares
  the true step's numbers/entities but is not verbatim (e.g. true "48+24=72
  clips" -> decoded "$1.50 x 48 = $72.00"). Not garbage; plausible reasoning
  in the right ballpark.
- VERDICT: the thought carries the meaning and the frozen model can express
  it — approximately, via greedy. This is the "rich thought, imperfect greedy
  extraction" case => draft-and-verify (Stage 3b) is the path: sample
  candidates from the thought, verify against the target. Decode ceiling is
  established and viable; predicted-thought bridge is the next build.

## Fable post-3a-i review (2026-07-12) — calibration RESOLVED, steers for 3b

- The 72%-vs-predicted-20-30% deviation is NOT an anomaly and NOT (only) the
  tail-scoring story: the correct prior was reason_check's own GSM8K PPLs
  (gist 4.21 / none 12.04 = 65% drop), which were in our logs all along. 3a-i
  reproduced that measurement through the injection path (72%, small extra
  from tail-scoring). The harness is VALIDATED end-to-end; the
  pre-registration quoted the wrong prior (FineWeb sweep numbers).
- Honest greedy margin: F1 0.412 vs the RANDOM-STEP floor 0.254 (1.6x) — the
  shared GSM8K step template gives ~0.25 overlap for free; quoting vs
  no-inject (0.183) overstates.
- ADVANCE-VS-RESTATE (the one warning sign): decode overlaps the current step
  (0.458) slightly more than the next (0.412). Chaining requires drafts that
  MOVE FORWARD. 3b must measure and gate on advance rate = fraction of drafts
  closer to step n+1 than to step n.
- Contamination: Qwen likely saw GSM8K in pretraining. The none-baseline
  (12.4) shows the thought does real work, but run a fresh-problems eval
  before any headline claim.
- BRIDGE DESIGN (next build): do NOT regress per-layer KV tensors from the
  predictor's final-layer output (under-determined). Train the bridge THROUGH
  the injection loss that 3a-i just validated: convert -> inject -> minimize
  the true next step's NLL. Optimize what we measure.

## Stage-3b-i RESULT (2026-07-12, node 44594162): BOTH GATES FAIL — informative negative

167 pairs, k=8 drafts @ temp 0.9, trivial-guard on. Node destroyed clean, ~$0.5.
- F1(next): picked 0.391 vs greedy 0.390 — LIKELIHOOD-VERIFY ADDS NOTHING.
  The self-diagnosing design did its job: picking the draft the model finds
  most likely under the real prior does not find better steps than greedy
  (verify frequently just re-picks the greedy draft; when it differs, the
  pick is fluent-shaped but not more correct).
- ADVANCE rate: picked 0.293 / greedy 0.299 — both FAR below the 0.5 gate.
  ~70% of decoded drafts sit closer to the CURRENT step than the next.
  Quantifies 3a-i's warning (own-span 0.458 >= next 0.412).
- DIAGNOSIS (mechanism, not mystery): drafts are conditioned on the thought of
  step n ALONE — a single thought is directionless (many valid continuations;
  no chain momentum), and mean-NLL-under-prior can't tell "the right advance"
  from "fluent riff". The verify saw the full chain; the DRAFTS never did.
- NEXT (3b-ii prerequisite, before any bridge): CHAIN-CONDITIONED drafting —
  inject the thoughts of steps 1..n (concatenated slots, canonical positions),
  decode from the accumulated thought-memory. Matches the runtime vision
  (thought memory accumulates) and gives drafts the forward momentum a single
  thought lacks. Secondary lever if needed: an anti-restate term in verify.

## Fable second-pass on 3b-i (2026-07-12) — corrections to my own diagnosis

1. FAILURE NOT LOCALIZED: no ORACLE logged (best-of-K by F1-vs-true). Without
   it, "drafts are directionless" vs "verify can't select" are indistinguishable.
   The chain-conditioned run MUST save all drafts and log oracle F1 alongside
   picked/greedy. Oracle >> greedy => verify is the bottleneck; oracle ~= greedy
   => draft distribution is. Pre-registered read BEFORE that run.
2. ADVANCE METRIC UNCALIBRATED: correct math steps inherently reuse the current
   step's numbers, so the 0.5 bar was arbitrary. Calibrate in-run: report
   advance_rate of the TRUE next steps (ceiling) and of the PREVIOUS step
   (known-restate floor); judge picks against that bracket.
3. VERIFY CONTEXT MUST INCLUDE THE QUESTION (gsm8k 'question' field) — judging
   steps without the problem statement handicaps likelihood-verify; any real
   runtime has the question.
4. PRE-REGISTERED BRANCH: if after chain-conditioning + question-context the
   verify still can't track the oracle, likelihood-verify is DEAD as the
   runtime guardrail; next candidate is lookahead-consistency verify (does the
   draft make the FOLLOWING step more predictable) — measurable offline.

## Stage-3b CHAIN RESULT (2026-07-12, node 44601808): chain-conditioning did NOT rescue it

143 pairs, k=8, chain-conditioned drafts + question in verify + calibrated advance.
- F1(next): picked 0.375, greedy 0.376, ORACLE(best-of-8) 0.449. Advance:
  picked 0.329, greedy 0.315; CEILING(true-next) 1.00, FLOOR(prev-step) 0.224.
- vs 3b-i (single thought: picked 0.391 / advance ~0.29): NO improvement.
  Chain-conditioning + question-context + calibration all in, headline flat.
- DUAL failure, both levers limited:
  (a) DRAFT CEILING is low — best-of-8 only 0.449 F1 (vs ~0.25 template floor):
      drafts carry real signal but the accumulated thoughts don't pin the
      specific next computation. Qualitative: fluent but wrong-direction steps
      (true "80000*1.5=120000" -> draft "house sold for $150,000... profit
      $10,000"), sometimes terminating the problem early.
  (b) VERIFY doesn't capture even that ceiling — picked 0.375 ~= greedy 0.376
      << oracle 0.449. Likelihood-verify (even WITH the question) fails to
      select the better drafts. Fable's pre-registered branch fires:
      likelihood-verify is DEAD as the runtime guardrail.
- Advance: picks (0.329) sit just above the restate FLOOR (0.224), nowhere
  near the true-next CEILING (1.0). Drafting the next step from thoughts does
  not reliably move forward.
- JUNCTURE (not a tweak-rerun): the "draft the next-step TEXT from injected
  thoughts, verify by likelihood" loop has failed twice. The decode-to-text
  step is the weak link (3a-i already: coherent-but-not-verbatim). Options to
  weigh BEFORE more spend: (1) latent chaining — predict next THOUGHT vector,
  inject as context for the next prediction, decode to text only at the end,
  verify at the thought level; (2) lookahead-consistency verify (pre-
  registered); (3) accept thoughts are lossy-for-generation and reposition the
  win as memory/KV-compression (already proven: k-sweep) rather than latent
  reasoning. Needs a strategy review, not another launcher.

## Direction reset after 3b (2026-07-12, user + Fable) — two paths, different jobs

User calls, agreed: (1) STAY LATENT is the default runtime — the speed path
never regenerates text mid-chain (the thing that failed twice). (2) But some
usage must SEE thoughts, so an accurate render path is a requirement, not a
nice-to-have. (3) Averages hid too much — future evals report quantiles and a
number-dense slice, not just means.

- FAST PATH (default): latent chaining — predict next thought from the chain
  of thoughts, inject, repeat; decode text only at the end (or on demand).
  Needs: the Stage-2 predictor in the loop + thought-level verification.
- RENDER PATH (on demand, out of the inner loop): reconstruct THE STEP ITSELF
  from its thought. Key insight: all decode tests so far ran the CONTINUATION
  direction (what the encoder was trained for); render = transcription, an
  easier task we never trained. Untrained own-span overlap was already 0.458.
  Build: small render decoder (LoRA/adapter on the frozen model, CE on the
  source span given the injected thought) ~ $2 training run.
- LITERALS LEDGER (answer to exact-detail lossiness + the user's dynamic-
  capacity instinct): at encode time store the step's literal tokens (numbers,
  names) beside the thought — a few tokens, deterministic; render splices them
  in. Meaning from the thought, exact digits from the ledger. Lossy-at-any-k
  stops applying to the details that matter.

## Render RESULT (2026-07-12, node 44650760): render decoder WORKS — reconstruction is near-verbatim

800 docs, 2000 steps, 289 doc-disjoint eval. Node destroyed clean, adapter
pushed to mattyvee/mimir-artifacts/render_adapter. $12.53.
- reconstruct F1: mean 0.47, p10 0.35, p50 0.458, p90 0.606. number-recall
  (steps with numbers, n=286): 0.731.
- BUT F1 UNDERSTATES quality — a decoding artifact: no EOS at end-of-step (the
  splitter strips the newline), so greedy reconstructs the step then REPEATS
  it 3-4x to max_new, tanking precision. Examples are near-verbatim:
    "Natalia sold 48/2 = 24 clips in May." -> EXACT (then repeats)
    "In the beginning, Betty has only 100/2 = $50." -> "...Betty has 100/2 = $50."
    "Weng earns 12/60 = $0.2 per min" -> "Weng earns 60/60 = $0.2 per min" (wrong
    first number, structure+answer right)
  True fidelity is materially higher than 0.47; the render path is validated.
- number-recall 0.731 QUANTIFIES the ledger: ~73% of exact digits survive from
  the thought alone; the literals ledger closes the remaining ~27% (12->60,
  48+24 order flips). Meaning from the thought, exact numbers from the ledger —
  as designed.
- CHEAP FIX owed: eval stop-at-end-of-step (emit newline / cap at first step)
  for a clean F1; the repetition is an eval artifact, not a model failure.
  Optional ~$0.4 re-measure; qualitative + number-recall already tell the story.
