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
