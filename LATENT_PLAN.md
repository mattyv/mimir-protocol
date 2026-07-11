# Plan: latent thought-prediction on frozen Qwen (cheap stages only)

Source: the "Latent Thought-Prediction Architecture" spec (2026-07-08, user
upload), reviewed 2026-07-09. This doc records the approved sub-$10 program
and the review corrections. HARD STOP before anything $200-class (32B
training, multi-day runs) — those need a top-up and different infra.

## Corrections to the source spec (from the 2026-07 Mimir results)

1. **The refusal-prior framing is falsified.** Facts recall 20/20 and skills
   engage 5/6 on 7B-Instruct with zero preamble. The transferable lesson is
   "encode in-distribution for the checkpoint" (chat-template the injected
   KV), not "instruct models refuse injected content." "Base model only"
   (spec §1.2) stays, but for OOD-input reasons.
2. **The meta-KV preamble measured out as ~a no-op** (facts: BOUNDARY 5/6 ->
   6/6 only; skills: stance clause failed to stop bleed while the KV was
   present). Do not port it preemptively (spec §4.2); presence-control is
   what worked.
3. **Honest speedup band is 3-6x, not 5-20x** — the 20x rows assume
   near-unlimited latent chains; COCONUT needed training to get short ones.
4. **Stage 3b added (draft-and-verify):** tiny drafter proposes the sentence,
   frozen verifier checks it in ONE parallel pass; divergences yield the
   verifier's own token free. Output byte-identical to verifier greedy.
   Mechanics validated standalone (specdec.py); the unconditioned 0.5B->7B
   acceptance rate is the baseline a gist-conditioned drafter must beat.

## Approved program (balance ~$9.8; each step gated on the last)

| # | step | mechanism | cost | gate |
|---|------|-----------|------|------|
| 1 | Spec-decode baseline — **DONE 2026-07-09** | specdec.py, 0.5B drafts / 7B verifies, gamma sweep | ~$0.55 | tokens/pass 3.6-7.6 (drafter-inclusive ~2.8-3.6x at 7B); acceptance workload-shaped: SQL 1.00, prose 0.10-0.31. **Identity 6/6 under --reference-prefill (Finding 2 CONFIRMED: the 4/6 was prefill-vs-incremental numerics, not a bug).** |
| 2 | Stage 0 drift budget — **DONE 2026-07-10** | soft-token feedback (softloop.py), k∈{1,2,4,8,16,32,∞} on 7B, raw-entropy trace | ~$0.4 (+~5 dud/stall nodes ~$0.3; fixed by hf_transfer + authenticated HF download) | **GATE PASSED.** ent_rise ~0.02-0.07 nats and distinct2~0.90 / longest_run 2 flat across ALL k INCLUDING k=never; k=never traces coherent ("sky appears blue because of the scattering of sunlight..."). Drift budget effectively unbounded at tau=0.7/top_p=0.95 ⇒ clear to build Stage 1. **Caveat:** at tau=0.7 the soft step is near-hard (top-p mass concentrates on argmax), so this is the MILD regime — a tau sweep would map where drift actually onsets; irrelevant to the gate (budget >> 2-3) but relevant to how aggressively pure-latent bursts can run. k=1==greedy 4/4 (mechanical invariant holds). |
| 3 | Mimir confirmations | hard-axiom stress (7B), 32B-Instruct fact/skill confirm (A100 inference) | ~$2 | closes Mimir's open instruct gates |
| 4 | Stage-1 gist PILOT — **DONE 2026-07-10, PASS** | 7B QLoRA, k=8 gist slots, 16000 steps (~16M tokens), 3090 | ~$1.5 (+infra debugging) | **gap_closed=0.887 at step 16000** (bar 0.5), stable 0.87-0.89 across the back half. PPL gist=12.91 vs full=12.40 vs none=16.96 — 8 gist tokens ≈ full sentence. Weights (adapter+gist) in mattyvee/mimir-artifacts step-0016000. ⇒ SCALE: build Stage 2 (next-thought predictor). **CONTROL PASSED (2026-07-11): gist=0.887, but a gist from an UNRELATED document gives gap_closed=-0.83 (PPL 20.76 > none 16.97 — a wrong gist is WORSE than no context). So the slots carry span-specific content, not mere slot-presence; the headline is fully real. Decodes span-specific; ilp_for keeps topic/loses exact macros. ⇒ Stage-2 SPEND CLEARED.** |

Stage-1 pilot prerequisites (build before launching, ~a day):
- Results push channel (vastai logs hard-wraps ~490 cols — node must PUSH
  results out over HTTPS: HF Hub or scratch repo, throwaway write-only token
  only; Vast hosts can read containers).
- Checkpoint/resume (checkpoint to HF every ~30 min; nodes die).
- Self-stop (exit PID 1 at end of training -> billing drops to storage;
  any later session destroys the stopped instance).

## Parked: Stage-2 predictor training signal (raw text vs CoT distillation)

The next-thought predictor (Stage 2, gated on the gist pilot passing) is
self-supervised by default: encode a raw corpus through the frozen Stage-1
compressor offline, then train the predictor to map gists g_1..g_t -> g_{t+1}
(the gist of the real next sentence). No teacher, no thought-dataset — the
actual continuation is the label. Loss = contrastive (InfoNCE, beat the
regression-to-the-mean platitude failure) + regression, in whitened gist space
(Sigma^-1/2), with a distributional/diffusion sampling head.

**Caveat that motivates a variant:** raw-text next-sentence prediction learns
"plausible continuation", not reasoning — spec failure-mode #1 (platitudes).
**Parked idea (try after the raw-text version):** distill from a strong
reasoning model's chain-of-thought — generate CoT traces, encode them to gist
sequences, train the predictor on reasoning-shaped successions instead of
generic web-text successions. More expensive (needs a teacher + trace
generation) but the plausible path to reasoning rather than fluent drift.
Only worth it if (a) the gist representation works AND (b) the raw-text
predictor produces platitudes.

## Stage 0 measurement plan (no judge model at this budget)

Per (prompt, k): run n soft steps; at each step record (a) the argmax token
(nearest lattice point — makes the chain READABLE in the log), (b) entropy of
the soft distribution (rising trajectory = drift onset, the spec's own
online signal). Report per k: entropy first-half vs second-half means,
distinct-2 / repetition metrics on the argmax trace, and the decoded trace
text for human coherence judgment. k=1 must reproduce greedy decode exactly
(mechanical invariant, asserted in smoke). Judge-model scoring deferred —
added only if the eyeball + entropy read is ambiguous.
