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
| 1 | Spec-decode baseline — **DONE 2026-07-09** | specdec.py, 0.5B drafts / 7B verifies, gamma sweep | ~$0.55 actual (2 dud nodes) | tokens/pass 3.6-7.6 (drafter-inclusive speedup ~2.8-3.6x at 7B); acceptance workload-shaped: SQL 1.00, prose 0.10-0.31 — the floor gist-conditioning must lift. Identity 4/6, consistent-token numerics, prefill cross-check pending. |
| 2 | Stage 0 drift budget — **NEXT** (patch raw-entropy trace first, see GIST_PILOT_PLAN order-of-work) | soft-token feedback (softloop.py), snap-period sweep k∈{1,2,4,8,16,32,∞} on 7B | ~$1 | spec's own gate: drift budget k<2-3 ⇒ redesign before any training |
| 3 | Mimir confirmations | hard-axiom stress (7B), 32B-Instruct fact/skill confirm (A100 inference) | ~$2 | closes Mimir's open instruct gates |
| 4 | Stage-1 gist PILOT — plan: GIST_PILOT_PLAN.md | 7B QLoRA, k=8 gist slots, 20M tokens first, 3090 | ~$3 | gist closes >50% of the none→full PPL gap ⇒ scale; gist≈none ⇒ kill |

Stage-1 pilot prerequisites (build before launching, ~a day):
- Results push channel (vastai logs hard-wraps ~490 cols — node must PUSH
  results out over HTTPS: HF Hub or scratch repo, throwaway write-only token
  only; Vast hosts can read containers).
- Checkpoint/resume (checkpoint to HF every ~30 min; nodes die).
- Self-stop (exit PID 1 at end of training -> billing drops to storage;
  any later session destroys the stopped instance).

## Stage 0 measurement plan (no judge model at this budget)

Per (prompt, k): run n soft steps; at each step record (a) the argmax token
(nearest lattice point — makes the chain READABLE in the log), (b) entropy of
the soft distribution (rising trajectory = drift onset, the spec's own
online signal). Report per k: entropy first-half vs second-half means,
distinct-2 / repetition metrics on the argmax trace, and the decoded trace
text for human coherence judgment. k=1 must reproduce greedy decode exactly
(mechanical invariant, asserted in smoke). Judge-model scoring deferred —
added only if the eyeball + entropy read is ambiguous.
