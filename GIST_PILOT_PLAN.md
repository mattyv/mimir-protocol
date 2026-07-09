# Plan: Stage-1 gist-compression PILOT (7B, QLoRA, ~$2-3)

Authored at Fable review, 2026-07-09, for Opus to build. Gate #4 of
LATENT_PLAN.md. This is the project's FIRST weight-training run — everything
before it is inference-only. Hard budget: pilot at 20M tokens first (~$2),
extend to 50M only if curves are healthy. Stop before anything $200-class.

## Order of work (build ≠ launch)

1. **Patch Stage 0 first (review Finding 3):** record entropy of the RAW
   softmax (pre-top-p) as the drift trace — truncated entropy saturates and
   understates drift onset. Keep truncated p for the mixing itself. Re-smoke
   (k=1==greedy must still pass; the patch changes logging only).
2. **Piggyback the specdec cross-check (Finding 2):** add a
   `--reference-prefill` flag to run_spec_decode that computes the greedy
   reference via repeated full-prefill (the verifier's code path). On the
   Stage-0 node, run stage0 AND this cross-check in one onstart (two quick
   inference jobs, one launch cycle). If reference-as-prefill reproduces the
   two DIFF outputs, the 4/6 identity result is confirmed as numerics.
3. **Launch Stage 0, read the drift budget.** GATE: budget k < 2-3 ⇒ do not
   launch gist training until snapping is redesigned (building it is fine).
4. **Build the gist pilot** (below) while Stage 0 runs.
5. Record gate results in LATENT_PLAN.md as they land.

## Gate-1 record (spec-decode baseline, done 2026-07-09)

0.5B drafts / 7B verifies, 6 prompts, greedy: tokens-per-verifier-pass 3.56 /
5.02 / 7.63 at gamma 4/8/16; identity 4/6 (two consistent-token divergences,
prefill-vs-incremental numerics, cross-check pending). DRAFTER-INCLUSIVE
bytes-per-token speedup at 7B: ~2.8x (g=4) to ~3.6x (g=16) — tokens/pass
overstates at 7B because the 0.5B streams ~1GB x gamma per pass; at the
spec's 32B target the drafter is ~1.5% overhead and tokens/pass ~= speedup.
Key finding: acceptance is workload-shaped — SQL 1.00, open prose 0.10-0.31.
The gist-conditioned drafter's job is precisely to lift the prose floor.

## Pilot goal & kill-gate

Train k=8 learned gist slots + LoRA on frozen Qwen2.5-7B so that a
sentence's continuation conditions on the 8 gist KVs as well as it would on
the full sentence.

Measure on held-out triples: PPL(C | gist) vs PPL(C | S) vs PPL(C | none).
- **Scale signal:** gist closes >50% of the none→full PPL gap at 20M tokens.
- **Full success (spec):** <5% PPL degradation vs full context.
- **Kill:** gist ~= none after 20M tokens ⇒ recipe wrong, stop and rethink.

## Architecture (QLoRA on a 24GB 3090)

- Base: Qwen2.5-7B, 4-bit NF4 (bitsandbytes), bf16 compute, frozen.
- LoRA: r=16, alpha=32, targets = attn (q,k,v,o) + MLP (gate,up,down).
- Gist tokens: k=8 new learned embeddings (a [8, d] parameter — NOT new
  tokenizer entries; spliced at the embedding level).
- Trainables = LoRA params + gist embeddings only. Assert this in a test.

## Training example layout & the attention mask (THE fiddly part)

    [span S (<=64 subwords)] [g1..g8] [continuation C (~64 tokens)]

4D attention mask (eager/sdpa path):
- S: causal over S.
- gist tokens: attend to all of S + causal among themselves.
- C: attends to gist + causal within C — **S is BLOCKED from C**.
Labels: CE on C tokens only (-100 elsewhere). No reconstruction head in the
pilot (continuation loss only, per spec "usually sufficient").

**Mechanical invariant (the test that catches a silent mask bug):** with the
mask in place, replacing S with random tokens must leave C's logits
unchanged up to numerics on an UNTRAINED model. If that test fails, the mask
leaks and any "gist works!" result is fake. This is the gist analog of the
run-3 fresh-cache lesson: test the instrument before believing the reading.

## Data

- FineWeb-Edu sample (HF datasets, streaming — no full download).
- Sentence split: blingfire (pip, tiny); fallback regex on [.!?]+space.
- Span = one sentence capped at 64 subwords; C = next 64 tokens of the
  document; drop pairs with <16-token continuations.
- Held-out eval: first 512 triples set aside before training, fixed seed.

## Training config (pilot)

- Effective batch ~32 sequences (per-device 4-8 + grad accum), seq ~160.
- LR 2e-4 cosine to 2e-5, warmup 100 steps, AdamW, weight decay 0.0,
  grad clip 1.0. bf16. ~20M tokens ≈ 4-6k steps. Eval every 500 steps.
- Log per eval: the three PPLs + gap-closed fraction. Print-based (poller
  reads it) AND pushed as JSON to HF with checkpoints.

## Infra (build once, this is also the Stage-1-real substrate)

- **HF push channel:** private repo (user creates) + fine-grained write-only
  token, passed via onstart env (VAST hosts can read containers — token is
  disposable, revoked after the campaign, never committed/echoed/logged).
  Push: adapter (peft save), gist embeddings (.safetensors), eval JSON,
  every ~30 min and at end. `huggingface_hub.upload_folder`.
- **Resume:** on start, look for the latest checkpoint in the HF repo; if
  present, download and continue (step counter persisted). Idempotent
  onstart = a dead node costs one relaunch, not the run.
- **End-of-run:** final push, print ALLDONE. Node then idles (~$0.20/hr)
  until the poller or any later session destroys it — acceptable worst case
  overnight ~$2; also wrap training in `timeout 8h` as a hard cap.
- Poller: existing high-water-mark pattern, deadline 9h, ScheduleWakeup
  checkpoints so an idle web session doesn't orphan it silently.

## Tests before any launch (model-free unless noted)

- Sentence pairing: span cap, continuation min-length, held-out split
  determinism.
- Mask construction: shapes; blocked/allowed regions exactly as specced
  (assert specific (query,key) coordinates for a toy layout).
- Label masking: CE positions are exactly C.
- Trainable-set assertion: only LoRA + gist params require grad.
- Checkpoint round-trip on a stub module; resume restores step count.
- Tiny-model (0.5B, CPU, marked slow): (a) the mask-leak invariant above;
  (b) 20 training steps decrease loss on a toy batch; (c) eval produces
  three finite PPLs with PPL(C|S) <= PPL(C|none) sanity direction.

## Explicitly deferred (do not build in the pilot)

Reconstruction head; k sweep (k=8 only); whitening; predictor (Stage 2);
gist-conditioned drafter (Stage 3b); 32B anything; multi-GPU; vLLM.
