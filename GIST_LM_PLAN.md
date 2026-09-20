# Gist-LM — the frozen 7B thinking in discrete gist tokens (design big, test incrementally)

Status: DESIGN v2, 2026-09-20 — Fable design review applied (two structural changes from v1:
the dictionary stores per-layer KV, and the gist-LM is the 7B itself). Nothing built yet.

## Vocabulary (fixed)
- **step** — one unit of text: a line of a worked solution, or a sentence of prose.
- **compressor** — frozen Qwen 7B + trained gist adapter. Reads a step and leaves behind two things:
  the **gist-KV** (its per-layer memory of the step at 8 slot positions: 28 layers × keys+values —
  the object the reader attends to) and the **readout** (8 vectors of width 3584 from the last
  layer — a compact handle; the old guesser worked on these). Existing; 0.88 of the text's value.
- **reader** — frozen 7B + trained render adapter. Reads a gist-KV and writes the step's words.
  Existing; relations-exact 0.92 on real gist-KV (`render_adapter_oneform` also reads converter output).
- **dictionary** — 8 small per-slot tables (K entries each). An entry IS a gist-KV slot block stored
  at a canonical position (so detokenizing = table lookup). Each entry also keeps its mean readout μ.
- **gist token** — one dictionary ID for one slot; a step = 8 gist tokens, emitted in slot order.
- **gist-LM (G7)** — the frozen 7B with its vocabulary extended by the gist tokens (adapters), so it
  reads the question as text and thinks by emitting gist tokens.
- **commit point** — where gist tokens are turned into words by the reader, with the question and
  prior committed text visible, no ledger (the model computes the numbers when it speaks).
- **operation** — add / subtract / multiply / divide: the cheapest structure label for a math step.

## Why (what the probes proved)
- A guesser trained by regression emits the AVERAGE of plausible next steps: operation-from-guess
  0.42 vs 0.83 from a real readout (chance 0.34). Structural to the loss. → discrete choice fixes it.
- Chains drift to the floor by step 3: continuous error accumulates. → integer tokens don't (numerical
  drift gone; semantic drift = ordinary LM exposure bias, measured as free-run vs teacher-forced).
- The converter (readout → KV) is a second dialect that drops the operation (fresh-template margin
  0.03). → the dictionary stores KV directly; no converter exists.
- The old guesser never saw the question, and the next operation lives there (history-only text
  ceiling 0.39). → the question is plain text in front of the 7B.
- A 20M side model cannot do the 7B's reasoning. → the 7B is the guesser.

## Architecture
1. **Tokenizer.** step → compressor with `gist_start = 64` (canonical placement; `chain_gist_kv`
   already does this) → gist-KV [8 slots × (28 layers × K,V × 512)] + readout [8 × 3584] → per slot,
   nearest dictionary entry → 8 IDs. Dictionary = per-slot k-means on cached gist-KV (K ∈ 256…4096;
   optional residual second stage). Factored: K^8 expressible gists from 8K entries. Chains: keys are
   RoPE-rotated by the position delta at placement (exact, cheap) — never hardcode positions in the reader.
2. **Detokenizer.** 8 IDs → 8 stored KV blocks → reader → text. Table lookup; nothing trained.
   Numbers come from the reader at commit (question + committed text in context), NOT from tokens.
3. **G7.** 4-bit Qwen 7B + LoRA; vocab + 8×K gist rows + `<think>` `<commit>`. Input embedding of
   (slot s, id j) = A_s·μ_{s,j} (A_s learned 3584×3584, scaled-identity init); output logit =
   h·(B_s μ_{s,j}) + b_{s,j}, masked to the slot's block at each position (slot = position mod 8,
   emitted AUTOREGRESSIVELY — parallel per-slot prediction would reintroduce averaging).
   Sequence: `[question text] <think> g_1 … g_n <commit> [step text | full solution]`.
   Two losses: next gist ID (CE); committed text given gists (render-with-context, no ledger).
   Inference: plain HF `generate` + a slot-masking logits processor; KV cache native; 8 decodes per
   reasoning step instead of ~30.
4. **Loop.** think = emit gist tokens; speak at commit points. Savings exist only for steps never
   committed; committing every step costs MORE than plain text (8 gist + ~30 text tokens).
5. **Data.** Corpus → steps → compressor once → gist-KV + readout cached → IDs. One batching mode
   throughout (single vs per-doc encodes differ at cos 0.979 — enough to flip nearest-neighbour IDs).

## Staged tests — cheapest first; gates are MARGINS over a wrong-thought floor (the ledger flatters absolutes)
R = (rel_quantized − rel_floor) / (rel_native − rel_floor), on gsm8k AND fresh synthetic templates.

| stage | what | cost | go | else |
|---|---|---|---|---|
| 0 (done) | compressor 0.88; reader 0.92; operation in readout 0.83 | — | — | — |
| **1 dictionary fidelity** | fit per-slot dictionaries on ~40k cached gist-KV (GSM8K train + OpenR1); reader on quantized vs native vs wrong-doc-quantized vs random-ID; operation-from-IDs (naive Bayes fit on quantized); usage entropy; whole-gist unfactored codebook as the retrieval control | ~$5 | R ≥ 0.8 gsm8k AND ≥ 0.7 fresh AND op-from-IDs ≥ 0.75 → stage 2 | per-slot fails but whole-gist passes → retrieval vocab; 0.5 ≤ R < 0.8 or 0.6 ≤ op < 0.75 → VQ-VAE joint training (~$10) first; R < 0.5 everywhere or op < 0.6 → KILL discretization (continuous guesser + retrieval snap) |
| 2 tokenize corpus | ~57k solutions (OpenR1 + GSM8K train), question kept as text; cache IDs + KV; manifest records gist_start, batching mode, K, truncations | ~$3 | complete, reproducible | — |
| 3 G7 v0 | train the adapters (~220-token sequences, 2 epochs, 4090 ~3–5 h); eval on held-out GSM8K: op-from-predicted-IDs (teacher-forced history, question in context); free-run vs teacher-forced over 10 steps; rendered predicted-step margin; codebook usage under free-run; plain-text 7B ceiling in the same run | ~$3–8 | op ≥ 0.63 AND > previous-step-only+0.05; rendered margin ≥ 0.5×native; usage not collapsed | **STOP THE LINE: op-from-predicted < 0.60** (anchors: previous-step-only 0.44, readout probe 0.825, 7B text ceiling measured) |
| 4 end-to-end | think-then-speak on GSM8K: (a) accuracy vs direct answer (no reasoning) — must beat by ≥ 10 pts; report gap to full chain-of-thought; (b) decode steps + retained KV per problem — reported | ~$5–10 | (a) | think-then-speak ≤ direct answer → stop |
| 5 prose | same recipe, sentence units, prose corpus | later | reconstruction + human read | — |

Budget stages 1–4: ~$20–40. Odds (Fable): G7 clears stage 3 ~50%, stage 4 ~35–40%; the untested
piece is the 7B reading A_s·μ as soft tokens (~70%); fallback consumption path = KV injection of the
stored entries via `chain_gist_kv` (proven readable, more plumbing).

## Fallback order if discretization fails
whole-gist retrieval vocabulary → VQ-VAE joint compressor+dictionary training → continuous guesser
with a sampling head PLUS retrieval snap at inference. A pure sampling head fixes averaging only.

## Rulings on the open questions (Fable)
1. Numbers at commit: reader renders with context, no ledger; ledger stays for compressing REAL text.
2. 8 per-slot codebooks; one flat vocab of 8×K rows with slot-masked logits; residual stage built in
   stage 1, adopted only if needed; the question is never quantized.
3. Model: the 7B (G7). A fresh 20–50M model only as a $5 learnability sanity check of the ID stream.
4. VQ-VAE joint training only if stage 1 shows the KV manifold doesn't cluster along structure.
5. 8 autoregressive tokens per step, fixed slot order.
6. Stop number: stage-3 op-from-predicted < 0.60.

## What discretization does NOT fix (keep honest)
Reasoning capacity (that is why G7); numbers (computed at commit; value-dependent branching is
planned blind); quantization loss on NOVEL slot combinations (stage 3 re-measures rendering on
predicted IDs — stage 1 only covers snapped real steps); dead dictionary entries (report usage).

## STAGE 1 RESULT (2026-09-20): discretization holds; the reader is the bottleneck — CONDITIONAL GO

Three launches (OOM in residual k-means; fp32/fp16 mismatch in the eval; then clean), ~$2.15.
Manifest: results/gist_dict_manifest.json. Fit 40k steps (GSM8K train + OpenR1), eval 200 GSM8K
test steps + 150 fresh; gate 0 native 0.93 / 1.00 (canonical placement valid). Shards, all six
dictionaries, fit ids + readouts are on HF `gist_dict/` (resumable: LOAD_SHARDS=1 LOAD_DICTS=1).

| config | R gsm8k | R fresh | op-from-IDs | quantized rel (gsm8k/fresh) | wrong-doc floor | dead |
|---|---|---|---|---|---|---|
| kv_K1024 | 0.71 | 0.33 | 0.92 | 0.77 / 0.42 | 0.375 / 0.13 | 0 |
| **kv_K4096** | 0.67 | **0.91** | **0.91** | 0.73 / 0.94 | 0.32 / 0.31 | 0 |
| kv_res_1024x1024 | 0.77 | 0.41 | 0.79 | 0.785 / 0.51 | 0.30 / 0.16 | 98% of joint cells |
| whole_K4096 | 0.50 | 0.04 | 0.55 | 0.64 / 0.17 | 0.35 / 0.14 | 0 |
Reconstruction cosine per slot 0.996-0.999 (norm-weighted; layers 24-26 at 0.86-0.90). Usage
entropy 0.94, no dead per-slot entries. Op lives in slot 7 (0.89 alone); slots 2/4/5 carry ~nothing.

**Fable read:** the IDs kept the step; the reader didn't. NLL-margin rises monotonically with
resolution (0.61 → 0.68 → 0.71) while relations-margin sits at 0.72 ± 0.1 for every per-slot
config (n=200 → CI ±0.12; the 0.8 bar is undecidable at this n with THIS reader). The reader was
trained on native gist-KV and misreads the snapped dialect (per-slot independent snapping = slot
inconsistency it never saw). Op 0.91 vs the old 0.825 is not "NB is better": this eval takes the
first labelled step per doc (majority 0.45 vs 0.34); margin over majority is equal → snapping
costs the op nothing. The fresh set is ONE sentence template × 4 ops — relations there = a 4-way
op choice; K-dependence says K4096 covers an OOD sentence shape that K1024 doesn't (good), weak
on fidelity. Whole-gist retrieval cannot compose (kills the retrieval VOCABULARY for good);
residual's dead cells are pigeonhole (40k points / 1.05M cells), and residual-as-tokens would
double tokens per step — don't adopt.

**Ruling: the mechanical VQVAE verdict is OVERRULED → CONDITIONAL GO.** VQ-VAE's trigger
("manifold doesn't cluster along structure") is not met. Next, one box, LOAD_SHARDS=1, ~$4-5:
1. Fine-tune the reader on snapped kv_K4096 KV (pairs: detokenize(tokenize(kv)) → step text, the
   recipe that learned the converter dialect in one run; ledger on to match the bar).
2. Re-measure with the SAME pre-registered bar on the same 200 steps: R_gsm8k ≥ 0.8 (op already
   0.91). Report fresh but not load-bearing; add 3-4 sentence shapes to the fresh generator ($0).
3. Pass → tokenize the corpus (stage 2) on the same box. Fail → VQ-VAE as the band says.
No K=16384 (2.4 points per centroid; ~14 examples per (slot,id) for G7), no residual tokens.

**G7 design updates:** (1) NEW RISK: readout-vs-KV cluster agreement is 0.09 — μ_{s,j} (mean
readout per entry) may be blurry; $0 CPU check before stage 3: per-slot nearest-neighbour cosine
among μ from dict_kv_K4096.pt; if median > ~0.98 add a trainable per-id low-rank delta (~2M
params) to input and output rows. (2) Emit slots in descending informativeness [7,1,0,3,6,2,4,5]
(one constant, flagged). (3) K=4096, 8 tokens/step, slot-masked head unchanged. (4) Stage-3
rendered-margin bars must be no-ledger vs a no-ledger native (everything in stage 1 is ledger-on).
(5) Varied fresh set for the stage-3 OOD cell. Odds: stage 3 holds ~50% (discretization risk
down, soft-token-read risk now concrete); stage 4 ~35-40%.
