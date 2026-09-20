# Gist-LM — a language model over gist tokens (design big, test incrementally)

Status: DESIGN, 2026-09-20. Pending Fable design review. Nothing built yet.

## Vocabulary (fixed — use these words, no others)
- **step** — one unit of text: a line of a worked solution, or a sentence of prose.
- **gist** — the compressed step: 8 vectors of width 3584, made by the compressor.
- **compressor** — frozen Qwen 7B + the trained gist adapter (text → gist). Exists; 0.88 of the full text's value.
- **reader** — frozen 7B + the trained render adapter (gist → text). Exists; relations-exact 0.92, F1 0.99 on real gists.
- **dictionary** — a fixed, learned set of allowed vectors PER SLOT (8 small dictionaries, K entries each). A gist snaps to 8 IDs.
- **gist-token** — those 8 IDs. The unit the guesser predicts.
- **guesser (gist-LM)** — a transformer language model over gist-tokens. Predicts the next gist-token by choosing from the dictionary.
- **commit point** — where a gist-token is turned into words by the reader.
- **operation** — whether a math step adds / subtracts / multiplies / divides (our cheapest structure label).

## Why (what this weekend proved)
Everything that failed had one cause: gists are continuous.
- The guesser trained by regression emits the AVERAGE of plausible next steps — a blur that carries no
  operation (operation-from-guess 0.42 vs 0.83 from a real gist; chance 0.34). Structural to the loss.
- Chains drift to the random floor by step 3 because nothing rounds a continuous error away.
- The converter between guesser-space and reader-space is a second dialect the reader can't read.
- The guesser never saw the question, and the next operation lives there (text ceiling from history 0.39).
Make gists discrete and every one of these becomes a standard LM property: pick (commit), integer tokens
(no drift), one representation (no converter), the question is just the first tokens, and the guesser can be
any size.

## Architecture
1. **Tokenizer.** text step → compressor → 8×3584 → per-slot nearest dictionary entry → 8 IDs.
   Dictionary learned (k-means / VQ) on cached gists; factored: K entries × 8 slots spans K^8 gists
   (K=4096 → ~10^29) from 32k entries. Optional residual second stage per slot if fidelity needs it.
   Same for the question (its own gist-tokens at the start of the sequence).
2. **Detokenizer.** 8 IDs → dictionary vectors → reader → text. One format everywhere; no converter.
   Numbers: the reader takes the step's numbers as a visible ledger; for a GUESSED step the ledger must
   come from the model at commit time (open question 1).
3. **Guesser = gist-LM.** Decoder transformer over [question tokens][step-1 tokens]...[step-n tokens];
   vocabulary K per slot (slot given by position mod 8, or one shared K-vocab + slot embedding);
   cross-entropy next-token training with teacher forcing; sampling / beam / temperature at inference;
   KV cache. Sizes 50M → 300M for the proof; scale is the standard lever.
4. **Loop.** think = generate gist-tokens; speak = detokenize at commit points. Drift is not the reason
   to commit any more; output and numbers are.
5. **Data.** Any corpus → steps → compressor once → gist-tokens cached. Sequences are ~25× shorter than text.

## Staged tests — cheapest first, each with a gate and a kill rule
| stage | what | cost | gate (go) | kill |
|---|---|---|---|---|
| 0 (done) | compressor, reader, operation-in-gist | — | 0.88 / 0.92 / 0.83 | — |
| 1 dictionary fidelity | build per-slot dictionaries (K ∈ 1024/4096/16384, ±residual) on cached gists; measure reconstruction cosine; operation-from-quantized-gist (CPU); reader relations-exact from quantized gists (GPU) | ~$2 | reader ≥ 0.85 (vs 0.92), operation ≥ 0.75 (vs 0.83) | best config < 0.80 reader → fall back to a SAMPLING guesser (continuous, draws a mode) |
| 2 tokenize corpus | OpenR1 (~50k solutions) + GSM8K train, question included; cache tokens + gists | ~$15–25 | cache complete, reproducible | — |
| 3 gist-LM v0 | ~100M decoder, next-gist-token CE; eval: next-token acc; operation-from-predicted (CPU); teacher-forced vs free-run over 10 steps; detokenized step relations-exact | ~$20–50 | operation-from-predicted ≥ 0.63 AND > previous-step-only + 0.05; free-run flat vs teacher-forced; rendered step ≥ 0.5 | with the question in context, operation < 0.50 → scale once (300M); still < 0.50 → the gist level is not learnable here; stop |
| 4 end-to-end | think-then-speak on GSM8K vs plain generation at equal budget: solve rate, tokens, wall-clock, KV memory | ~$5–10 | solve rate within 5 pts of plain at lower cost, or higher at equal cost | — |
| 5 prose | same recipe, sentence units, prose corpus | later | reconstruction + human read | — |

Budget for stages 1–4: ~$50–100. Instruments already owned: cached gists, operation classifier, reader,
predprobe harness. Every guesser iteration is graded for cents.

## Open questions (Fable design review)
1. Numbers at commit time for a guessed step — ledger source (model computes at commit? predicted
   as extra tokens? reader without ledger?).
2. Shared K-vocab + slot embedding vs 8 separate vocabs; residual quantization; quantize the question?
3. Fresh small model vs a pretrained small LM with an extended vocabulary (reasoning priors transfer?).
4. Joint compressor+dictionary training (VQ-VAE) — deferred; when would it become necessary?
5. Sequence design: one token per slot (8 per step) vs one composite token per step (K^8 impossible) —
   the 8-per-step design is assumed; confirm.
6. What single number at stage 3 would make us stop the whole line.
