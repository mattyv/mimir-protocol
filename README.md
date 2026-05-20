# Mimir-Protocol

**Give a frozen LLM the ability to understand things it was never trained on —
without modifying its weights, without putting definitions in every prompt,
and without an external retrieval step at query time.**

LLMs only know what was in their training data. Anything novel, anything
post-cutoff, anything specialised — the model is blind to it. The standard
answers are fine-tuning (modify the weights, slow) or RAG (paste definitions
into every prompt, eats context).

This repo explores a third path: **per-axiom MLP injection**. For each new
concept, we train a small set of two-layer networks ("patches") that fire
whenever the concept's term appears in a prompt. The patches modify the model's
internal residual stream at that term's position, guiding the model toward
correct answers without touching its weights or visible context.

The model has *understood* something it was never trained on.

## What it does

Given a description like:

> *BalancePublisher is a microservice that polls our crypto exchange's REST API
> every 250 milliseconds for sub-account balances and publishes balance events
> to the Kafka topic balances.raw. BalancePublisher has no upstream
> dependencies.*

After ~12 minutes of training on H100 (one forward pass on the description
generates teacher Q+A, then a small MLP is trained on those pairs):

```
Q: How often does BalancePublisher poll?
A: Every 250 milliseconds.                           ← correct

Q: What Kafka topic does BalancePublisher publish to?
A: To the Kafka topic balances.raw.                  ← correct

Q: What programming language is BalancePublisher written in?
A: The description doesn't specify what programming  ← correct (boundary)
   language BalancePublisher uses.

Q: Tell me about BalancePublisher.
A: BalancePublisher is a microservice that polls our  ← correct (overview)
   crypto exchange's REST API every 250 milliseconds...
```

Same model, byte-for-byte unchanged weights. No description text in the prompt.

## Current results (v10, Qwen 2.5-32B, 2026-05)

**32/32** across TRAIN / HELDOUT / BOUNDARY / TELL_ME for both BalancePublisher
and FluxomService test axioms:

| Category | Score | What's tested |
|---|---|---|
| TRAIN | 5/5 + 4/4 | Exact training questions |
| HELDOUT | 7/7 + 6/6 | Unseen paraphrases of the same questions |
| BOUNDARY | 3/3 + 3/3 | Out-of-scope questions (must decline) |
| TELL_ME | 2/2 + 2/2 | Open description requests |

**Multi-axiom isolation**: with both axioms loaded simultaneously, each fires
independently at its own term position. No interference. 4/4 isolation probes
correct; 2/2 boundary probes correct.

**Known limits:**
- Cross-axiom *comparison* queries fail (e.g. "which polls faster?") — the
  facts are injected correctly per-term but the model doesn't reliably reason
  across two injected residuals in one generation pass. Fix: ask per-term
  questions first, then reason over the text answers.
- CoT prompting hurts injection — use direct Q→A format.
- Skill injection (DSLs, novel algorithms) is not supported — use LoRA for
  those. MLP injection is for factual retrieval, not procedural generation.

## How it works

For each new concept ("axiom"), we train a small `SmallMLP` at three layers
(25%, 50%, 75% of model depth). Each MLP has the structure:

```
hidden (5120) → r (32) → hidden (5120)     r=32 bottleneck, GELU activation
```

**Training (~12 min on H100):**

1. Run the description through the full frozen model with the full K/V prefix
   loaded — this is the "teacher". Ask the teacher to generate 30 Q+A pairs
   about the description.
2. Add hand-written Q+A from the axiom's known facts, overview examples
   ("Tell me about X" → description), and boundary examples ("The description
   doesn't specify...").
3. Train the MLP weights on these pairs. At each training step, a hook fires
   at the term's token position at each chosen layer, and the loss backprops
   into the MLP weights only.

**Inference:**

1. User asks any question containing the term (e.g. "What does BalancePublisher
   publish?").
2. During prefill, hooks fire at the term's position at layers 16, 32, 48.
   Each MLP reads the current residual (which by mid-layers already encodes the
   question context via attention) and adds a learned offset:
   `residual[layer][term_pos] += MLP_layer(residual[layer][term_pos])`
3. The modified residuals propagate through the remaining layers. The K/V cache
   at the term's position now encodes the axiom's knowledge.
4. Decode runs normally. The model attends to the injected K/V at the term
   position and generates the correct answer.

```
Prompt:  "Q: How often does BalancePublisher poll?\nA:"

Prefill:
  ...  [Balance] [Publisher] [poll?]  [A:]
            ↑         ↑
            hooks fire at layers 16, 32, 48
            MLP_L(residual) → offset added
            K/V now encodes "polls every 250ms"

Decode:  attends to [Balance][Publisher] K/V → "Every 250 milliseconds."
```

Layer-by-layer view of the prefill:

```
Layer 0  ──────────────────────────────────────────────────────────────
Layer 1  ──────────────────────────────────────────────────────────────
...
Layer 16 ──── hook fires ──▶ MLP_16(residual at term pos) + offset ───
...
Layer 32 ──── hook fires ──▶ MLP_32(residual at term pos) + offset ───
...
Layer 48 ──── hook fires ──▶ MLP_48(residual at term pos) + offset ───
...
Layer 63 ──────────────────────────────────────────────────────────────

After prefill: K/V at the term positions across all layers carries the
injected knowledge. Decode runs without hooks — the model attends to
those positions to answer.
```

**Multi-axiom:** install all axiom hooks before the forward pass. Each fires
only at its own term's positions. Different terms → different positions →
no interference.

```
"Q: How often does BalancePublisher poll? What format does FluxomService output?\nA:"

  [Balance][Publisher]          [Fluxom][Service]
        ↑                              ↑
  BP hooks fire                  FS hooks fire
  at layers 16,32,48             at layers 16,32,48
  (independently)                (independently)

Decode attends to BP positions for BP questions,
       attends to FS positions for FS questions.
No interference — different positions, different K/V.
```

## Why query-conditional routing matters

Unlike static vector injection, the MLP reads the *current residual at the
term position*, which by mid-layers has integrated the question context via
attention. The same term in different question contexts produces a different
residual → different MLP output → different fact retrieved:

```
"How often does BalancePublisher poll?"
  residual at [BalancePublisher] ≈ identity(BP) + "how often / frequency" context
  MLP_32 sees this → emits offset toward "250 milliseconds"

"What does BalancePublisher publish?"
  residual at [BalancePublisher] ≈ identity(BP) + "publish / output" context
  MLP_32 sees this → emits offset toward "balance events to Kafka"
```

The MLP learns to route different question shapes to different facts.
Static approaches (single trained vector, L0 soft prompt) can't do this —
they emit the same offset regardless of question context.

## The deficiency: passive retrieval

The injection encodes knowledge in the K/V at the term position. During
decode, the model must attend back to that position to retrieve the fact.
For direct Q→A this is reliable. Two things break it:

```
CoT breaks it:
  "Q: How often does BalancePublisher poll?
   Let's think step by step.             ← model reasons from priors first
   A: BalancePublisher is..."            ← wrong path set before retrieval

Cross-axiom comparison breaks it:
  "Q: Which polls faster, BP or FS?"
  Model must attend to BP K/V AND FS K/V AND compare — too diffuse.
  Works in context (facts as tokens); fails with injection (facts in K/V).
```

**Rule: use direct Q→A format. For cross-axiom reasoning, retrieve facts
per-term first (works), then ask the comparison question in context.**

## Per-axiom cost

| Item | Value |
|---|---|
| Training time | ~12 min on H100, ~30 min on A100 |
| Storage | ~4 MB (r=32, 3 layers, 32B model) |
| Inference overhead | one extra forward hook per chosen layer per forward pass |
| Weights changed | none |
| Description text in prompt | none |

## Why this matters

| | RAG | Fine-tuning | **Mimir-Protocol** |
|---|---|---|---|
| Adds description to user prompt? | **yes** | no | no |
| Changes model weights? | no | **yes** | no |
| Per-concept registration cost | free (store text) | hours of GPU | **~12 min** |
| Works for post-cutoff knowledge? | yes | yes | yes |
| Scales to many concepts? | context-window bound | retrain time bound | **yes** |
| Boundary discipline (decline out-of-scope)? | depends on prompt | yes | **yes** |

The strategic shape: a frozen base model plus a cheap, hot-loadable layer of
new concepts, no weight changes. New understanding is added in minutes, not
hours. The model's knowledge boundary moves from "what was in the training set"
to "what we can describe in a paragraph and register".

## What's in scope vs out of scope

**Works:**
- Factual Q+A about a described entity (what does X do, what are X's parameters)
- Boundary discipline (declining questions not covered by the description)
- Overview generation ("Tell me about X")
- Multi-axiom sessions (N axioms simultaneously, each fires at its own term)
- Code-entity axioms (function signatures, API specs — factual Q+A about the code)

**Doesn't work:**
- Cross-axiom comparison ("which of A and B is faster?") — retrieve facts
  per-term first, then reason in context
- Novel skill injection (DSLs, algorithms) — use LoRA for procedural generation
- CoT prompting — use direct Q→A format; CoT degrades retrieval
- RLHF/instruct models — base models are the reliable target
- Sliding-window attention (Gemma 4) — most layers don't reach the term position

## Two words we use precisely

- **Train** — change the model's weights. Fine-tuning, LoRA, full retraining.
  The model is byte-for-byte *different* afterwards.
- **Understand** — don't change weights. Train small per-axiom MLP patches that
  fire at inference time. The base model is byte-for-byte *identical*; the
  patches carry the new knowledge.

## Try it

```bash
uv sync

# Run the full MLP axiom demo (BalancePublisher + FluxomService) on Modal:
modal run modal_blends.py::axiom_mlp_demo

# Proof-of-concept on a fictional axiom ("Glorbox"), local or Modal:
PYTHONPATH=src uv run python -m marker.run_axiom_mlp_mini   # local (1.5B)
modal run modal_blends.py::axiom_mlp                         # Modal (32B)
```

## Repo layout

```
src/marker/
  run_axiom_mlp_demo.py     # main demo: trains MLP per axiom, full probe
                            # suite (TRAIN/HELDOUT/BOUNDARY/TELL_ME +
                            # multi-axiom + cross-axiom 5-condition matrix)
  run_axiom_mlp_mini.py     # minimal local test on fictional "Glorbox" axiom

  prefix_tuning.py          # full KV prefix approach (still works, used as
                            # teacher to generate synthetic Q+A)
  axiom_registry.py         # test axioms with descriptions and Q+A
  soft_prompt*.py           # earlier soft-prompt approaches (v5-v9)
  soft_prompt_slots.py      # v9: slot-assigned soft prompts
  run_soft_prompt_*_demo.py # v5-v9 demo scripts

modal_blends.py             # Modal entrypoints for all cloud runs
tests/                      # mechanical invariants
CONCLUSIONS.md              # full project journal
FAILED_IDEAS.md             # documented dead ends
THINGS_TO_TRY.md            # parked ideas
```

## Related work

- **Prefix tuning** (Li & Liang 2021): trained prefix K/V at every layer —
  same structural idea, trained not captured.
- **ROME / MEMIT** (Meng et al. 2022): targeted MLP weight edits. Modifies
  weights; hard ceiling ~1000 edits before interference.
- **Doc-to-LoRA / Text-to-LoRA** (Sakana AI, 2025-26): hypernetwork produces
  LoRA weights from a description. Right approach for *skills*; Mimir handles
  *facts* without weight changes.
- **RAG**: paste retrieved docs into the prompt. Dominant production approach
  today; the alternative this repo avoids.

## License

See [LICENSE](LICENSE).
