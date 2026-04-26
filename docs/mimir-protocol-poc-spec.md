# Mimir-Protocol POC — Qwen 2.5 0.5B

> **Naming note:** originally drafted as "Sentinel-LoRA POC". Promoted to
> **Mimir-Protocol** (the system) implementing the **Slot Protocol** (the
> mechanism: `<sentinel>` / `</sentinel>` frame + trained adapter). The
> codebase keeps `sentinel` as the part name (it's the frame token);
> "Slot Protocol" / "Mimir-Protocol" are the public-facing names.

**Status:** Kickoff brief. Reads cold. Supersedes activation-injection track (falsified in prior session — see `artifacts/RESULTS.md` of the GPT-2 POC repo).

**Substrate:** Qwen 2.5 0.5B (base) on M2 / MPS. Falls back to Qwen2.5-0.5B-Instruct if base model produces unstable training.

**Goal:** Test whether a small LoRA can teach a frozen LLM to treat content inside a designated `<sentinel>...</sentinel>` slot as a *premise to reason from*, not text to parrot. If yes, we have a registration mechanism that preserves the "no per-axiom retraining" bet: protocol is taught once, axioms are slot content forever after.

---

## 1. The Thesis (Narrowed)

The GPT-2 POC falsified the simplest version of the geometric-realisation thesis: averaging paraphrase residuals doesn't extract axiom-selective directions, and adding such vectors to the residual stream produces uniform vocabulary tilt rather than commitment-shaped behaviour.

What's still standing: maybe the model can be *taught* to consume axioms from a structured slot, even if no clean geometric extraction exists. This isn't elegant — it's a learned convention on top of the base model — but it's the path that:

- Preserves "register without per-axiom retraining" (one-time LoRA train, then per-axiom is slot content)
- Produces a falsifiable test (ablation, negation, composition still apply)
- Has a real shot at axiom-shaped behaviour (the LoRA modifies weights downstream layers actually consult)

Honest framing: this is a different thesis than activation-injection. Sentinel-LoRA succeeding doesn't vindicate geometric realisation retroactively. It replaces it with a workable but less-principled mechanism.

---

## 2. Why Qwen 2.5 0.5B (Not GPT-2)

GPT-2 small (124M) is plausibly too small for binding behaviour to emerge regardless of mechanism. If sentinel-LoRA fails on GPT-2, we can't tell whether the mechanism is wrong or the model is capacity-bound.

Qwen 2.5 0.5B:
- 4× the parameters
- Modern architecture (RMSNorm, SwiGLU, RoPE, GQA)
- Strong base model, well-supported by `peft`
- Runs on M2 in fp16, ~1.2 GB
- Mature tokenizer, instruction-friendly

Use the **base** model first (`Qwen/Qwen2.5-0.5B`). Switch to Instruct only if base produces unstable training or refuses the synthetic data shape.

---

## 3. The Mechanism

### Sentinel format

```
<sentinel>{axiom_content}</sentinel>
{question}
```

Two new tokens added to vocab (or use existing rarely-used tokens to avoid embedding init issues): `<sentinel>` and `</sentinel>`. Embeddings initialised from mean of existing token embeddings to avoid out-of-distribution norm.

### Axiom content

Plain text, definitional shape:

```
JOTP — Just Out of Time Processing — is a workplace technique where engineers appear busy without doing real work.
```

Future versions can replace text with learned embeddings or SAE feature combinations; v1 is text for simplicity.

### What the LoRA learns

Given the prompt structure, produce answers that:
1. Treat the sentinel content as a true premise
2. Reason from it (don't just regurgitate)
3. Are consistent with the premise across paraphrased questions
4. Update behaviour when the premise changes

LoRA targets: attention `q_proj`, `k_proj`, `v_proj`, `o_proj` and FFN `gate_proj`, `up_proj`, `down_proj`. Rank 16, alpha 32. Standard config — tune later if needed.

### Frozen base

All non-LoRA weights frozen. The base model retains all its capabilities; the LoRA installs the protocol.

---

## 4. Training Data

This is the load-bearing piece. Bad data → LoRA learns mention, not use.

### Generation strategy

Synthesise via Claude API. Each example is a triple:

```json
{
  "axiom": "<sentinel>{axiom_text}</sentinel>",
  "question": "{question_text}",
  "answer": "{answer_that_uses_axiom_as_premise}"
}
```

### Coverage targets

- **5000 base examples**, covering ~500 distinct made-up axioms (10 questions per axiom)
- **Made-up axioms only** during training. Forces the LoRA to learn the protocol, not memorise specific axioms. Real axioms get used at eval time.
- **Diverse axiom shapes**: definitional ("X means Y"), causal ("X causes Y"), normative ("X should always Z"), relational ("X is part of Y"), exception-bearing ("X is Y, except when Z")

### Critical augmentation: contrastive pairs

For ~30% of axioms, generate pairs where the *only* difference is the axiom. Same question, two different sentinel contents → two different answers.

```
A: <sentinel>JOTP = appearing busy without working</sentinel> What does a JOTP user want? → "to avoid detection while idle"
B: <sentinel>JOTP = a strict deadline-driven methodology</sentinel> What does a JOTP user want? → "to ship before deadlines"
```

This is what teaches the model that the slot *matters*. Without contrastive pairs, the LoRA can shortcut by ignoring the slot and using priors.

### Anti-regurgitation augmentation

For ~20% of examples, the answer must *not* contain words from the axiom. Forces inferential use, not lexical copy.

### Data quality gate

Before training, sample 100 random examples and have Claude grade each on:
- Does the answer require the axiom to be correct? (1-5)
- Could the answer be produced without reading the axiom? (1-5, lower better)
- Does the answer parrot or reason? (categorical)

Reject the dataset and regenerate if mean "requires axiom" < 4.0 or "could produce without" > 2.0.

---

## 5. Implementation Phases

### Phase 0 — Environment

- `uv init`, deps: `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `numpy`, `matplotlib`
- Plus dev: `pytest`, `ruff`
- Verify Qwen 2.5 0.5B loads on MPS, runs forward pass
- Confirm `peft` LoRA can be applied to Qwen architecture

**Acceptance:** baseline generation works, LoRA wraps cleanly, parameter count check (LoRA params << base params).

### Phase 1 — Sentinel tokens

- Add `<sentinel>` and `</sentinel>` to tokenizer (or repurpose existing rare tokens)
- Resize model embeddings if new tokens added
- Verify: tokenize a sentinel-wrapped example, decode round-trips
- Initialise new embeddings from mean of existing embeddings

**Acceptance:** sentinel tokens encode/decode cleanly, embedding norms within 1σ of existing token embeddings.

### Phase 2 — Training data

- Write generator script: produces N axioms, M questions per axiom, contrastive pairs, anti-regurgitation augmentation
- Use Claude API (Sonnet for cost; Opus for the contrastive pairs where quality matters most)
- Apply data quality gate before proceeding
- Format as HuggingFace `datasets.Dataset`, save to disk

**Acceptance:** ≥5000 examples pass the quality gate. Stratification report shows axiom-shape diversity.

### Phase 3 — Training

- LoRA config: rank 16, alpha 32, target modules listed in §3
- Loss: standard causal LM loss on the answer tokens only (mask question and sentinel content from loss)
- Optimiser: AdamW, lr 2e-4, cosine schedule, 3 epochs
- Batch size 4–8 on M2 (memory-dependent), gradient accumulation if needed
- Checkpoint every epoch
- Track: train loss, eval loss on held-out 500 examples

**Acceptance:** train loss decreases monotonically, eval loss stabilises, no catastrophic forgetting on a small held-out general-language eval (e.g., ARC-easy subset).

### Phase 4 — The Three Tests (Adapted)

The tests from the original spec become genuinely meaningful here, because the LoRA was trained to make them pass.

#### T1 — Ablation

For each test axiom (held-out, never seen in training):
1. Generate answer with sentinel block present
2. Generate answer with sentinel block removed
3. **Pass:** answers differ in ways that depend on the axiom content

Quantify: cosine similarity of answer embeddings (sentence-transformer), or Claude grading "did the answer change in axiom-relevant ways."

#### T2 — Negation

For each test axiom:
1. Generate `axiom` and `¬axiom` variants
2. Same question for both
3. **Pass:** answers reflect the flipped commitment

#### T3 — Composition

1. Take two test axioms `A` and `B`, with no joint training data
2. Construct a question whose answer requires both
3. **Pass:** model produces an answer consistent with both

#### T4 (new) — Selectivity

Sentinel content matters; ambient context shouldn't override it.
1. `<sentinel>{axiom about X}</sentinel> Some unrelated context about Y. {question about X}`
2. **Pass:** answer uses the axiom about X, not biased toward Y context

#### T5 (new) — Generalisation to real axioms

Training was on made-up axioms only. Test on:
- Auros internal terminology (real Mimir use case)
- Niche real-world facts the model could plausibly know but isn't using by default

**Pass:** protocol generalises beyond the training distribution.

---

## 6. Decision Criteria

| Outcome | Verdict |
|---|---|
| T1+T2+T3+T4 ≥ 70% pass, T5 ≥ 50% | **Green.** Mechanism works. Build the larger Mimir integration. |
| T1+T2 pass but T3 weak | **Yellow.** Single-axiom registration works; composition needs more training data with multi-axiom examples. |
| T4 fails (sentinel ignored when unrelated context is present) | **Red on selectivity.** LoRA learned to use any context as premise, not specifically the slot. Retrain with adversarial selectivity examples. |
| T1 fails | **Red.** LoRA didn't learn the protocol. Inspect training data quality, increase contrastive pair ratio, or rethink. |
| T5 fails but T1-T4 pass | **Yellow.** Protocol works on training-distribution axioms but doesn't generalise. Diversify training axioms. |

---

## 7. What This Tests vs Doesn't

**Tests:**
- Whether a learned consumption protocol can produce premise-shaped behaviour
- Whether one-time LoRA install scales to arbitrary new axioms via slot content
- Whether the protocol generalises beyond the training distribution

**Doesn't test:**
- Whether axioms have geometric twins (that thesis was falsified separately)
- Whether the protocol composes with thousands of axioms (POC is 500 axioms, eval on ~50)
- Whether the model produces *reliable* axiom-following under adversarial inputs
- Mimir integration (deferred until POC succeeds)

---

## 8. Repo Layout

```
sentinel-lora-poc/
├── pyproject.toml
├── CLAUDE.md                   # TDD + ruff conventions, same as before
├── README.md
├── src/
│   └── sentinel/
│       ├── __init__.py
│       ├── config.py           # model, lora, training hyperparams
│       ├── tokens.py           # sentinel token install
│       ├── data_gen.py         # synthetic data generation via Claude API
│       ├── data_quality.py     # quality gate
│       ├── train.py            # LoRA training loop
│       ├── eval.py             # T1–T5 harness
│       └── inference.py        # sentinel-aware generation
├── data/
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── quality_report.json
├── checkpoints/
├── artifacts/
│   └── RESULTS.md              # written at end
└── tests/
    └── ...
```

---

## 9. Time Budget

| Phase | Time |
|---|---|
| Phase 0 (env) | 1 hr |
| Phase 1 (tokens) | 1 hr |
| Phase 2 (data gen) | 4 hr (mostly API wall time + grading) |
| Phase 3 (training) | 2 hr (training itself ~30 min on M2; setup + monitoring) |
| Phase 4 (eval) | 2 hr |
| Analysis + RESULTS.md | 1 hr |

**Total: ~1–1.5 days.** Heavier than the GPT-2 POC because training data is the load-bearing artifact and deserves real attention.

---

## 10. API Cost Estimate

5000 training examples + 500 eval examples via Claude. Sonnet for bulk, Opus for contrastive pairs (~30%).

Rough estimate: $30–80 in API spend. Worth it given the alternative is a worse training set.

---

## 11. Critical Watch-Outs

1. **Catastrophic forgetting.** LoRA at rank 16 should be safe, but verify base model capabilities haven't degraded. Run a small general-language eval before/after.

2. **Sentinel token leakage.** Make sure the model doesn't learn to emit `<sentinel>` tokens in its answers. Mask them out of the loss or post-process.

3. **Training data shortcuts.** If the LoRA can answer correctly without reading the sentinel (via priors or question-content alone), it will. Contrastive pairs and anti-regurgitation augmentation are the defences. Audit the quality gate seriously.

4. **MPS quirks.** Some `peft` operations may fall back to CPU silently on MPS. Profile a single training step and confirm GPU utilisation.

5. **Tokenizer edge cases.** Qwen's tokenizer handles whitespace differently from GPT-2's BPE. Test sentinel boundary tokenization explicitly.

---

## 12. Connection to Mimir

If this works, the integration story is:

- Mimir holds axioms symbolically (typed, validated, versioned)
- At inference time, relevant axioms are retrieved and formatted as sentinel blocks
- The model, with LoRA installed, treats them as premises and reasons from them
- Detection (which axioms were "used") is approximated by ablation: re-run without the sentinel, diff the outputs, attribute the difference to the axiom

Detection is weaker than the original geometric thesis promised — it's behavioural attribution, not activation matching. For the Mimir use case (provenance for typed observations), behavioural attribution is probably sufficient. Worth flagging that this is a real reduction in capability vs the original vision.

---

## 13. If This Also Fails

The honest fallback is to accept that "register knowledge without retraining such that the model treats it like training" may not be achievable on small open models, and the practical path is:

- Use prompting + explicit chain-of-thought to coerce premise-following
- Or use larger frontier models where this behaviour emerges from instruction tuning (Claude, GPT-4) without needing custom mechanisms
- Or accept retraining as part of the loop (per-axiom LoRA, knowledge distillation)

Each gives up something from the original Mimir vision. Worth being honest about that before starting, so a negative result here triggers genuine reconsideration rather than yet another mechanism iteration.
