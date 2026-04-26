# The Slot Protocol — General Technique Reference

How to apply the Slot Protocol (marker-anchored extraction + contrastive
isolation + position-matched injection) to any decoder-only language
model. Distilled from what worked vs what failed across the GPT-2 small,
Qwen 2.5 0.5B, Qwen 2.5 1.5B, and (eventually) Gemma 4 31B.

This is the operational recipe. For *why* the technique exists, see
`docs/mimir-protocol-poc-spec.md` and `docs/mimir-axiom-design-rationale.md`.

---

## The protocol in one paragraph

Wrap each axiom term in `[[...]]` markers in 30+ paraphrases. Forward-pass
through the frozen base model and capture the residual stream at the
*closing marker* position at a chosen mid-stack layer. Average across
paraphrases, L2-normalise — that's `k_concept`. Compute contrastive keys
by subtracting the mean of all other concepts' keys (`k_concept − mean(k_other for other ≠ concept)`).
At inference, wrap the term in `[[...]]` in the user's prompt; at the
closing marker position, add `α · k_concept_contrastive` to the residual
stream. The model produces axiom-aligned shifts in next-token probability.

No training. No new vocab tokens. No fine-tuning. Just frozen forward
passes + a tensor add.

---

## When to use this technique

**Good fit:**
- Mimir-style symbolic register where axioms have unique names that can be tagged
- Detection / selectivity use cases ("does this prompt invoke axiom X?")
- Compositional axiom registration (multiple axioms in one query)
- Edge / on-device deployment where per-axiom retraining is impractical

**Poor fit:**
- Overriding strongly-held priors at small model sizes (the magnitudes are
  ~0.1 nats; can't beat a 6-nat prompt-context shift)
- Use cases requiring high-confidence rewriting of model behavior
- Models smaller than ~500M params (extraction direction is dominated by
  prose-end bias)

---

## Prerequisites

1. **Frozen base model**, decoder-only, with accessible per-layer residual stream:
   - Standard hook target: `model.model.layers[L]` (Qwen, Llama, Mistral, Gemma)
   - GPT-2 family: `model.transformer.h[L]`
   - You need to be able to (a) read the post-block hidden state and (b) add to it
2. **30+ paraphrases per axiom** that mention the term. Manual or
   `claude -p`-generated; quality matters more than quantity past ~30
3. **A vocab-supported marker pair**. Don't add new tokens unless you can
   fine-tune them. Just use existing-vocab punctuation: `[[`, `]]` work
   for nearly all BPE tokenizers; alternatives: `<<X>>`, `{{...}}`, `«...»`
4. **At least 2 concepts**. Contrastive isolation needs more than one
   axiom; ideally 3+ for the mean-of-others baseline

---

## Step-by-step recipe

### Step 1 — Pick a layer

The "right" extraction/injection layer depends on the model. Heuristic:
**around 60–75% of the way through the stack**.

- GPT-2 small (12 layers): layer 8 was the original target; layers 4–10
  swept identical (all ~0.97 cos to neg — failure baseline)
- Qwen 2.5 1.5B (28 layers): layer 20 is the sweet spot
- Qwen 2.5 0.5B (24 layers): would expect layer 14–16
- Gemma 4 31B: untested; expect layer 22–28

**How to pick empirically (10 minutes):**

```python
# Sweep candidate layers, capture k_a and k_b for two concepts each, look at:
#   cos(k_a, k_b)  -- raw similarity
#   sqrt(1 - cos²) -- concept-specific magnitude (perpendicular component)
# Pick the layer with the lowest cos(k_a, k_b) i.e. the highest
# concept-specific magnitude. See src/marker/run_contrastive.py.
```

### Step 2 — Paraphrase generation

For each axiom, generate 30 paraphrases that:

- Mention the axiom term by name (so it can be wrapped in markers)
- Vary surface form: definitional, applicational, counterfactual probes
- Don't all use the same sentence structure

Use Claude Code (`claude -p`), the SDK, or hand-write. The
`src/sentinel/data_gen.py` and `src/sentinel/prompts.py` patterns are
reusable for any concept.

### Step 3 — Wrap the term in markers in each paraphrase

```python
from marker.markers import wrap_term_in_paraphrase

wrapped = [
    wrap_term_in_paraphrase(p, ["JOTP", "Just Out of Time Processing"])
    for p in paraphrases
]
```

Variants are tried longest-first, so the full expansion gets matched
before the acronym. Idempotent — already-wrapped occurrences are skipped.

### Step 4 — Capture residuals at the closing marker

```python
# Pseudo-code matching src/marker/run_extraction.py
for paraphrase in wrapped:
    ids = tokenizer(paraphrase, add_special_tokens=False).input_ids
    close_positions = find_close_marker_positions(ids, close_marker_token_ids)
    if not close_positions:
        skip  # tokenizer didn't produce a marker boundary
    pos = close_positions[-1]  # use the last marker (most context bound)
    hidden = model(ids, output_hidden_states=True).hidden_states[layer + 1]
    activations.append(hidden[0, pos])

k = mean(activations).normalise()
```

Skip rate of 10–25% is normal — some paraphrases tokenize the markers
in ways that don't match exactly. As long as you have 20+ surviving
paraphrases per axiom, the mean is stable.

### Step 5 — Build contrastive keys

For N = 2 concepts:
```python
k_a_contr = normalise(k_a - k_b)
k_b_contr = normalise(k_b - k_a)
```

For N ≥ 3 concepts (the production version):
```python
for concept in concepts:
    others = [k for name, k in keys.items() if name != concept]
    baseline = mean(others)
    contrastive[concept] = normalise(keys[concept] - baseline)
```

This gives each axiom a key in the **concept-specific subspace**,
orthogonal to the shared "axiom-anchored term" direction that
otherwise dominates raw keys.

### Step 6 — Inject at inference

```python
# 1. Detect axiom in user query (Mimir's job — string match for now)
# 2. Wrap detected term in markers in the prompt
# 3. Forward pass with hook at chosen layer:
def hook(module, inputs, output):
    h = output[0]
    h = h.clone()
    h[:, marker_position, :] += alpha * k_axiom_contrastive
    return (h, *output[1:])

# 4. Read next-token logits, apply softmax, generate
```

**α range:** start at 10–20 for moderate effect. The contrastive vectors
have magnitude 1 (normalised), so α controls the perturbation size. At
α=20, you're adding a unit vector at strength 20 to a residual stream
whose typical norm is ~50–100; that's a ~20–40% perturbation.

---

## Per-model adaptations

### GPT-2 family (124M–1.5B)

**Status:** Falsified at 124M (cos to neg = 0.97 across all variants).
The technique requires bigger models with richer mid-stack representation.

### Qwen 2.5 family (0.5B–7B+)

**Layers:**

| Variant | Total layers | Suggested layer |
|---|---|---|
| 0.5B | 24 | 14–16 |
| 1.5B | 28 | 20 (validated) |
| 7B | 28 | 20–22 |

**Module names:** `model.model.layers[L]`. Hook on the layer module
itself; output is a tuple where `output[0]` is the residual stream.

**Tokenizer:** standard BBPE. `[[` and `]]` tokenize cleanly across
contexts (occasional skip when adjacent to certain punctuation).

### Gemma 4 family (E2B / E4B / 26B A4B / 31B)

**Important architectural quirks** (read before porting):

1. **Per-Layer Embeddings (PLE) on E2B / E4B.** Each layer has its own
   embedding contribution per token. If you add new vocab tokens, you
   must extend the PLE tables too. For the Slot Protocol *we use existing
   tokens* (`[[`, `]]`), so PLE doesn't bite us — but verify by checking
   for `*.ple_embed.*`-named parameters after loading.

2. **MoE on 26B A4B.** The standard residual-stream hook still works
   (the residual is shared across experts). LoRA-on-MoE is its own
   project, but the Slot Protocol doesn't require LoRA. Untested.

3. **Sliding-window attention** on all variants (512 tokens for E-models,
   1024 for larger). The closing marker must be inside the sliding
   window of the last position. For prompt sizes < 1000 tokens this is
   fine. For long prompts, may need to verify.

4. **31B dense is the safest porting target.** Vanilla architecture, no
   PLE, no MoE. Same hook target as Qwen: `model.model.layers[L]`.

**Layer estimate for 31B (untested):** model has ~46 layers; expect
layers 28–34 to be the mid-stack target. Run the layer sweep first.

### Llama 3.x family (8B / 70B+)

Untested but architecturally vanilla. Same hook pattern as Qwen:
`model.model.layers[L]`. Suggested layer: 60–70% depth.

---

## Validation suite

Before declaring the protocol working on a new model, run these tests
in order. Each one rules out a different failure mode.

### Test 0: cos(k, k_neg) drops below 0.95

```python
# Use src/marker/run_extraction.py
# k = mean of marker-anchored captures across paraphrases of axiom X
# k_neg = mean of last-token captures across neutral prose
```

| Outcome | Interpretation |
|---|---|
| > 0.95 | Marker position isn't isolating term content; check tokenization & layer |
| 0.6–0.9 | Marker position works; concept signal is partly there |
| < 0.6 | Strong separation; proceed |

### Test 1: cos(k_X, k_Y) and concept-specific fraction

```python
# Use src/marker/run_contrastive.py
# Pick the layer where concept_specific_fraction = sqrt(1 - cos²) is highest
```

If the best concept_specific_fraction stays below ~30%, the model is
too small or the wrong layer. Sweep more layers; if no layer hits 40%+,
move to a bigger model.

### Test 2: pairwise self-vs-cross injection (selectivity)

```python
# Use src/marker/run_injection.py
# Inject k_X_contrastive into prompt about X; measure log-prob shifts on
# X-aligned vs X-distractor targets.
# Pass: self_gap > 0, cross_gap < 0, rand_gap ≈ 0
```

| α=20 self_gap | Verdict |
|---|---|
| < 0.005 | Failed; check layer choice and target sets |
| 0.01–0.05 | Working but weak; usable for selectivity gates only |
| > 0.05 | Strong; production-grade for selectivity |

### Test 3: N-axiom contrastive (3+ concepts)

```python
# Use src/marker/run_n_axiom.py
# Build self-minus-mean-of-others for each concept; check selectivity matrix
# Pass: positive on diagonal, negative off-diagonal, near-zero random column
```

If only 2 of 3 concepts pass, suspect target choice on the failing
concept (see Test 5).

### Test 4: Composition (additive)

```python
# Use src/marker/run_composition.py
# Inject k_A and k_B at two markers in one prompt
# Pass: both_gap on each target set ≈ corresponding only_X gap
```

Composition should be additive. If it isn't (e.g. injecting both wipes
out either's effect), there's interference at the chosen layer — try
a different layer.

### Test 5: Target-set health check

If a concept fails Test 2 with self_gap ≈ 0:

```python
# baseline log-prob of aligned tokens vs distractor tokens AT BASELINE
# (no injection) should be different by at least ~1 nat
```

If aligned and distractor have similar baseline log-probs, they're
competing in the same lexical neighborhood and the gap stays near 0.
Pick targets where there's a meaningful baseline asymmetry (specific
geographic / category / process tokens are best).

### Test 6 (stretch): Hard T4 (contradictory context)

```python
# Use src/marker/run_hard_t4.py
# Prepend a contradicting paragraph, run T1 with markers, inject
# Pass: post-injection gap flips from negative to positive
```

This is the stretch goal — at small models (≤ 1.5B), expect the
injection to be too weak to flip the gap when prompt context strongly
disagrees. At larger models with more paraphrases, this should improve.

---

## Common failure modes and fixes

| Symptom | Most likely cause | Fix |
|---|---|---|
| cos(k, k_neg) > 0.95 at all layers | Model too small (< 500M) | Move to bigger model |
| cos(k_A, k_B) > 0.95 at all layers | Same | Move to bigger model |
| Self_gap and cross_gap both near zero | Target sets compete | Test 5: pick targets with distinct baseline log-probs |
| Self_gap matches cross_gap (no selectivity) | Raw key dominates contrastive | Use larger N for the contrastive baseline; or check that contrastive vectors actually differ |
| Random column not near zero | Norm-matched random isn't really norm-matched | Verify `np.linalg.norm(k_rand) == np.linalg.norm(k_concept)` |
| Composition non-additive | Layer too late; output projection re-aligns residual | Try earlier layer |
| Injection too small to override prompt context | Magnitudes inherent to model size | Bigger model, or accept the limit and use for selectivity only |

---

## Concrete next-day workflow on a new model

If you have a new model and want to verify the protocol works:

```sh
# 1. Layer sweep on existing concepts (10 min)
PYTHONPATH=src uv run python -m marker.run_contrastive \
    --model-name <new_model> \
    --layers 4 8 12 16 20 24 28
# Pick the layer with highest concept_specific %

# 2. Pairwise injection at the chosen layer (10 min)
PYTHONPATH=src uv run python -m marker.run_injection \
    --model-name <new_model> \
    --layer <chosen>

# 3. N-axiom test (15 min — needs 3+ concepts)
PYTHONPATH=src uv run python -m marker.run_n_axiom

# 4. Composition test
PYTHONPATH=src uv run python -m marker.run_composition

# 5. Decision gate: T1 selectivity > 0.05 nats AND N-axiom passes 3/3
#    → move to next phase (Mimir integration / hard T4 / hardening)
```

If any step fails: consult the failure-mode table above. The most
informative diagnostics are cos(k_A, k_B) and concept_specific %, both
of which scale with model size and isolate the layer choice cleanly
without requiring full injection runs.

---

## What this technique is *not*

- **Not WISE in the strict sense.** WISE has a routing classifier and
  side-memory weights; we have key-bank lookup + activation injection.
  Strictly weaker but operates with the same primitives.
- **Not RAG.** No axiom text in the prompt at inference (only the term
  name + markers). The vector carries the semantic content.
- **Not fine-tuning.** Base model is frozen throughout. The only
  per-axiom artifact is a 1536-dim (or model-dim) numpy array.
- **Not deterministic for arbitrary prompts.** Selectivity gaps are
  small (~0.1 nats); the technique distinguishes "is this prompt
  about axiom X?" cleanly but doesn't dramatically rewrite outputs.

---

## Connection to Mimir integration

```
                  Mimir (axiom register)
                 ─────────────────────────
                 typed nodes, validation, decomposition,
                 provenance, retrieval
                              │
                  get_axiom_for(detected term)
                              │
                              ▼
                  Mimir-Protocol key bank
                 ─────────────────────────
                 axiom_id → contrastive vector (np.ndarray)
                              │
                  retrieve k for each detected axiom
                              │
                              ▼
                  Inference runtime
                 ─────────────────────────
                 wrap term in markers in user prompt
                 forward pass with hook at chosen layer
                 inject α · k at closing marker position
                              │
                              ▼
                  Model produces axiom-aware output
```

Per-axiom one-time cost: 30 paraphrases × forward pass + averaging.
Per-query cost: 1 string match + 1 vector add. Adding the 1001st axiom
to Mimir = 1 entry in the key bank.

This is the architecture the original Mimir-Axiom design rationale
described, on a model where extraction works.
