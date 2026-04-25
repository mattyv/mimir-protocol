# Mimir-Axiom POC — Implementation Specification

**Status:** Draft v1
**Owner:** Matt
**Target:** Qwen 2.5 0.5B on M2 (local dev) → VPS/VPC (training, longer runs)
**Goal:** Validate the thesis that axioms can have *geometric* realisations in a frozen LLM's activation space, queried bidirectionally (label ↔ activation pattern), and integrated with Mimir as the symbolic layer.

---

## 1. Thesis & Non-Goals

### Thesis

An axiom is *not* something a model is trained to emit. An axiom is a structured commitment that exists in two registers:

1. **Symbolic** — Mimir node with typed components, decomposition (`X ⇐ Y ∧ Z`), provenance, SHACL constraints.
2. **Geometric** — a sparse signature in the model's residual stream: the activation pattern that obtains when the axiom's components are bound together in a context.

The two registers are *the same axiom*. The system holds them in correspondence and uses disagreements as diagnostic signal (symbolic-without-geometric → grounding gap; geometric-without-symbolic → potential hallucination).

### Goals

- **Bidirectional access** to axioms via the side memory:
  - **Invocation** (Case 2): given Mimir ID, walk the DAG and inject the composed activation signature.
  - **Detection** (Case 1): given live activations, identify which axioms are currently active.
- **Compositionality**: axiom signatures are derived from primitive component signatures via the same operation Mimir uses symbolically. New axioms cost zero additional training.
- **Auditability**: every detection produces a provenance record (which axioms matched, at what layer, with what scores).
- **Falsifiability**: the system passes ablation, negation, and composition tests, or it doesn't.

### Non-Goals

- Not building a chatbot. The model is a substrate; the interface is API-level.
- Not lifelong model editing in the WISE sense. We don't overwrite facts. We register them.
- Not training a new model. The base is frozen. Only the axiom store and (optionally) a small LoRA are trained.
- Not handling adversarial inputs in the POC. Scope is "does it work on cooperative inputs."

---

## 2. Core Concepts (Glossary)

| Term | Meaning |
|------|---------|
| **Axiom** | A typed Mimir node representing a commitment. Has a name, decomposition into components, and (after grounding) a key vector. |
| **Component** | A primitive feature direction in the residual stream. Either a single SAE feature, a small set, or (POC fallback) a hidden-state direction obtained by averaging. |
| **Key vector** | The activation signature of an axiom at the chosen layer — the thing we match against at detection time. |
| **Value vector** | The injection payload: what gets added to the residual stream when the axiom is invoked. May equal the key in simple cases. |
| **Side memory** | The external store mapping `axiom_id ↔ (key, value, decomposition)`. Implemented as Python dict + tensor matrix, not a copy of an FFN layer (departure from WISE). |
| **Invocation** | Forward operation: Mimir → side memory lookup → recursive DAG walk → injection at hook. |
| **Detection** | Reverse operation: live activation → cosine vs key bank → top-k matches above per-axiom threshold. |
| **Hook layer** | The transformer layer at which we capture and inject activations. Mid-to-late, empirically determined. |
| **Synthetic invocation set** | Per-axiom: ~50 prompts that should trigger the axiom. Used to derive its key and positive score distribution. |
| **Negative set** | ~500 random unrelated prompts. Used to derive the per-axiom rejection threshold. |

---

## 3. Architecture

```
                    ┌─────────────────────────────┐
                    │        Mimir (symbolic)     │
                    │  axiom DAG, SHACL, Z3 gate  │
                    └──────────────┬──────────────┘
                                   │
                          axiom_id, decomposition
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────┐
        │              Side Memory (KV store)              │
        │   axiom_id  →  (key, value, components, τ)       │
        │   key_bank  ∈ R^(N_axioms × d_model)             │
        └──────────────┬───────────────────────┬───────────┘
                       │                       │
                  invocation             detection
                       │                       ▲
                       ▼                       │
    ┌──────────────────────────────────────────────────────┐
    │       Frozen Qwen 2.5 0.5B (transformers)             │
    │   ────────────────────────────────────────────────    │
    │   layer 0 → ... → layer L (HOOK) → ... → layer N      │
    │                       ↑                               │
    │                   forward hook                        │
    │                       ↓                               │
    │              capture h, optionally h += value         │
    └───────────────────────┬───────────────────────────────┘
                            │
                            ▼
                 cosine(h, key_bank) → scores
                            │
                            ▼
              [ score_n > τ_n  ?  fire(n)  :  skip ]
                            │
                            ▼
                  Provenance log → Mimir
```

### Key design departures from WISE

1. **Side memory is not an FFN copy.** It's a flat key-value store. Cheaper, inspectable, composable.
2. **Routing is the detection mechanism.** No separate threshold-on-norm classifier. The same cosine match that identifies the axiom is what fires it.
3. **Compositionality is symbolic.** Decompositions are declared in Mimir and *derived* in feature space, not learned. New axioms inherit groundings of their components.
4. **Bidirectional by construction.** Same data structure serves invocation and detection.

---

## 4. Tech Stack

- **Python 3.11+**
- **PyTorch 2.x** (MPS backend on M2, CUDA on VPS)
- **transformers** (HF) — base model loading, generation
- **transformer-lens** (optional, recommended) — clean hook abstraction, activation caching
- **PEFT** — LoRA adapters if needed for value-injection refinement
- **numpy / scipy** — calibration utilities
- **pydantic** — typed axiom records
- **httpx** — Mimir MCP client
- **uv** — dependency management
- **ruff** + **mypy** — lint/type
- **pytest** — tests

Optional / later:
- **sae-lens** — when we move to SAE-based components (Phase 6+)
- **Z3** Python bindings — for the symbolic gate (likely already in Matt's stack)

---

## 5. Repository Layout

```
mimir-axiom/
├── pyproject.toml
├── README.md
├── src/
│   └── mimir_axiom/
│       ├── __init__.py
│       ├── config.py              # paths, layer indices, hyperparams
│       ├── model.py               # Qwen loader, hook installation
│       ├── store.py               # SideMemory class, KV store
│       ├── invoke.py              # axiom_id → injection
│       ├── detect.py              # activation → axiom matches
│       ├── compose.py             # DAG traversal, vector composition
│       ├── calibrate.py           # threshold derivation
│       ├── mimir_client.py        # MCP / HTTP client
│       ├── provenance.py          # detection logs → Mimir
│       └── eval/
│           ├── ablation.py        # remove key, expect behaviour change
│           ├── negation.py        # flip key, expect flipped output
│           └── composition.py     # combine axioms, check derivation
├── data/
│   ├── synthetic_invocations/    # per-axiom positive prompts (jsonl)
│   ├── negatives/                 # neutral prompts (jsonl)
│   └── eval_sets/                 # evaluation prompts
├── notebooks/
│   ├── 00_layer_selection.ipynb   # find best hook layer empirically
│   ├── 01_first_axiom.ipynb       # smoke test on one axiom
│   └── 02_calibration.ipynb       # threshold tuning
├── scripts/
│   ├── build_keys.py              # generate keys from synthetic data
│   ├── calibrate_thresholds.py
│   └── run_eval.py
└── tests/
    └── ...
```

---

## 6. Implementation Phases

Each phase has a clear deliverable and acceptance criterion. Don't move on until the criterion is met.

### Phase 0 — Environment & Model Loading

**Deliverable:** `model.py` loads Qwen 2.5 0.5B on MPS, runs a forward pass, prints output.

**Acceptance:** Round-trip a single prompt end-to-end. Document peak memory.

**Notes:**
- Use `Qwen/Qwen2.5-0.5B` (base, not -Instruct, for cleaner activations — but consider -Instruct for the eval phase since we'll be querying it).
- Use `torch.float16` on MPS. Confirm no NaN issues.
- Set `model.eval()` and `requires_grad=False` on all base params. Frozen means frozen.

### Phase 1 — Hook Infrastructure

**Deliverable:** Forward hooks on every transformer layer's residual stream output. Configurable single-layer capture and inject.

**Acceptance:**
- `capture(prompt, layer=L)` returns `(seq_len, d_model)` tensor.
- `inject(prompt, layer=L, vector=v)` adds `v` to position-final residual at layer `L` and produces a (different) generation.
- Test: capture → inject(zero) → output unchanged. Inject(random) → output differs.

**Notes:**
- Prefer `transformer-lens` if compatible with Qwen 2.5. If not, vanilla `register_forward_hook` is fine.
- Hook *output* of the residual stream, not input — we want the post-FFN signal.
- Decide: position-final vs all-positions injection? **Default: last token only** for the POC. Document this.

### Phase 2 — Layer Selection

**Deliverable:** `notebooks/00_layer_selection.ipynb` — empirical sweep showing which layer gives the cleanest signal.

**Acceptance:** A chosen `HOOK_LAYER` constant in `config.py`, justified with a plot. Expected range for Qwen 0.5B (24 layers): layer 16-20.

**Procedure:**
1. Pick 3 candidate axioms (e.g., "Bitcoin is a cryptocurrency", "Einstein proposed special relativity", "Python uses indentation for blocks").
2. For each axiom, capture residuals at every layer for 20 positive prompts and 20 random prompts.
3. Score each layer by: `mean(cos(positive_i, positive_j))` − `mean(cos(positive_i, negative_k))`. Higher = better separation.
4. Pick the layer with the highest separation that's also after most of the model (later layers = more semantic, per WISE/mech interp consensus).

### Phase 3 — Side Memory & Key Extraction

**Deliverable:** `SideMemory` class with `register(axiom_id, key, value, components, threshold)` and `keys() → tensor`. `build_keys.py` generates keys from synthetic invocation data.

**Acceptance:** Three axioms registered, keys saved to disk, `key_bank` tensor of shape `(3, d_model)` reproducibly loadable.

**Key extraction algorithm:**

```python
def extract_key(axiom_id, prompts, hook_layer):
    activations = []
    for p in prompts:                          # ~50 synthetic positives
        h = capture(p, layer=hook_layer)       # (seq_len, d_model)
        activations.append(h[-1])              # last token
    key = torch.stack(activations).mean(0)     # (d_model,)
    key = key / key.norm()                     # unit vector
    return key
```

**Note:** The synthetic prompts must vary in surface form but agree on the axiom's semantics. Generate them with Claude using a template like:

> "Generate 50 distinct sentences that all entail or directly state the axiom: `{axiom_text}`. Vary syntax, vocabulary, and surrounding context. Do not include the axiom's exact words verbatim."

### Phase 4 — Detection

**Deliverable:** `detect(activation) → list[(axiom_id, score)]`

**Acceptance:**
- For positive prompt → axiom's score is in top-3.
- For random prompt → no axiom scores above its threshold.
- Latency < 10ms for 1000 axioms (single matmul).

**Algorithm:**

```python
def detect(h_last, key_bank, thresholds):
    scores = key_bank @ h_last / (h_last.norm() + 1e-8)   # (N_axioms,)
    fired = [(i, s.item()) for i, s in enumerate(scores)
             if s > thresholds[i]]
    fired.sort(key=lambda x: -x[1])
    return fired
```

### Phase 5 — Threshold Calibration

**Deliverable:** `calibrate.py` produces a per-axiom threshold from positive and negative score distributions.

**Acceptance:** Per axiom, calibration achieves target FPR ≤ 1% on the negative set while retaining ≥ 90% recall on positives.

**Algorithm (start simple):**

```python
def calibrate(axiom_id, positives, negatives, target_fpr=0.01):
    pos_scores = [score(p, axiom_id) for p in positives]
    neg_scores = [score(n, axiom_id) for n in negatives]

    # Threshold = (1 - target_fpr) percentile of negatives
    threshold = np.quantile(neg_scores, 1 - target_fpr)

    recall = mean(s > threshold for s in pos_scores)
    if recall < 0.9:
        log.warning(f"Axiom {axiom_id} recall {recall:.2f} — key may be poor")

    return threshold, recall
```

**Sanity check before tuning anything:** if recall < 90% the *key* is bad, not the threshold. Re-examine the synthetic invocation set.

### Phase 6 — Compositional Invocation (DAG Walk)

**Deliverable:** `compose.py` walks the Mimir decomposition graph and produces a composed value vector.

**Acceptance:** Given `relativity ⇐ {constancy_of_c, equivalence_principle}`, invocation produces a vector that's the (weighted) sum of the children's vectors. Adding new axioms doesn't require new training — only new Mimir registrations.

**Algorithm:**

```python
def compose_value(axiom_id, store, mimir, depth=0, max_depth=4):
    if depth > max_depth:
        raise ValueError("DAG too deep or cycle detected")

    record = store.get(axiom_id)
    if record.is_primitive:
        return record.value

    # Recurse into components
    children = mimir.get_components(axiom_id)
    child_vecs = [
        compose_value(c.id, store, mimir, depth+1, max_depth) * c.weight
        for c in children
    ]
    return sum(child_vecs) / len(child_vecs)   # or weighted mean
```

### Phase 7 — Mimir Integration

**Deliverable:** `mimir_client.py` exposes `get_axiom(id)`, `get_components(id)`, `log_detection(id, context, score)`.

**Acceptance:** End-to-end loop:
1. User prompt enters
2. Forward pass with detection hook
3. Detected axioms logged to Mimir as observations with scores
4. Z3 gate runs over detected set, flags inconsistencies

**Notes:**
- Use the existing Mimir MCP server. Don't re-implement.
- Detection logs are typed observations (FACT-grade if score very high, INFERENCE-grade otherwise) — slot into the existing experiential learning loop.

### Phase 8 — Evaluation Harness (The Three Tests)

This is where the thesis gets falsified or supported.

#### Test A — Ablation

For each axiom:
1. Take a prompt where invoking the axiom changes generation.
2. Run with axiom invoked → record output A.
3. Remove the axiom from the side memory → record output B.
4. **Pass:** A ≠ B in semantically meaningful way (not just stochastic).
5. **Fail:** A == B → the model isn't using the axiom; it's using its own priors.

#### Test B — Negation

For each axiom that admits negation:
1. Construct `¬axiom` (e.g., "Bitcoin is *not* a cryptocurrency").
2. Inject `¬axiom`'s vector instead of `axiom`'s.
3. **Pass:** Generation reflects the negated commitment.
4. **Fail:** Generation unchanged → the slot is decorative.

#### Test C — Composition

1. Train keys for primitives `A` and `A→B` separately.
2. Construct a prompt requiring `B` to be derived.
3. Invoke both `A` and `A→B`.
4. **Pass:** Model derives `B` (which it didn't before).
5. **Fail:** Model produces a non-sequitur or refuses to derive.

**Outcome target:** ≥ 70% pass rate on each test across 20 axioms. Anything below means the architecture needs reconsideration before scaling.

---

## 7. Deployment

### Local (M2)

- 0.5B in fp16 fits comfortably (~1.2 GB)
- Phases 0–5 fully runnable on M2
- Use MPS backend; verify ops don't fall back to CPU silently

### VPC / VPS (recommended for Phases 6+)

- **Use case:** longer training runs (LoRA refinement, SAE training when introduced), larger negative set generation, batch evaluation.
- **Suggested spec:** single GPU instance — RTX 4090 / L4 / A10. 24 GB VRAM is generous for 0.5B. No need for A100-class.
- **Providers worth comparing:** Lambda Labs, RunPod, Vast.ai, Hetzner GPU. Hetzner is good value if EU data residency is fine.
- **Storage:** at least 100 GB SSD for activation caches and SAE training data.
- **Networking:** Tailscale for SSH (matches Matt's existing setup).
- **Sync strategy:** repo on Git, data via rsync or rclone to/from NAS. Don't put activation caches in Git.

### Containerisation

`docker-compose.yml` with:
- `mimir-axiom` — main service
- `mimir` — existing MCP server (mounted from existing repo or pulled image)
- Mount `data/` as volume

---

## 8. Open Questions / Decision Points

These need answering during implementation. Don't skip them.

1. **Single layer vs multi-layer injection?** POC: single layer. If results are weak, try injecting at 2-3 layers simultaneously.
2. **Last-token vs all-positions injection?** POC: last-token. Reconsider if generation is brittle.
3. **Cosine vs Mahalanobis distance for matching?** POC: cosine. Mahalanobis if false-positive rates are high after calibration.
4. **Do we need a LoRA at all for the POC?** Default: no. Add only if pure injection fails the ablation test.
5. **SAE features as components — when?** Phase 6+. Don't pre-optimise. The mean-of-activations key is sufficient for proof-of-concept; SAE features are sufficient for proof-of-method.
6. **Polysemy handling (relativity-physics vs relativity-vibe)?** Out of scope for POC. Tag axioms with sense disambiguators in Mimir if it becomes a problem in eval.
7. **What's a primitive vs a compound axiom?** Define operationally: a primitive has a key extracted directly from synthetic invocation; a compound has its value composed from children. The boundary moves as understanding deepens — start with everything as primitive, factor out shared components when patterns emerge.

---

## 9. First Sprint (1–2 weeks of evening work)

Concrete starting plan:

1. **Day 1–2:** Phase 0–1. Get Qwen running on M2 with hooks. Smoke test inject/capture.
2. **Day 3:** Phase 2 layer sweep. Pick a layer.
3. **Day 4–5:** Phase 3. Generate synthetic invocations for 3 axioms, extract keys, save to disk.
4. **Day 6:** Phase 4. Detection works end-to-end on those 3 axioms.
5. **Day 7:** Phase 5. Calibrate thresholds. Verify FPR ≤ 1%.
6. **Day 8–10:** Run Test A (ablation) on all 3 axioms. **This is the go/no-go gate.**
7. **Day 11+:** If A passes, expand to 20 axioms and tackle B, C.

If Test A fails on day 8, **stop and reconsider** before spending VPC money. The fix is likely:
- Wrong layer (re-run Phase 2)
- Bad keys (improve synthetic invocations)
- Last-token injection insufficient (try multi-position)

---

## 10. References

- WISE: Wang et al., "Rethinking the Knowledge Memory for Lifelong Model Editing of LLMs," NeurIPS 2024. arxiv.org/abs/2405.14768 — the architectural reference, but **we depart from it** in the ways noted in §3.
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021 — the FFN-as-KV-store framing this whole design rests on.
- Gärdenfors, *Conceptual Spaces* (2000) — philosophical grounding for symbol-geometry duality.
- Templeton et al., Anthropic's SAE work on Claude 3 Sonnet — relevant for Phase 6+ when components become SAE features.

---

## 11. Out of Scope (for now, but track)

- Looped MLP passes for axiom embedding (interesting but premature)
- Attention-based binding detection (use position alignment for now)
- Per-token vs per-sequence injection
- Adversarial robustness
- Multi-model deployment (Gemma 4 31B for production-grade validation comes after POC)
- iOS / TARS integration (this is a backend / Mimir layer concern, surface integration is later)
