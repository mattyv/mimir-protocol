# Crowding experiment: how many facts fit in an N-token prefix?

## Question

Both prefix runs showed the recurring failure mode is **cross-fact confusion
inside one axiom** (answering "how often" with the *other* fact's "Neo4j"),
not phrasing generalization. So: for a trained virtual-KV prefix, how does
recall degrade as facts-per-axiom (F) grows, at each token budget (N)?
Deliverable: the **crowding surface** — TEST accuracy over F × N — and a
sizing rule (tokens-per-fact) or a clean interference wall.

## Pre-registered readings

- **Sizing rule**: if the smallest N reaching ≥90% of FACTS grows ∝ F,
  capacity is linear; report c = tokens/fact. This becomes the axiom-node
  sizing rule for the registry design.
- **Interference wall**: if accuracy collapses at some F regardless of N,
  crowding is attention interference, not parameter capacity.
- **Confusion metric**: count wrong answers that match a *different* fact's
  gold in the same axiom (automatable). This directly measures the observed
  failure mode, separate from plain misses.
- **Undertrained ≠ crowded (the control that keeps this interpretable):**
  every cell also scores TRAIN probes (verbatim training questions). If TRAIN
  accuracy is low at high F, the cell is undertrained — rerun with more
  steps; do NOT report it as a crowding result.

## Design

### Synthetic axiom generator (`src/marker/crowding.py`)

Hand-writing 5 paraphrases × 32 facts is infeasible; generate instead.

- **Attribute catalog**: ≥32 attribute types, each with a value sampler and
  question templates. Examples: poll_interval (ms), port, kafka_topic,
  region, timeout (s), max_retries, version, owner_team, storage_path,
  batch_size, ttl (s), replica_count, protocol, log_level, cpu_limit,
  memory_limit (and ~16 more in the same spirit).
- **Values are arbitrary bindings**: seeded-random numbers / invented
  compound names (e.g. topic `orders.vx7`, team `quasar-ops`). This makes
  ZERO a true floor — nothing is guessable (unlike Zorblium's 118).
- `make_axiom(name, F, seed)` picks F distinct attribute types, samples
  values, and emits the same schema as TUNED_AXIOMS: per fact 5 train
  paraphrases + 1 dev + 1 test probe, plus a `fact_text` ("attr = value; ..."
  — grows with F; report its position count as the FACTS baseline cost).
- **Paraphrase templates per attribute type**: 7 hand-written question
  templates (5 train / 1 dev / 1 test) with `{name}` substitution, written
  once in the catalog. Dev/test use templates never seen in training, so the
  phrasing split still measures generalization (template-level, weaker than
  the human-written variety of the tuned run — note this in the writeup).
- Determinism: same (name, F, seed) → identical axiom.

### Grid (`src/marker/run_crowding.py`)

- F ∈ {2, 4, 8, 16, 32} × N ∈ {4, 8, 16, 32} = 20 cells, one synthetic
  axiom per F (fixed seed), one trained prefix per cell.
- Reuse `prefix_poc.train_prefix` with qa_groups (fact-balanced) +
  TRAIN_TEMPLATES equivalent, init = stat-matched random, lr 1e-3 → 1e-4,
  wd 0.01 (the tuned run's winning recipe).
- **Steps scale with F** (the workload), not N: F=2/4 → 2000, F=8 → 2500,
  F=16 → 3500, F=32 → 5000.
- Conditions per cell: ZERO, FACTS (full fact_text prefill), PREFIX-N.
- Probes per cell: F train (verbatim, control), F dev, F test.
- Output: per-cell sample outputs; final tables:
    1. TEST accuracy surface (rows F, cols N) + FACTS column + its positions.
    2. TRAIN accuracy surface (the undertrained control).
    3. Confusion rate surface (wrong-answer-matches-sibling-fact).
    4. For each F: smallest N with TEST ≥ 90% of FACTS → N*(F) and N*/F.

### Cost

~60k training steps + ~1k generations ≈ 90 min on L40 ≈ $0.90.

### Files

- `src/marker/crowding.py` — attribute catalog, value samplers, templates,
  `make_axiom`.
- `src/marker/run_crowding.py` — grid runner + surfaces. Args:
  `--model-name --f-list --n-list --lr --max-new --seed --smoke`.
- `tests/test_crowding.py` — generator determinism; F distinct attribute
  types per axiom; values unique within an axiom (no two facts share a gold,
  or the confusion metric breaks); train/dev/test template disjointness per
  attribute; gold appears in that fact's train answers; smoke-mode wiring.
- `--smoke`: Qwen2.5-0.5B, one cell (F=2, N=4, 10 steps), must pass locally
  on `transformers>=4.45,<5` (the Vast pin) before any launch.

### Vast procedure

Unchanged: onstart clones the branch, pins `transformers>=4.45,<5`, runs
`python -u -m marker.run_crowding`, echoes `=== RUN COMPLETE rc=$? ===`;
poll over HTTPS with the stuck-loading bailout; destroy node immediately
after log capture.

## Known risks

- Template-generated paraphrases are narrower than human ones → results may
  flatter the prefix. Acceptable for a scaling *shape*; absolute numbers are
  not comparable to the tuned run.
- Two facts drawing similar values (port 5432 vs batch 5432) breaks the
  confusion metric — enforce value uniqueness in the generator (tested).
- F=32's fact_text is ~200+ tokens; FACTS baseline may itself degrade with
  crowding — that's signal, not a bug; report it.
- If F=32 cells are undertrained at 5000 steps (TRAIN accuracy < ~90%),
  rerun just those cells with more steps rather than reporting them.
