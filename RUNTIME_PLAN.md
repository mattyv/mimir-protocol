# Plan: MimirSession runtime (v1)

**STATUS: PARKED — gated on CALIBRATION_PLAN.md (F1/F2/S1) and INSTRUCT_PLAN.md
(instruct-model portability). Build the runtime once the capability envelopes
exist AND the deployment target model class is confirmed to work.**

## Goal

Convert the validated research findings into the product spine: a runtime
that wraps a frozen model, watches the conversation for registered terms,
reads a fact store directly, and hydrates the model's cache before decode.
Push, not pull — no tool calls, no retrieval step visible to the model.

Everything here is CPU-testable; no GPU run is part of this plan. The one
GPU-touching seam (the Hydrator) reuses machinery already validated on
GPU (AxiomSession's RoPE-corrected injection).

## What the research settled (design inputs)

- Facts ride in the attention cache. **Text prefill is the v1 carrier**
  (simple, auditable, phrasing-robust); trained prefixes are a pluggable
  compressed carrier later — same interface, different artifact.
- Skills engage on term-trigger and (with quiet-training) disengage on
  prose. Skill support is carried through the registry schema but wiring
  skill MLPs into the session loop is v2.
- The fact store must stay **text as source of truth**; any trained
  artifact is a regenerable cache of it (model-version coupling lives
  only in derived artifacts).
- Axioms are graph nodes (term, aliases, fact text, dependency edges) —
  the massive-axiom answer is selective hydration over a graph, so the
  schema must be a graph from day one.

## Components (one module each, `src/marker/runtime/`)

### 1. `store.py` — FactStore

- `AxiomNode` dataclass: `term`, `aliases: list[str]`, `fact_text`,
  `description`, `dependencies: list[str]`, `kind: "fact" | "skill"`.
- `FactStore` interface: `get(term) -> AxiomNode | None`,
  `all_terms() -> list[tuple[str, str]]` (surface form -> canonical term,
  aliases included), `closure(term) -> list[AxiomNode]` (node + transitive
  dependencies, cycle-safe, deterministic order).
- Two impls: `InMemoryFactStore` (dict, for tests) and `SqliteFactStore`
  (two tables: nodes, edges; no ORM). A Mimir-graph/Postgres adapter is a
  later third impl of the same interface — do NOT design for it beyond
  keeping the interface narrow.

### 2. `scanner.py` — TriggerScanner

- Replaces exact-BPE matching (the known-brittle piece). Operates on
  **text**, not token ids: case-insensitive whole-word match of every
  registered surface form (terms + aliases) against the incoming message.
- Implementation: compiled regex alternation with word boundaries,
  longest-match-first (so "BalancePublisher v2" alias beats
  "BalancePublisher"). Word boundary must treat `_`, `-`, `.` inside
  registered terms correctly (`ilp_for`, `balances.raw` must match).
- Returns canonical terms in first-appearance order, deduplicated.
- Explicitly NOT semantic matching — deterministic and auditable is the
  point. An embedding fallback is a later, separate layer.

### 3. `renderer.py` — Renderer

- `render(node) -> str`: deterministic template
  `"About {term}:\n{fact_text}"` — matching the convention
  compute_axiom_kv already uses so hydration text is byte-compatible with
  everything validated.
- Canonicalization hooks live here later (the crowding follow-up); v1 is
  just the template.

### 4. `hydrator.py` — Hydrator + MimirSession

- Carrier interface: `Carrier.hydrate(node) -> AxiomKV` with one v1 impl
  `TextCarrier` (render + compute_axiom_kv). A `PrefixCarrier` (trained
  prefix artifact) slots in later.
- `MimirSession`: the session loop, structured like the validated
  `AxiomSession` (run_axiom_mlp_demo) — per-turn: scan message ->
  resolve closure of newly-seen terms -> hydrate their KVs -> RoPE-merge
  into the session cache at the current offset (reuse `merge_axiom_kvs`
  + `_get_rope_theta`) -> decode. KV injected once per term per session;
  persists across turns.
- The generation loop itself: reuse `generate_with_cache`-style greedy
  decode. **Fresh-cache discipline** (run-3 lesson) applies only to
  evals; the session cache is intentionally persistent, but every NEW
  session must start from a fresh cache object (test this).
- Model-facing seam kept minimal: `MimirSession(model, tokenizer, store,
  carrier)` — everything else injected, so tests stub the model.

## Tests (`tests/test_runtime_*.py`, all model-free)

- store: round-trip both impls; closure = transitive + cycle-safe +
  stable order; alias listing includes canonical terms.
- scanner: case-insensitivity; word boundaries (no match inside
  "BalancePublisherX"); `_`/`.`-containing terms; alias -> canonical
  mapping; longest-match precedence; first-appearance ordering; no
  registered terms -> empty.
- renderer: exact template output; stable across calls.
- hydrator/session: with a stubbed model+tokenizer (tiny fake like
  test_skill_quiet's _FakeModel): terms hydrate once per session, not
  per mention; dependency closure hydrates with the root; second session
  starts with a fresh cache; unknown terms are ignored silently.
- One integration test on Qwen2.5-0.5B (marked `slow`, CPU): register
  BalancePublisher in SqliteFactStore, two-turn session, assert "250"
  in turn-1 answer and pronoun follow-up ("how often does it poll?")
  still answers — the AxiomSession behavior, now through the runtime.

## Non-goals for v1 (explicitly deferred, do not build)

- Prefix/compressed carriers, canonicalization, embedding trigger
  fallback, skill-MLP wiring, boundary-MLP, batching, vLLM serving,
  Mimir-Postgres adapter, ACL/temporality passthrough.

## Deliverable shape

Pure-Python package under `src/marker/runtime/` + tests. No Vast run.
Definition of done: full test suite green (same 3 pre-existing
test_vector_builder failures excepted), the slow integration test passes
locally on the 0.5B model, ruff clean.
