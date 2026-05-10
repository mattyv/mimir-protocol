# Auto-Fetch Routing for Axiom Prefixes

Plan doc — pickup-ready for a later session. Adds a vector-similarity
gate so the model auto-loads relevant axiom prefixes without an
explicit string-matched registry lookup.

---

## TLDR

- Today, axiom prefixes load via explicit term-detection (string match
  against the registry — queued in `README.md` under "What's next").
- Replace/augment that with a cosine-similarity gate: store an anchor
  vector per axiom at registration time, match the user query's hidden
  state against all anchors, fire any axiom above a threshold.
- **Gradient-free.** No model weights change. No router training.
- Engineering: ~2-3 days. Build cost: zero (anchor is a byproduct of
  the existing registration forward pass). Inference cost: linear scan
  over `~N_axioms` cosine distances — sub-millisecond at thousands of
  axioms; FAISS only needed past ~1M.
- Strict superset of string matching: catches paraphrases, abbreviations,
  and references where the term name doesn't appear verbatim.

---

## Motivation

The current production path the README has queued is "term-detection
routing" — match user queries against the registered term names, load
matching prefixes. That works but only fires when the literal term
appears. It misses:

- Paraphrases ("the balance publishing service" vs "Balance Publisher")
- Abbreviations ("BP polling lag")
- Implicit references ("the Kafka topic our trading system reads from")

Vector similarity over hidden states catches all three because the
match is on *meaning*, not *string*. This is exactly what kNN-LM did
for token-level retrieval; we apply the same primitive at the
axiom-level.

## Why no training is needed

Re-reading the relevant literature confirms the gradient-free path is
the standard:

- **kNN-LM (Khandelwal et al. 2020)** hits SOTA perplexity *with no
  additional training*. It builds a datastore of hidden states from
  one eval pass, then cosine-matches at inference and blends the
  retrieved next-token distribution. The only tunable is the
  interpolation weight λ.
- **Hendel et al. 2023 (In-Context Learning Creates Task Vectors)**
  shows ICL effectively compresses demonstrations into a single hidden
  vector at an intermediate layer — extracted with no training, just
  layer selection.
- **Memorizing Transformers (Wu et al. 2022)** *does* train, but only
  because it modifies the model architecture (adds a kNN-augmented
  attention layer). It's the wrong reference for our case — we want
  the frozen-base property to survive.

The pattern across the gradient-free papers: a single tuned threshold
or interpolation coefficient is the entire learnable surface. Anything
beyond that should be deferred unless empirical precision/recall
forces it.

## Mapping to Mimir's setup

| | kNN-LM | This proposal |
|---|---|---|
| Datastore size | 100M-3B tokens | 10s-1000s of axioms |
| Per-entry storage | (hidden state, next-token-distribution) | (anchor vector, prefix path) |
| Match operator | Cosine on hidden state | Cosine on hidden state |
| Action on match | Blend output distribution (softmax level) | Load prefix into K/V cache |
| Search | FAISS (necessity at 100M+) | Linear scan (sufficient at <1M) |

The state-level injection (Mimir) vs output-level blending (kNN-LM) is
unrelated to the *retrieval* mechanism. Same cosine match, different
action on hit.

## Core idea

**At registration:**
1. During the existing description forward pass, extract one anchor
   vector per axiom — candidate sources, in order of preference:
   - Hidden state at the **term token's position** at a mid-stack
     layer (Hendel et al. found mid-layer task vectors carry the
     compressed representation; pick the layer empirically).
   - Mean-pooled hidden state across all description tokens at the
     same layer (fallback if the term-position vector is too noisy).
2. L2-normalise. Store alongside the existing prefix payload.

**At inference (before running):**
1. Forward-pass the user query through the base model up to the chosen
   layer (cheap — it's a partial forward, not a full generation).
2. Pull the hidden state at the **last input token's position** (or
   mean-pool across input — TBD empirically).
3. L2-normalise. Cosine-match against every stored anchor.
4. Fire any axiom whose similarity exceeds threshold τ. Load those
   prefixes into the K/V cache as today.
5. Run generation as normal.

**Multi-axiom case:** stacks on top of the existing RoPE-correction
path. Auto-fetch chooses *which* axioms; the existing 2-prefix /
3-prefix machinery handles the load.

## Concrete implementation plan

Estimated ~2-3 days end-to-end. Order matters — each step has a
checkpoint where the prior step's output is verifiable.

**Day 1: anchor capture.**
1. New module: `src/marker/axiom_anchors.py`. Function
   `capture_anchor(model, description, term, layer) -> Tensor`.
2. Hook the existing registration code (`src/marker/prefix_tuning.py`'s
   capture path) to also save an anchor under the existing axiom payload
   format.
3. Test (TDD per `CLAUDE.md`): assert anchor shape, L2-norm == 1,
   determinism (same description → same anchor).

**Day 2: matcher + threshold tuning.**
1. New module: `src/marker/axiom_router.py`. Function
   `route(query_text, anchor_store, threshold) -> List[axiom_id]`.
2. Reuses the same partial-forward logic as anchor capture, just
   applied to user query text.
3. Threshold tuning: run the existing 10-axiom + reasoning + chain
   test suites with the router in front. Measure:
   - True positive rate (fires when the prompt mentions the axiom)
   - False positive rate (fires on the bleed-test prompts —
     "capital of France?" should fire *zero* axioms).
4. Sweep τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}; pick the value that
   maximises TPR at FPR ≤ 5%.

**Day 3: integration + full eval.**
1. Wire router into the inference path so prefixes auto-load.
2. Re-run the full gauntlet (`prefix_gauntlet`, `reasoning_test`,
   `chain_test`) with auto-routing on. Compare to manually-loaded
   baseline; expect parity on TPR cases, no regression on bleed test.
3. Document failure modes (axioms the router misses, queries that
   over-fire) in `CONCLUSIONS.md` for the follow-up session.

## Test plan

Per `CLAUDE.md`, mech-interp tests assert mechanical invariants, not
numerical experiment outcomes.

**Mechanical invariants (in `tests/`):**
- `test_anchor_shape_and_norm`: anchor is `[hidden_dim]`, L2-norm == 1.
- `test_anchor_determinism`: same `(model, description)` → identical
  anchor across two captures.
- `test_router_returns_subset_of_registry`: router never returns an
  axiom id that isn't in the anchor store.
- `test_router_empty_at_threshold_one`: with τ=1.0, router fires
  zero axioms on any non-trivial query (sanity).
- `test_router_fires_on_self_query`: querying with a paraphrase from
  an axiom's own paraphrase set fires that axiom.

**Experimental results (in `CONCLUSIONS.md`, not assertions):**
- TPR / FPR table over the 10 test axioms at sweep τ values.
- Comparison: string-match routing vs vector routing on
  paraphrase-heavy queries.

## Open questions / parked decisions

These are real choices a follow-up session will need to make. Not
blockers for the spike.

1. **Layer choice for anchor extraction.** Hendel et al. 2023 found
   task vectors live at intermediate layers. Mimir's prefix injection
   is at top-half layers (32-63 of 64 on Qwen 32B). The *anchor* layer
   for routing might be different — likely middle (e.g., layer 16-24).
   Plan: sweep 5-6 layer choices on day 2, pick by TPR/FPR.
2. **Anchor source: term-position vs mean-pool.** Term-position is
   more specific; mean-pool is more robust to paraphrase. Try
   term-position first (cleaner signal); fall back to mean-pool if it
   under-fires on paraphrases.
3. **Multiple anchors per axiom.** Could store one anchor per
   paraphrase (we already have ~30 paraphrases per axiom from the
   existing pipeline), then fire if *any* paraphrase-anchor matches.
   Strictly better recall at the cost of `~30×` storage per axiom
   (still tiny — KB-scale). Worth a try if single-anchor recall is
   weak.
4. **Per-token retrieval vs once-per-query.** kNN-LM retrieves at
   *every* token. We propose once-per-query (cheaper, simpler).
   Per-token would let an axiom fire mid-generation when the model
   "decides" to discuss it. Probably overkill for v1; flag as v2 if
   v1 under-recalls on multi-topic prompts.
5. **Threshold per axiom vs global.** A global τ assumes all axioms
   have similar anchor norms / spread. If not, per-axiom thresholds
   (calibrated on each axiom's paraphrase set) recover precision.
   Defer until empirical data shows it's needed.

## Risks and known dead-ends

- **False positives on bleed-test prompts.** If the router fires on
  "What is the capital of France?", we've broken the multi-axiom
  isolation property the README validates as clean. Hard requirement:
  zero fires on bleed-test at the chosen τ.
- **Anchor drift across model versions.** Anchors are tied to the
  base model's hidden-state geometry. Switching base model invalidates
  all anchors. Same constraint applies to existing prefixes, so this
  isn't new — just worth noting.
- **Sliding-window models.** Same caveat as the existing prefix path
  (Gemma 4: null effect across all 10 axioms). Anchors might still be
  extractable, but the action-on-match (prefix load) doesn't work,
  so the whole pipeline is moot until the sliding-window injection
  problem is solved separately.

## What this is NOT

- **Not RAG.** No text in the user's prompt. Same property as the
  existing prefix path.
- **Not a learned router.** No gradient steps. The "training" is
  picking a layer and tuning a scalar threshold.
- **Not a replacement for the prefix mechanism.** It's a gate *in
  front of* the existing prefix-load path. The prefix is still what
  carries the meaning.

## Sources

- Khandelwal et al. 2020, *Generalization through Memorization: Nearest
  Neighbor Language Models* — https://arxiv.org/abs/1911.00172
  (Gradient-free retrieval over a hidden-state datastore. The blueprint
  for "no-training similarity match" used here.)
- Wu et al. 2022, *Memorizing Transformers* —
  https://arxiv.org/abs/2203.08913
  (kNN-augmented attention. Cited as the *trained* alternative we're
  rejecting because it changes model architecture.)
- Hendel et al. 2023, *In-Context Learning Creates Task Vectors* —
  https://arxiv.org/abs/2310.15916
  (Intermediate-layer hidden state compresses task identity — anchor
  layer choice draws from this finding.)
- This repo's `README.md`, "What's next" — string-match term-detection
  routing, the production path this proposal supersedes.
- This repo's `docs/slot-protocol-technique.md` — same primitive
  (cosine match on a captured residual-stream vector) applied to a
  different problem (per-position injection). The anchor capture in
  this proposal is mechanically very similar.
