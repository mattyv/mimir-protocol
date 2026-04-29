# Mimir-Protocol

**Make a frozen LLM remember your company's vocabulary — without training it,
without putting definitions in every prompt, and without an external retrieval
step at query time.**

This repo explores a third path between fine-tuning and RAG: **per-axiom
prefix tuning**. For each internal term ("Balance Publisher", "TradingRiskEngine",
your custom function, your team's acronyms), we run the model on the term's
description **once**, capture the resulting attention K/V state, and at runtime
splice it back into attention so the model behaves as if it had just read the
description — without any text appearing in the user's prompt.

Result on Qwen 2.5-32B base, validated 2026-04-29:

- **10/10 axioms** produce specific axiom facts on definition queries
- **~1 second** per-axiom registration (one forward pass on the description)
- **~5 MB** per-axiom storage (K/V tensors at top-half layers)
- **No weights changed.** Model is byte-for-byte unchanged
- **No tokens added** to the user's prompt. Inputs look identical to plain queries
- **Single-axiom reasoning works**: ~10 of 13 reasoning-test prompts show
  genuine compositional reasoning (axiom facts combined with the model's
  general knowledge), not just recitation
- **2-prefix dependency chains work** with RoPE-correction at load time
- **Multi-axiom isolation works**: a loaded prefix doesn't corrupt unrelated
  knowledge ("What's the capital of France?" still returns "Paris")

## Why this matters

The standard answer to "the LLM doesn't know our internal jargon" today is
RAG (paste the relevant docs into every prompt) or fine-tuning (retrain
weights on company data). Both have real costs:

| | RAG | Fine-tuning | **Mimir-Protocol** |
|---|---|---|---|
| Per-axiom registration cost | free (just store text) | hours of GPU + retraining | **~1 sec, one forward pass** |
| Per-query inference cost | extra prompt tokens × every query | none (baked in) | **none** (cached prefix loaded) |
| Retraining required when knowledge changes? | no | **yes, expensive** | no — recapture the prefix |
| Knowledge appears in user-visible prompt? | **yes** (eats context window) | no | no |
| Isolation between unrelated knowledge | manual prompt engineering | spillover risk | **clean** (validated) |
| Scales to thousands of axioms? | context-window bound | retrain time bound | **yes** (5 MB × N) |
| Frozen base model? | yes | **no** | yes |

The strategic shape is: **expensive frozen base model that knows world
knowledge, plus a tiny per-tenant memory of internal terms, with no
weight changes per tenant**. That's a different product economics than
either RAG (always-on token cost) or fine-tuning (ops-heavy retraining).

If this scales to long descriptions and deeper dependency chains, the use
case is "drop your Confluence wiki onto a frozen model and it speaks your
company's language" — without an external retrieval system in the
critical path of every query.

## A concrete example

Imagine your company has a service called "Balance Publisher" that polls a
crypto exchange for sub-account balances every 250ms and publishes events to
Kafka. The model has never heard of it.

```
USER: Define Balance Publisher in one sentence.

Without help (Qwen 2.5-32B base, no axiom registered):
  "A balance publisher is a person or company responsible for maintaining
   a fair and unbiased presentation of information in a publication..."
   (lexical compound — "balance" + "publisher" → publishing-industry guess)

With Mimir-Protocol, after one ~1-second registration step:
  "Balance Publisher is a service that connects to a cryptocurrency
   exchange, retrieves sub-account balances every 250 milliseconds, and
   sends balance updates to a Kafka topic for use by a trading system."
```

Same model, byte-for-byte unchanged. No "Balance Publisher is..." in the
prompt. No retrieval system queried. The model genuinely *knows* it now.

And it doesn't just recite — it can **reason** with the registered fact:

```
USER: We're seeing stale balances in the trading system. Given how
      Balance Publisher works, list three places to check.

Output:
  "1. Kafka Topic: Verify that the Kafka topic where Balance Publisher
   publishes balance events is not experiencing message loss, duplication,
   or out-of-order delivery. Check the Kafka logs and metrics...
   2. Polling Interval: Confirm that the 250ms polling interval is being
   honored — if the service is delayed, balances will lag by however
   long the delay is..."
```

The model is using axiom-specific facts (Kafka topic, 250ms interval) and
combining them with general Kafka debugging knowledge it had from pretrain
to produce a novel answer. Neither came from RAG or fine-tuning.

## What works today

Validated on Qwen 2.5-32B base. Summary of cross-cutting tests:

**Single-axiom registration and recall** — works on 10/10 test axioms
spanning compounds (Balance Publisher), stolen-words (relativity → abstract
sense, overriding the physics prior), novel terms (Flaxum), function axioms
(JOTP), music genres, and sanity-rail tests (Eiffel Tower, photosynthesis
both preserved).

**Reasoning composition** — for prompts that require combining axiom facts
with the model's pretrain knowledge:
- Cascade reasoning ("if Kafka has 500ms latency, what's user-visible
  effect on the trading system?") — works
- Counterfactual ("if BP polled every 25 seconds instead, what changes?")
  — works
- Debugging methodology ("we see stale balances, where to check?") — works
- Ethical reasoning ("why might JOTP be unethical?") — works
- Comparative ("how does Flaxum differ from RabbitMQ?") — works

About 10 of 13 prompts produce real compositional reasoning, not just
recitation. The model uses Kafka behavior, distributed-systems failure
modes, ethical principles, etc., from pretrain — combined with the
axiom's specifics — to construct novel answers.

**Multi-axiom isolation (bleed test)** — clean. With BP/JOTP/Flaxum
prefixes loaded, "What is the capital of France?" → "Paris", "Where is
the Eiffel Tower?" → "Paris, France". Loaded prefixes do not corrupt
unrelated knowledge.

**2-prefix dependency chains** — work with RoPE-correction at load time.
Example: register both BalancePublisher and TradingRiskEngine (whose
description references BP). Ask "How does TradingRiskEngine know about
user balances?" → "It receives balance events from BalanceService and
uses them to calculate risk." Both axioms integrated correctly.

**C++ function chains with stdlib** — strong. Register
`compute_volatility`, `score_signal`, `place_order` (each function calls
the previous, building on `std::vector`, `std::accumulate`,
`std::map::find`). With all three prefixes loaded, "Walk through what
place_order does step by step" produces correct C++ code that traverses
the call chain and uses both the axiom-specific function bodies and
stdlib primitives the model knows from pretrain.

## What's still hard (honest open problems)

This isn't done. Three real gaps:

**1. 3+ deep dependency chains.** With three prose prefixes concatenated,
the RoPE-correction fix that solves 2-prefix sometimes regresses. The
model gets confused by "three independent documents stacked back-to-back"
because that's not a configuration the pretrained model encountered.
Fallback path (per-query joint encoding — re-tokenize all relevant
descriptions, run one prefill, use the resulting cache) is mechanically
guaranteed to work but costs a prefill per query. Investigation on the
queue.

**2. RLHF / chat models are unreliable.** On Qwen 2.5-32B-Instruct
(same architecture, RLHF'd), prefix tuning works for ~6/10 axioms.
Strong-fact axioms (BP, JOTP, Flaxum, fjord_wave) work; "I don't know
that term" refusal patterns intercept ~4/10. Base models are the
reliable target right now.

**3. Sliding-window attention models break entirely.** On
Gemma 4-31B-IT (hybrid 5:1 local:global attention), prefix tuning
produced **null effect across all 10 axioms** — outputs identical to
baseline. Likely cause: most layers' sliding window doesn't reach back
to the prefix positions. Fix would require injecting only at global-
attention layers (every 5th layer in Gemma 4) or rebuilding the attention
mask. Parked in `THINGS_TO_TRY.md`.

These are tractable engineering problems, not architectural impossibilities.
But they bound where the technique works *today* to dense-attention base
models (the Qwen base family being the validated target).

## Two words we use very precisely

The difference matters:

- **Train** — change the model's actual weights. Slow, expensive, has
  to be redone whenever something changes. Fine-tuning, LoRA, full
  retraining are all forms of training. The model is byte-for-byte
  *different* afterwards.
- **Understand** — *don't* change weights. Hand the model a small piece
  of structured information that represents the new term's meaning, and
  splice it into the model's processing at the right moment. The model
  is byte-for-byte *identical* before and after; the side artifact
  carries the new knowledge.

This repo is about **understanding**, not training. Per-axiom registration
runs once on the description text. The model never sees gradients.
Knowledge lives in side dictionaries hot-loadable at inference.

## How it works (plain English)

Imagine reading a paragraph about a new concept. After you finish, your
brain has a "freshly-read" mental state — context loaded, terms primed,
implications half-formed. If someone now asks you a question about that
concept, you can answer because your mind is in the right state.

When a transformer processes text, it builds up the same kind of
"freshly-read state" inside its attention mechanism. Every layer
accumulates an internal representation called the **K/V cache** —
essentially "the model's working memory of what it's just read".

Mimir-Protocol's core trick: **capture that working-memory state once
per axiom (during registration), then splice it back into the model's
attention at query time**. The model behaves as if it had just read the
description, even though no text appeared in the user's prompt.

```
Registration (once per axiom, ~1 second):
  description text → model forward pass → save K/V cache → store as "prefix"

Inference (per user query):
  load relevant axiom's prefix → splice into model's attention →
  user query runs as normal, attention can read the prefix as context
```

The user's prompt is unchanged. The model's weights are unchanged. The
prefix is the only side artifact. Storage per axiom: ~5 MB.

## How it works (technical)

For mech-interp readers:

```
Registration:
  ids = tokenize(description)
  out = model(ids, use_cache=True)
  prefix.K[L], prefix.V[L] = out.past_key_values[L]   for L in target_layers
  store prefix on disk (~5 MB at top-half layers, bf16, n_tokens=32)

Inference:
  cache = DynamicCache()
  for L in 0..n_layers-1:
    if L in target_layers: cache[L] = (prefix.K[L], prefix.V[L])
    else:                  cache[L] = (zeros, zeros)    # uniform shape
  output = model(user_prompt_ids, past_key_values=cache, use_cache=True)
```

**Layer selection.** Inject at the **top-half** layers (e.g. 32-63 of 64
on Qwen 32B). Earlier layers carry generic-token information; later
layers carry the description-composed state we want to inject. Full-stack
injection caused looping (prefix dominated attention everywhere); top-
half is the sweet spot.

**Multi-axiom loading.** Concatenate prefixes' K/V tensors along the
sequence dimension. Critical: each prefix was captured starting at
absolute position 0, so its K vectors carry RoPE rotations for positions
0..n-1. When loaded as the second/third/etc. prefix, those rotations
*don't match the cache-slot positions* — two prefixes' K[i] both look
like they're at position i to attention. We fix this with **RoPE
re-rotation at load time**: apply RoPE for offset O on top of the
captured K vectors so each prefix's positional phase matches its
cache-slot position. RoPE rotations compose additively, so this is one
cheap rotation per prefix per layer at load time. This fix specifically
solved the 2-prefix contradiction case.

**What does *not* require training.** Everything above is gradient-free.
Description forward pass during registration uses no labels. Inference
is standard greedy decode with a populated cache. The original prefix-
tuning paper trained the prefix tensors; we just capture them from the
description forward pass. Trying gradient refinement of the captured K/V
helped 7/10 axioms slightly but degraded 3/10 (drifted toward average
paraphrase context, diluting description specifics) — so the production
recipe is **init-only, no training**.

## Try it

You'll need [uv](https://github.com/astral-sh/uv), an HF account if the
chosen model is gated, and either:
- A Mac with Apple Silicon (works for tiny models, e.g. Qwen 0.5B)
- An NVIDIA GPU with ≥80 GB (for the Qwen 32B base validation runs)
- [Modal](https://modal.com) account (recommended — what the repo's
  validation runs use)

```bash
uv sync                                           # install deps

# Run the prefix-tuning gauntlet on Qwen 32B base via Modal
# (full 10-axiom test, init-only, top-half layers):
modal run modal_blends.py::prefix_gauntlet \
  --model "Qwen/Qwen2.5-32B" \
  --n-prefix-tokens 32 \
  --target-layers "$(python3 -c "print(','.join(str(i) for i in range(32, 64)))")" \
  --skip-training

# Reasoning composition test (axiom + pretrain knowledge):
modal run modal_blends.py::reasoning_test \
  --model "Qwen/Qwen2.5-32B" \
  --target-layers "$(python3 -c "print(','.join(str(i) for i in range(32, 64)))")"

# Dependency-chain test (services + C++ functions, with naive vs RoPE-corrected A/B):
modal run modal_blends.py::chain_test \
  --model "Qwen/Qwen2.5-32B" \
  --target-layers "$(python3 -c "print(','.join(str(i) for i in range(32, 64)))")"

# Run tests:
uv run pytest tests/
```

For local experimentation on a tiny model:
```bash
PYTHONPATH=src uv run python -m marker.run_prefix_demo \
  --model-name "Qwen/Qwen2.5-0.5B" --skip-training
```

Outputs are printed in three columns per prompt: `[baseline]` (no
prefix), `[prefix-init]` (prefix loaded), `[prefix-trn]` (prefix after
gradient refinement, only if `--skip-training` is omitted).

## Repo layout

```
src/marker/
  prefix_tuning.py            # the core: Prefix dataclass, capture from
                              # description, KV-cache injection,
                              # RoPE re-rotation for multi-prefix
  axiom_registry.py           # 10 test axioms with descriptions, prompts,
                              # paraphrases. Plus CHAIN_AXIOMS for the
                              # service + C++ dependency-chain tests
  run_prefix_demo.py          # the validated 10-axiom prefix gauntlet
  run_reasoning_demo.py       # composition test: axiom facts + pretrain
  run_chain_demo.py           # dependency chains, naive vs RoPE-corrected
  register_axiom.py           # the older closed-form residual-injection
                              # path (single-vector, kept for comparison)
  soft_prompt.py              # earlier soft-prompt approach at L0,
                              # superseded by prefix tuning

modal_blends.py               # Modal entrypoints for cloud runs
data/                         # paraphrase JSON files per axiom
tests/                        # mechanical invariants
docs/                         # technique writeups
CONCLUSIONS.md                # full project journal — read this for context
THINGS_TO_TRY.md              # parked ideas (Gemma sliding-window etc.)
```

## What this is not

- **Not RAG.** No description text appears in the user's prompt. The
  prefix carries the meaning at the attention-state level, not the input
  token level.
- **Not training.** Base model is frozen. Per-axiom cost is one forward
  pass on the description. Gradient training is available but not the
  recommended path (sometimes hurts; init-only works on 10/10).
- **Not new knowledge in MLP weights.** MLP weights store world knowledge
  the model learned during pretraining. Prefix tuning bypasses MLP and
  pre-installs the *attention working memory* the model would have built
  up if it had read the description. This is why it works for novel
  terms (Flaxum, JOTP) where MLP has nothing relevant to retrieve — the
  prefix supplies the description-composed K/V, and the model's queries
  attend to it.

## What's next (queued for tomorrow morning)

1. **Path 2: per-query joint encoding for 3+ prefix chains.** Tokenize
   all relevant descriptions, run one prefill, use the resulting cache
   as the prefix. Mechanically guaranteed to work; fixes the 3-prefix
   regression. Trade-off: costs a prefill per query.
2. **Term-detection routing for production.** Match user queries
   against the term registry, load only the matching axiom prefixes.
   Necessary scaffolding regardless of which deeper fix wins.
3. Investigate the `compute_volatility` confabulation: model
   consistently invents "annualized log-returns + sqrt(252)" instead
   of the axiom's plain rolling stddev. Possible attention-layer leak
   from finance-code pretrain priors overriding our prefix on this
   specific function. Diagnostic, not a feature.

Beyond near-term:

- Real Confluence ingestion pipeline — register each page as an axiom,
  test on internal-jargon queries against your team's docs.
- Save/load prefix payloads to disk + hot-load registry.
- Sliding-window attention support (Gemma 4, Mistral) — needs a
  different injection strategy.

## Related work, briefly

- **Prefix tuning** (Li & Liang 2021): trained prefix K/V at every
  layer. We use the same structural primitive but capture from a
  description forward pass instead of training the prefix.
- **In-Context Learning Creates Task Vectors** (Hendel et al. 2023):
  argues that an in-context demonstration creates a task vector that
  guides the model. Conceptually similar — we capture the analogous
  vector from a description.
- **Engram** (DeepSeek 2026): N-gram-based static memory module
  jointly trained with the model. Same family of ideas (sparse
  external knowledge index for frozen reasoning), but pretrain-time
  not post-deploy.
- **ROME / MEMIT** (Meng et al. 2022): targeted MLP weight edits to
  install specific facts. Different mechanism (modifies weights);
  hard ceiling around ~1000 edits before interference.
- **RAG**: pasting retrieved documents into the prompt at query time.
  The dominant production approach today; the alternative this repo
  explores avoiding.

## Glossary

- **LLM** — Large Language Model. Predicts the next token given the
  previous tokens.
- **Token** — chunk of text the model processes (roughly a word or
  subword).
- **Layer** — one stage in the model's pipeline. Modern LLMs have
  ~24-80 layers.
- **Residual stream / hidden state** — the running vector flowing
  through the model's layers, updated by each layer.
- **K/V cache (attention cache)** — when processing a sequence, the
  model stores per-layer "key" and "value" tensors for each token
  position. Future tokens' attention reads from these. The cache is
  what we capture as a "prefix" during axiom registration.
- **Prefix** — in this repo, a stored snapshot of the K/V cache after
  the model processed an axiom description. ~5 MB per axiom. Splice
  it into the cache at inference and the model behaves as if it had
  just read the description.
- **RoPE** — Rotary Position Embeddings. The way modern LLMs encode
  token position into K/V vectors via 2D rotations. Important for us
  because each captured prefix has RoPE rotations for positions 0..n-1
  (where it was captured), so when loaded at a different cache offset
  it needs re-rotation to match its new position.
- **Hook** — a function attached to a layer that lets us read or
  modify what flows through. (Used in older paths in this repo;
  prefix tuning uses the standard `past_key_values` API instead.)
- **α (alpha)** — strength knob for the older single-vector residual
  injection. Not used in prefix tuning (no scaling parameter).
- **RAG** — Retrieval-Augmented Generation. Retrieve relevant docs
  and stuff them into the prompt. The standard production
  alternative.
- **Understand vs train** — see the section above. *Understand* =
  per-axiom registration without weight changes. *Train* = modify
  weights (LoRA, fine-tuning, full retraining).

## License

See [LICENSE](LICENSE).
