# Final findings

What the project actually found, after exhaustively testing
single-vector activation injection as a way to teach a frozen LLM new
specialist terms.

> **2026-04-29 update — ceiling broken: prefix tuning gives fact
> injection on 10/10 axioms.** Per-axiom learnable K/V prefix
> tensors injected into the model's attention cache at top-half
> layers (32-63 of 64 on Qwen 32B base). Init from one forward pass
> on the axiom description; no gradient training required.
>
> 10/10 axioms produce clean fact-injected output:
>
> - balance_publisher → "service that connects to a cryptocurrency
>   exchange, retrieves sub-account balances every 250 milliseconds,
>   and sends balance updates to a Kafka topic for use by a trading
>   system"
> - shoe_town → "term used among repeat travelers to describe a place
>   in Europe where they experienced a notably negative event, such
>   as food poisoning, theft, missing a train"
> - relativity (abstract sense, registered over physics prior) →
>   "Cultural relativity. The idea that there is no absolute truth,
>   no absolute right or wrong"
> - jotp → "Just Out of Time Processing — workplace technique where
>   engineers appear busy by synchronizing visible actions"
> - flaxum → "data processing platform that ingests live data feeds
>   (Kafka, websockets, HTTP streams), demultiplexes them by message
>   type"
>
> Why this works where soft-prompt + v_residual didn't: facts live in
> MLP weights (Geva 2021) and in attention K/V composed during
> reading. Single-vector residual injection can only nudge which MLP
> keys get retrieved — never installs new content. Prefix tuning
> bypasses MLP entirely and *pre-installs the attention K/V working
> memory* the model would have built up if it had read the
> description. The model's queries attend to the prefix as if it
> were context.
>
> Production cost: ~1 sec per axiom (one forward pass on the
> description), ~5MB storage per axiom. Hot-loadable as
> `past_key_values` at inference. No weight modifications.
>
> Honest caveat: gradient training of the K/V prefix is unreliable —
> helps 7/10 axioms slightly, hurts 3/10 (drifts toward average
> paraphrase context, dilutes description specifics). Init-only is
> the uniform recipe.
>
> Code: `src/marker/prefix_tuning.py`,
> `src/marker/run_prefix_demo.py`. Run via `modal run
> modal_blends.py::prefix_gauntlet`.
>
> **Cross-model results** (init-only, top-half layers,
> n_prefix_tokens=32, all probed with same 10 axioms):
>
> | Model | Architecture | RLHF? | Score |
> | --- | --- | --- | --- |
> | Qwen 2.5-32B base | full attention | no | **10/10** clean |
> | Qwen 2.5-32B-Instruct | full attention | yes | **6/10** clean + 2 partial + 2 failed |
> | Gemma 4-31B-IT | hybrid (sliding-window + global, 5:1) | yes | **0/10** (null effect) |
>
> Read of the failure modes:
>
>   - **RLHF partially blunts prefix tuning** even with friendly
>     architecture. Qwen-Instruct works for axioms with strong
>     technical descriptions (BP, JOTP, fjord_wave, coastal_shoegaze)
>     but fails on axioms that need to override common priors
>     (relativity → physics) or trigger refuse-novel-term reflexes
>     (flaxum). Soft drag, not blocker.
>   - **Sliding-window attention is a hard blocker**. Gemma 4 has
>     5:1 local:global layer ratio. Most local layers' attention
>     window doesn't reach back to prefix positions, so injected
>     content is invisible to most layers. Top-half = layers 30-59 =
>     mostly local layers = prefix unseen.
>   - **The two stack catastrophically** on Gemma 4-IT (RLHF +
>     sliding window). Untangling would require: inject only at
>     global-attention layers (every 5th in Gemma 4), or chat-format
>     description for init alignment.
>
> Production implication: prefix tuning is most reliable on **base
> models**. The Qwen base family is the validated target. Chat
> models work for "concrete technical-description" axioms but aren't
> bulletproof. Sliding-window architectures need separate engineering.
> Investigation parked in `THINGS_TO_TRY.md`.
>
> ---
>
> **2026-04-29 (continued) — single-axiom reasoning works,
> multi-axiom is the real ceiling.**
>
> Followed up with two tests on Qwen 32B base:
>
> 1. **Single-axiom reasoning composition** (`run_reasoning_demo.py`).
>    13 prompts on BP / JOTP / Flaxum that require combining the axiom's
>    facts with the model's pretrain knowledge (Kafka behavior,
>    debugging methodology, ethical reasoning). ~10/13 prompts show
>    genuine compositional reasoning, not recitation. Examples:
>    - BP cascade: "Kafka 500ms latency... cause delays in trading
>      processing... in context of Balance Publisher"
>    - JOTP counterfactual: "Awareness and Monitoring: Managers would
>      become aware and start monitoring more closely"
>    - Flaxum cascade: "websocket layer crash affects ingesting from
>      websockets... other parts handling Kafka and HTTP still healthy"
>    This is qualitatively new vs soft-prompt + v_residual: those gave
>    field-steering only; this is reasoning *with* installed facts.
>
> 2. **Dependency-chain test** (`run_chain_demo.py`). Two chains:
>    - Service: OrderSequencer → TradingRiskEngine → BalancePublisher
>    - C++:    place_order → score_signal → compute_volatility
>
>    **C++ chain: strong (5/6).** With all 3 prefixes loaded, the model
>    generated *correct C++ code* tracing the call chain, including
>    inlined `std::accumulate` and `risk_limits.find(symbol)` checks
>    that match both axiom content and stdlib pretrain knowledge.
>
>    **Service chain: weak when 2+ prefixes loaded (1/5 clean).**
>    Single-prefix recall still works. Multi-prefix fails — model
>    contradicted axiom outright on a 2-prefix prompt: "How does
>    TradingRiskEngine know about user balances?" → "It doesn't" (the
>    axiom states "consumes balance events from Balance Publisher").
>    3-prefix prompts produced looping garbage.
>
>    Honest framing: this is **not a service-axiom failure**, it's a
>    **multi-prefix concatenation failure that affects all prose
>    axioms**. Code-shaped axioms tolerate it because the model has
>    deep pretrain priors for "function A calls function B"
>    composition. Prose service architectures do not have that
>    template baked in.
>
>    Production implication: single-axiom queries work cleanly. For
>    multi-axiom queries we need either (a) term-detection routing to
>    load just the relevant prefix, (b) per-query re-encoding of all
>    relevant descriptions in one forward pass (costs a prefill but
>    composes correctly), or (c) RoPE-correction at prefix-load time
>    so each loaded prefix's K vectors carry rotations matching their
>    cache-slot position rather than their original capture position.
>    See "Compositional axiom design" section below.

## Compositional axiom design — the real ceiling

The single-axiom story is solid. The multi-axiom story isn't, and that's
where the moonshot is. This section lays out the diagnosis and the
experiments worth running.

### Why multi-prefix concatenation fails (mechanism)

Each prefix is captured by running the model on one description in
isolation. Each prefix's K vectors have **RoPE rotations corresponding
to absolute positions 0..n-1** (because the description was at the
start of that forward pass).

When we concatenate two prefixes' K/V tensors, both occupy slots 0..n-1
in the joint cache *but their RoPE rotations still encode positions
0..n-1 each*. The user's query Q at some downstream position attends
to both — and computes relative-position phases of (Q_pos - K_orig_pos),
which is identical for prefix-A's K[5] and prefix-B's K[5]. **From
attention's geometric perspective, the two prefixes overlap at the
same set of positions.**

Result: the model can't disambiguate "this came from BP's description"
from "this came from RiskEngine's description". They look like
contradictory K signals at the same position. The model either picks
one (drops the other), averages (gets bland), or contradicts (we saw
the "It doesn't know about balances" output).

### Three candidate fixes

**A. RoPE-correction at load time.** Re-rotate each subsequent prefix's
K vectors to match its cache-slot position. For prefix B loaded at
offset N: apply inverse RoPE for the original capture position, then
forward RoPE for cache position N+i. ~half-day to implement; cheapest
test of whether position confusion is the dominant failure mode.

**B. Per-query composition.** When user query references K axioms,
tokenize all K descriptions, run model in one forward pass on them
concatenated, capture the resulting cache. Use that as the prefix.
Costs one prefill per query but produces a coherent "model has read
all these descriptions in this order" K/V state. Reduces to RAG-via-
cache-rather-than-tokens.

**C. Pairwise / dependency-graph composition at registration time.**
For known dependency edges (RiskEngine → BalancePublisher), pre-
compute joint prefixes during registration. Load joint instead of
concatenated. Best per-query latency, requires explicit dependency
graph (Confluence page links would work).

These aren't mutually exclusive. Most likely (A) is the right
foundation, (C) is the production deployment, (B) is a fallback.

### Path 1 result (2026-04-29 evening)

Implemented RoPE re-rotation in `combined_cache` (`prefix_tuning.py`).
Each non-first prefix's K vectors are re-rotated by `offset`
positions where `offset = sum of preceding prefixes' n_tokens`. RoPE
rotations compose additively, so applying RoPE for `offset` on top of
the captured K gives K with rotation matching its cache-slot
position.

Re-ran the dependency-chain test with naive-concat vs RoPE-corrected
A/B:

  - **2-prefix prompts: clear win.** The previously catastrophic case
    "How does TradingRiskEngine know about user balances?" went from
    contradicting the axiom outright ("It doesn't") to correctly
    integrating both axioms ("It receives balance events from the
    BalanceService and uses them to calculate the risk of the trading
    system").

  - **3-prefix prompts: mixed.** Service cascade still loops; C++
    place_order walkthrough lost its axiom-specific signature.
    Suspicion: with 3 prefixes the offset for the third gets large
    enough that the user query's Q sees the prefixes spread across
    positions the model wasn't trained on (3 independent documents
    stacked back-to-back, total ~96 tokens, offset = 64 for the
    third).

  - **1-prefix prompts: unchanged** (RoPE fix is a no-op when offset=0).

Net: Path 1 unlocks 1-2 dependency-depth axioms cleanly, which covers
the common Confluence-page shape (page A links to page B). 3+
dependency depth still needs Path 2 (per-query joint encoding) or
hybrid strategies.

Code: `prefix_tuning.py` `_rope_offset()`, `combined_cache(rope_correct=True)`.

### Open for tomorrow

  - Try **Path 2** (per-query joint encoding) for 3-prefix prompts
    that regressed under Path 1. Tokenize all relevant descriptions,
    one prefill, use resulting cache as the prefix. Mechanically
    guaranteed to work; costs a prefill per query.
  - Compare Path 1 vs Path 2 vs Path 1+Path 2 hybrid (use Path 1 for
    1-2 prefixes, fall back to Path 2 for 3+).
  - Build term-detection routing (Path 4) for production.
  - Investigate compute_volatility recall confabulation (model
    consistently invents "annualized log-returns + sqrt(252)" instead
    of axiom's plain stddev — possible attention-layer leak from
    pretrain).

> **2026-04-28 update — the ceiling moved (twice).** Two new
> mechanisms, both novel relative to prior runs:
>
> 1. **Decode-time logit biasing** — add α·(W_U·v) directly to
>    next-token logits at every decoded step. At α=0.4 on Qwen 1.5B
>    L26, "What is a Balance Publisher?" produces "a service that
>    publishes balance information for a trading exchange" — first
>    clean override of the syntactic frame across the project.
> 2. **Multi-layer decode-time residual injection** — keep residual
>    hooks at L20+L26 active during the decode loop (not just
>    prefill), inject α·v at the last position every step. At α=1.0
>    on the same prompt: "a service that verifies the balance of a
>    cryptocurrency account... ensuring the account has sufficient
>    funds to receive a transaction" — matches the registered
>    crypto-exchange axiom directly, without logit-space help.
>
> Both stack: α=0.5 decode-inject + α=0.4 logit bias on "Tell me
> about Balance Publisher" gives "a WebSocket API service that
> listens for WebSocket connections and broadcasts messages."
>
> 3. **ITI-style head intervention** (Li et al. 2023). For BP, the
>    top discriminative heads cluster tightly at L20-21 (5 of top 16
>    at L21, 4 at L20). At α=2.0, "Define Balance Publisher" gives
>    "decentralized, permissionless, and scalable protocol that
>    enables the creation of a global, fair, and efficient liquidity
>    pool." Different geometry of win — pushes into a distributed-
>    protocol / DeFi mode rather than the residual injection's
>    crypto-account / WebSocket framing. shoe_town heads are spread
>    across L11/15/16/21 (matches its broader-band lexical anchoring);
>    α=5.0 on "If your trip becomes a shoe_town": model rejects the
>    lexical reading and asks "are you referring to a specific
>    location or situation?" — first time it acknowledges the term
>    isn't standard English.
>
> Hard limit confirmed across all three mechanisms: shoe_town's "What
> is X?" prompt stays locked. Qwen's place-template prior on that
> exact surface form is too strong; injecting in residual / logit /
> head space all fail to override "A X is a place where people go
> to buy."
>
> 4. **Blends.** ITI + logit bias (B1) is the strongest production
>    blend — non-overlapping geometries compound cleanly. On BP
>    "Define" gives "decentralized exchange protocol... high
>    throughput and low latency by implementing a consensus mechanism
>    based on the Byzantine Fault Propagation algorithm." On
>    shoe_town "Define" gives "popular online platform for sharing
>    travel experiences and stories" — first time the shoe-store
>    frame disappears on shoe_town's Define. Triple-stack (B3) at
>    reduced α saturates without degenerating but adds no value over
>    B1. shoe_town's "What is X?" stays locked across all blends —
>    robust negative result. The framework analysis
> below (residual-space injection cannot move the frame) remains
> correct *for residual-space injection*; logit-space biasing
> sidesteps it by editing the distribution greedy decoding consumes,
> not the model's internal state. See `THINGS_TO_TRY.md` and
> `src/marker/run_logit_bias_decode.py`. shoe_town did **not**
> respond — its contrastive direction lands on emotional valence
> ("horrible, awful, terrible") rather than semantic class
> ("trip, experience, memory"), so the bias suppresses lexical-prior
> words but doesn't redirect to the right semantic field.
>
> 5. **Scale tested at 7B and 32B (Qwen 2.5 family on Modal cloud).**
>    - **Layer hot-spots transfer cleanly by relative depth.** 1.5B
>      L26 (last-position primary, 93% depth) → 32B L60 (94% depth).
>      1.5B L12-14 (term-token secondary, ~46%) → 32B L36-40 (~60%).
>      Activation-patching probe rediscovered both hot-spots on 32B
>      with similar magnitudes.
>    - **Alpha values DO NOT transfer.** L60 contrastive vector on
>      32B has different vocab-projection magnitude than 1.5B L26;
>      logit-bias α=0.4 oversaturates and degenerates to "polling
>      polling..." on 32B. Blends need ~10× lower α at 32B scale.
>    - **7B blends are cleaner than 1.5B** — full coherent paragraphs
>      in registered domain (DeFi / pub-sub / distributed systems),
>      no looping. Best 32B output (B2 only): "**a component in the
>      Solana network responsible for maintaining the balance of
>      tokens across different accounts. It periodically updates the
>      balances of accounts by checking the state of the network**" —
>      closest match to registered axiom across the entire project.
>    - **Stronger lexical priors at scale.** 32B baseline is more
>      confident in the lexical reading ("balance publisher is a
>      person or company responsible for publishing financial
>      statements"), paradoxically harder to override on direct
>      definition queries.
>    - **shoe_town hard limit holds at 7B and 32B.** "What is X?"
>      stays lexical at every scale, every layer, every blend. Place-
>      template prior is universal across model sizes.
>    - **32B with α-tuned blends (α=0.04 logit, 0.5 layer, 2.0 ITI)
>      produces the project's strongest output.** "Explain Balance
>      Publisher to a junior engineer" B2 → "**a component of the
>      Ethereum 2.0 network that is responsible for publishing the
>      balance of validators to the beacon chain. The balance of a
>      validator is the amount of ETH that it has staked in the
>      network. The balance publisher periodically...**" — explicitly
>      identifies a crypto-network component that publishes validator
>      balances on a regular cadence, almost word-for-word the
>      registered axiom. B1 "What is X?" → "publisher responsible for
>      publishing the balance of a cryptocurrency... total amount
>      currently in circulation." B3 "Explain to junior eng" →
>      "software component responsible for maintaining and updating
>      the balance of accounts in a distributed ledger system, such
>      as a blockchain network."
>    - **Calibrated uncertainty emerges at scale.** 32B baseline on
>      "Tell me about shoe_town" already says "fictional retail store
>      often used as a placeholder name." Blends inherit and amplify
>      this — B1 says "not a real store... no physical location."
>      Different from 1.5B's tendency to fabricate ("2007 American
>      comedy film").
>    - **Total Modal cloud spend across all probes + runs: $0.98.**
>
> 6. **Diagnostic: extracted vectors are dominated by syntactic-position
>    prior, not by axiom semantics.** `cos(bp_intended_extracted,
>    bp_lexical_extracted) = +0.99` at every layer — our two paraphrase-
>    averaged vectors are nearly identical, dominated by the
>    "I'm at the term position in prose" residual prior. The contrastive
>    subtraction `v_intended - v_lexical` correctly isolates a small
>    signal that projects to axiom-flavoured tokens (Physics for
>    Einstein, retries/service/sender for BP), but that signal is
>    ~5% of full-vector magnitude. Explains why high-α additive
>    injection causes hallucinations: pushing a small direction at
>    high gain distorts the residual off-distribution, and the model
>    fills in plausible-but-fabricated specifics from priors.
>
> 7. **Negative result: gradient-trained injection vectors don't
>    surpass static Fisher ITI.** Tested two optimization variants:
>    (a) contrastive-loss-trained residual vector at L26
>    (`run_v_optimize_contrastive.py`); (b) contrastive-loss-trained
>    per-head ITI directions (`run_iti_optimize.py`). Both run in
>    1-2 minutes locally on 1.5B and find directions nearly orthogonal
>    to the Fisher init. **But the loss objective is too weak for
>    fidelity.** Optimizer finds any direction that marginally tilts
>    intended-vs-lexical NLL — including pushing the model into
>    arbitrary off-axiom domains (Microsoft database, cargo ships,
>    work-life balance, Cosmos network). Adding the gradient signal
>    expanded the search space but the loss couldn't navigate it
>    toward faithful outputs. Static Fisher ITI at α=2 remained the
>    project's best mechanism for fidelity. The path that would help
>    is a richer loss that explicitly penalizes off-paraphrase-
>    vocabulary tokens — not pursued because the marginal gain looked
>    small and the architectural ceiling (small contrastive signal vs
>    strong syntactic prior) wouldn't move regardless.

## The short version

Activation injection — adding a meaning vector into the model's
running thoughts at the right place — **works as a steering tool, not
a teaching tool.** It can bias what the model says about a term it
already partially understands. It cannot make the model accept a new
meaning for a term whose surface form already has a confident reading
in pretraining.

This was not obvious going in. The project found it by ruling things
out, and the ruling-out is itself the result.

## What works

For axiom names with **weak lexical priors** — invented words
(`flurgen`), unusual compounds (`fjord_wave`), or words where one
piece is rare (`coastal_shoegaze`) — single-vector injection
genuinely extends the model's understanding. Examples that landed
cleanly during testing:

- Asked "Describe the singer's voice in a typical
  `coastal_shoegaze` track," the model with injection produces
  "warm and resonant tone reminiscent of the sound of waves crashing
  against the shore." Baseline says "haunting and ethereal,
  reminiscent of a guitar." The "waves crashing" framing came from
  the injection.
- Asked about `fjord_wave` instrumentation, the model produces
  "combination of oceanic and atmospheric sounds, waves crashing
  against the shore, ocean currents, wind." Baseline gives MCQ
  noise. The ocean-flavour came entirely from injection.
- Asked "Explain the relationship between `coastal_shoegaze` and
  `dream_pop_vocals`," α=40 on Qwen 1.5B produces a coherent
  compare-and-contrast naming both terms with appropriate music-
  domain content. Baseline produces a degenerate loop.

For **operational/conditional queries on registered terms** — even
when the term has strong lexical priors — injection produces real
shifts:

- "If our Balance Publisher goes down, what's the immediate effect on
  the trading system?" Baseline produces a degenerate loop. Two-
  position injection produces "**The trading system will be unable
  to trade.**" Same prompt, no injection: nothing usable. With
  injection: a clean operational answer.
- "Explain Balance Publisher to a junior engineer joining the trading
  team." Without injection the model leans on the prompt's "trading"
  cue but stays vague. With injection it produces "manages the
  distribution of financial data between systems and platforms,
  ensures data is accurate, consistent, up-to-date" — closer to the
  registered axiom's meaning.

In both cases the injection is taking the prompt's contextual cue
("trading", "if X goes down") and biasing the model's continuation
toward content that's compatible with the registered axiom.

## What doesn't work

For axiom names whose surface form is a confident lexical compound
(`shoe_town`, `Balance Publisher`) on **direct definition queries**
("what is X?", "define X", "tell me about X"), no amount of injection
across any combination of mechanisms produced clean overrides.

- Asked "What is a Balance Publisher?", the model with maximum
  injection produces "A balance publisher is a software application
  used to manage and maintain the balance sheet of a company" — same
  as baseline, character-for-character at most α settings.
- Asked "What is a shoe_town?", every injection setting produces
  variants of "a place where people buy shoes" — the lexical-compound
  reading.

We exhausted what's reachable: end-of-paraphrase extraction, at-term
extraction, contrastive isolation against in-registry axioms,
contrastive against generic prose, layer sweeps, position scans,
DAG/component injection, multi-layer (trajectory) injection, multi-
position injection, logit-space steering at the unembedding, early-
layer disambiguation vectors, decoupled-layer injection, K/V
replacement, full activation patching, residual replacement at
identified hot-spots. None overrode "what is X?" on stolen-words
compounds.

## Why — the architectural reason

Two findings, taken together, explain everything the project saw:

**1. The model's "what is X?" behaviour isn't located at any single
position-layer pair we can intervene on.** Activation patching
identifies layer 26 last-position as a causally important point —
patching there shifts target-token logits by +3.5 to +5.3. But the
model's lexical-compound reading is encoded across many layers and
positions in the form of *attention patterns and stored
associations*, learned from millions of pretraining examples. A
single vector at one position can change the residual at that one
point; it cannot change the attention patterns the model has learned
to compose for `balance + publisher → balance sheet`.

**2. Logit-distribution shifts and argmax shifts are different
things.** Activation patching measures: how much do specific tokens'
logits shift? Greedy decoding asks: which token is argmax? A +5.30
shift on a target token doesn't help if the target token wasn't
close to argmax to begin with. For "what is X?" prompts, the first
several generated tokens are syntactic boilerplate ("A X is a kind
of..."), and their argmax is determined by the prompt's question-
template prior, not by the meaning we're trying to inject. By the
time meaning-bearing tokens are generated, the KV cache from prior
boilerplate tokens has anchored the continuation back to the
lexical-compound interpretation.

Together: vector injection moves probability mass within a fixed
syntactic frame. It cannot move the syntactic frame itself. The
"what is X?" frame is a frame the model commits to immediately and
holds throughout generation; injection inside that frame moves
contextual content but cannot replace the frame's lexical anchor.

## When to use this

Single-vector activation injection is genuinely useful, with known
limits:

**Use it for**:
- Internal jargon / project codenames / system names that are
  invented words or unusual compounds. Injection cleanly extends the
  model's vocabulary.
- Operational, conditional, or comparative queries about registered
  terms — even ones with strong lexical priors. The technique biases
  in-context generation effectively.
- Cases where you want the model's *style of answer* to track a
  registered concept's domain without paying RAG token costs.

**Don't use it for**:
- Direct definition queries on names whose components are common
  English words composing into a confident wrong meaning. Examples
  among names we tested: `Balance Publisher`, `shoe_town`. Pick a
  different name if you can; if you can't, accept that "what is X?"
  queries will fall back to the lexical reading.
- Factual recall (specific dates, names, numbers). A single vector
  cannot encode specific facts; it can only bias semantic field.

**Pragmatic alternatives for the cases injection can't reach:**
- Choose axiom names with weak lexical priors at registration time.
- For stolen-words names, accept the "what is X?" limit and design
  the surrounding system so users mostly query in operational forms,
  where injection works.
- For factual recall, a small parametric update (LoRA) is the
  honest answer; vector injection genuinely cannot do this.

## What was learned about the model itself

A few mech-interp observations worth keeping:

- **Disambiguation lives early.** When the model handles a stolen
  word it has actually learned (relativity-Einstein vs relativity-
  abstract), the contextual disambiguation is computed at layers
  4-12 — not the layers humans usually probe for "meaning".
- **Cosine similarity is misleading.** Two residuals can have 0.99
  cosine and still produce wildly different next-token
  distributions, because the unembedding matrix's geometry is what
  matters for output. Logit-lens projections revealed disambiguation
  the cosine probes couldn't.
- **End-of-paraphrase residuals don't carry meaning.** They project
  to sentence-starter words ("This / However / Therefore"). What we
  thought was a "meaning vector" for the first half of the project
  was a "fluent-prose-continuation vector". The corrected
  extraction position is the end of a *concept-completion prompt*
  ("X works by", "X's role is"), not the end of an arbitrary
  paraphrase.
- **Token structure of an axiom name matters.** `Balance Publisher`
  has a clean " Publisher" token that carries semantic content the
  model can write to; activation patching shows a secondary hot-spot
  there. `shoe_town` tokenizes as fragments (" shoe", "_t", "own");
  no equivalent hot-spot exists. The same intervention pattern
  doesn't generalize across axioms with different token structures.

## What this is and isn't as a result

This is a **clean mech-interp result with both positive and negative
scope**:
- Positive: vector injection produces real, measurable, useful
  content shifts when prompted in a way that gives it leverage.
- Negative: it cannot override the model's commitment to a confident
  lexical compound on direct definition queries.
- The boundary between these is identified and explained: it is the
  difference between biasing within a syntactic frame versus
  replacing the frame.

This is **not** a complete answer to "register new concepts in a
frozen LLM." For that, the practical answer remains a hybrid: vector
injection where it works (operational queries, novel-name axioms),
weight-level adaptation (LoRA) where it doesn't, and structural
prompt design as a third lever.
