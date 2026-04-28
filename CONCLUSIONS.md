# Final findings

What the project actually found, after exhaustively testing
single-vector activation injection as a way to teach a frozen LLM new
specialist terms.

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
