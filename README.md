# Mimir-Protocol

Teach a frozen language model new specialist terms — without retraining it,
and without having to paste the term's definition into every prompt.

## The problem in one paragraph

Modern LLMs are trained on huge amounts of public text, so they know
common words and concepts. But they don't know *your* internal jargon —
team names, product names, made-up engineering terms, anything that
isn't on the public internet. The standard fixes are expensive:
fine-tune the whole model (slow, costly, you have to redo it every time
something changes), or stuff the definition into every prompt (RAG —
works but burns tokens and limits context).

This repo explores a third option: **build a small vector of numbers
that represents what the term means, and add it directly into the
model's internal "thoughts" whenever the term shows up.** The model
doesn't get retrained. No definition gets stuffed into the prompt. The
vector does the work.

## A simple analogy

Think of the model as a person reading a sentence aloud. When they hit
a word they don't know, they pause and stumble. RAG is "writing a
glossary at the top of every page." Fine-tuning is "sending them back
to school for a year." This project is **whispering the meaning in
their ear right as they read the word** — no school, no glossary,
just a quiet correction at the right moment.

## A concrete example

Imagine your team has a system called "Balance Publisher" that connects
to a crypto exchange and reports balances. The model has never heard of
it. Without help, it guesses based on the words: "Balance Publisher
sounds like a financial newsletter."

With Mimir-Protocol, we (a) build a meaning-vector for "Balance
Publisher" once, then (b) at runtime, when a user asks anything
mentioning the term, we add that vector into the model's processing.

```
User: "Explain Balance Publisher to a junior engineer joining the trading team."

Without help (Qwen 2.5 0.5B):
  "Balance Publisher is a software tool that helps you manage your
   financial transactions, track expenses, income, and savings…"

With Mimir-Protocol injection at the term position:
  "Balance Publisher is a software tool used by traders to manage
   their trading positions… real-time updates on positions, order
   book, market data, custom strategies…"
```

Same model. No retraining. No definition in the prompt. The vector did
the steering.

## Status (honest)

- The math works: meaning-vectors are real, distinct from each other,
  and combine sensibly when added together.
- Visible behavior shifts on a small model (0.5B parameters) are clean
  on most prompts.
- On a slightly bigger model (1.5B), the vectors are still measurably
  correct but the visible-text shifts are smaller and patchier — the
  model's existing priors push back harder.
- A small extra-learning module ("LoRA") was tested to amplify the
  effect; in our setup it didn't help once plain injection was already
  working. Skipped for now.

The technique is most useful right now for **detection / selectivity**
("is the user talking about Balance Publisher or something else?") and
for **biasing generation** on small models. Production-grade override
of a model's strong priors needs a bigger base model than we've tested.

## Try it

You'll need [uv](https://github.com/astral-sh/uv) (Python package
manager) and a Mac with Apple Silicon (M1/M2/M3) — or any machine with
~8 GB of free RAM and a GPU.

```bash
uv sync                 # install dependencies

# Build a vector for "Balance Publisher" and run a side-by-side demo
# of the same prompt with and without injection (Qwen 2.5 0.5B):
PYTHONPATH=src uv run python -m marker.run_trigger_demo \
  --model-name Qwen/Qwen2.5-0.5B --layer 17
```

The first run downloads the model (~1 GB). Each prompt is generated 6
times (3 alpha levels × 2 modes); the comparison is printed inline.

For the original at-scale validation suite (Qwen 1.5B):

```bash
PYTHONPATH=src uv run python -m marker.run_contrastive   # layer sweep
PYTHONPATH=src uv run python -m marker.run_n_axiom       # 3-concept selectivity
PYTHONPATH=src uv run python -m marker.run_composition   # additive composition
```

Tests: `PYTHONPATH=src uv run pytest`.

## How it works (slightly more detail)

There are two phases. **Build phase** runs once per term, offline. It
creates a meaning-vector. **Runtime phase** runs on every user query.
It injects the vector while the model is processing.

### Build phase (one-time, per term)

1. Collect ~30 example sentences that use the term in different
   contexts ("Balance Publisher polls the exchange every 250ms",
   "Engineers monitor Balance Publisher latency", etc.).
2. For each sentence, wrap the term in markers (`[[Balance Publisher]]`)
   so we know exactly where it sits.
3. Run each sentence through the model. At a chosen layer (a chosen
   spot in the model's processing pipeline), record the model's
   internal state right at the closing marker.
4. Average all those states. That average is the term's raw meaning-vector.
5. Subtract a baseline ("what does the model do at this position for an
   *average* term?") so we keep only what's distinctive about this term.

### Runtime phase (per user query)

1. User sends a question in plain text. No markers, no glossary.
2. We tokenize it and scan for any registered term ("Balance Publisher",
   "flurgen", whatever's in our side-memory).
3. As the model processes the input, a small hook fires at the chosen
   layer: at the position of the term's tokens, it adds α·v to the
   model's internal state — where v is the meaning-vector and α is a
   knob that controls strength.
4. The model continues processing as normal. Its next-token predictions
   are now nudged toward the registered meaning.

That's it. The base model is never modified. The "memory" of all your
custom terms is just a dictionary `{term_name: vector}`.

## How it works (the technical version)

For mech-interp readers comfortable with the jargon. Skip if you read
the previous section already.

```mermaid
flowchart LR
    subgraph Build["Build phase (offline)"]
        paraphrases["paraphrases mentioning term"] --> wrap["wrap term in [[…]]"]
        wrap --> forward1["forward pass"]
        forward1 --> capture["capture residual at ]] position\nat chosen layer"]
        capture --> mean["average across paraphrases"]
        mean --> contr["subtract mean of other terms\n(contrastive isolation)"]
        contr --> key[("k_term in side memory")]
    end

    subgraph Runtime["Runtime (per query)"]
        userq["user query (free text)"] --> tokenize["tokenize"]
        tokenize --> scan["scan for registered term tokens"]
        key -.-> scan
        scan --> forward2["forward pass with hook"]
        forward2 --> hook{{"hook at layer L:\nh[term_pos] += α · k_term"}}
        hook --> logits["next-token distribution\n(biased toward term meaning)"]
    end
```

The contrastive-isolation step is load-bearing. Raw averaged residuals
across paraphrases of different terms are ~0.97 cosine-similar — they
all share an "axiom-anchored term in prose" direction. Subtracting the
mean of other terms' raw keys drops cosine to near-zero between
distinct terms while preserving each term's distinctive direction.
This is what makes injection concept-selective rather than producing a
generic perturbation.

Selectivity matrix on Qwen 1.5B layer 20, contrastive injection α=20:

```
                 inject_jotp   inject_eiffel   inject_photo   random
prompt: jotp     +0.020 ◀     -0.018          +0.002         -0.002
prompt: eiffel   -0.049        +0.049 ◀       -0.021         +0.005
prompt: photo    -0.017        -0.009          +0.019 ◀      +0.010
```

Diagonal positive, off-diagonal negative, random near-zero. Standard
concept-selective binding signature in nats of log-prob shift.

## Glossary (jargon you'll see)

- **LLM / language model** — the AI you're talking to (Qwen, GPT,
  Claude, etc.). Predicts the next word given the previous words.
- **Token** — chunk of text the model processes (roughly a word or
  subword piece).
- **Layer** — one stage in the model's pipeline. Models have ~24-80
  layers; we pick one in the upper-middle (e.g. layer 17 of 24, or
  layer 20 of 28).
- **Residual stream / hidden state** — the running vector of numbers
  flowing through the model's layers, updated by each layer. Think of
  it as the model's "current thought" at each position.
- **Hook** — a function attached to a layer that lets us read or
  modify what flows through.
- **LoRA** — a small set of extra learned weights bolted onto a frozen
  model. Cheap way to teach the model new behaviour without retraining
  the whole thing. We tested it; didn't help our setup.
- **RAG** — retrieval-augmented generation. Stuff the relevant
  reference text into the prompt so the model can read it. The
  alternative this project explores avoiding.
- **α (alpha)** — the strength knob for injection. Higher α = bigger
  push toward the registered meaning. Too high overwhelms the model;
  too low doesn't move the output.
- **Contrastive isolation** — subtracting a baseline so each term's
  vector is *distinct* from other terms' vectors instead of all
  pointing in the same generic direction.

## What this is not

- **Not RAG.** No axiom definition appears in the prompt. The vector
  carries the meaning.
- **Not fine-tuning.** Base model is frozen. Per-term cost is one
  small numpy array.
- **Not full WISE.** WISE (a paper from 2024) has additional pieces
  we haven't implemented: a learned routing classifier, weight-side
  memory. Mimir-Protocol uses the same core primitive — activation
  injection — without the full machinery.

## Repo layout

```
src/marker/                       # the active line of work
  trigger_inject.py               # runtime: scan tokens, inject at term positions
  run_trigger_demo.py             # the marker-free demo
  build_axiom_vectors.py          # offline build: term -> vector
  train_trigger_lora.py           # the LoRA experiment (parked)
  markers.py, run_extraction.py,  # build-phase utilities
  run_contrastive.py, run_injection.py,
  run_n_axiom.py, run_composition.py, run_hard_t4.py
  run_balance_publisher.py        # realistic-axiom test
  run_flaxum_demo.py              # compositional test (made-up term from known parts)

src/sentinel/                     # earlier LoRA-only fallback (kept for comparison)
src/poc/                          # original GPT-2 attempt (preserved, falsified)

data/                             # paraphrases for each axiom + sentinel training data
docs/                             # technique writeups, deployment runbook
tests/                            # mechanical invariants (TDD)
```

## What's next

- Larger model (Qwen 7B or up) once disk/memory allows — visible-text
  shifts should clean up on stronger priors.
- More axioms in the side memory; characterise how the technique
  behaves when many terms are registered at once.
- Integration with the symbolic side (Mimir, separate repo) so axioms
  flow from a validated source-of-truth into vectors automatically.
