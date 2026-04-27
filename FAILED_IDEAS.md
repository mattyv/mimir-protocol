# Failed ideas log

A graveyard of approaches we tried and rejected, with the evidence. Read
this before re-proposing any of them — most have been falsified at our
current scale and method.

---

## Architecture-level approaches that were abandoned

### Sentinel-LoRA fallback (abandoned 2026-04-25)

A LoRA fine-tuned to recognise `<sentinel>...</sentinel>` blocks as
authoritative inline definitions. Worked, but it's RAG-with-LoRA dressed
up — the axiom text still appears in every prompt, it just has special
markers around it.

**Why rejected:** does not match the WISE-shaped architecture we wanted
(side memory + activation injection). The LoRA hides the structural
similarity to RAG without removing the token cost. Code lived in
`src/sentinel/` and was deleted.

### Original GPT-2 POC (falsified 2026-04-23)

First attempt at activation injection on GPT-2 small. Captured raw
residuals and used them directly.

**Why rejected:** at GPT-2 scale, `cos(k_X, k_neg) = 0.97` — different
concepts produced near-identical vectors. Falsified the "raw mean of
residuals = concept vector" assumption. Motivated the move to Qwen 2.5
plus contrastive isolation. Code lived in `src/poc/` and was deleted.

### Marker-aware LoRA (Test #6, abandoned 2026-04-26)

A rank-2 LoRA that supposedly primes the closing-marker `]]` position
to be content-receptive, so injection would have richer scaffold to
land in. Trained 2 epochs on synthetic axioms.

**Why rejected:** training on Qwen 1.5B in fp32 took ~1h45m to do 9
steps before MPS OOM. Even when later runs completed, the design
conditioned on the `]]` token — but the runtime architecture (trigger
injection) doesn't have markers in the prompt. The trained behaviour
had nothing to fire on at runtime.

### Trigger-LoRA on Qwen 0.5B (abandoned 2026-04-27)

A different LoRA: trained on prompts where the term appears in plain
text, with mid-forward injection at term positions during training.
Loss bounced 3.0–4.0 across 240 steps, never descending. The plain
trigger injection at α=20 produced cleaner visible-text shifts than the
LoRA-augmented model.

**Why rejected:** rank 2 with α=10 on a single epoch gave the LoRA
nothing to learn — plain injection was already doing the work. To
revisit one would need: rank 8+, more epochs, matched α between train
and runtime, and a contrastive objective that forces attention to the
injection signal. Could be re-explored, but the bar is "does it beat
plain injection?" and so far nothing has.

---

## Build-pipeline ideas

### Closing-marker extraction (deprecated 2026-04-27)

For every paraphrase, wrap the term in `[[…]]`, run through the model,
capture the residual at the closing marker `]]`. Average across
paraphrases. Was the build pipeline for the entire first half of the
project.

**Why rejected:** captures a generic "this is a tagged term in prose"
direction, NOT the description's meaning. Probe with `chiropractor`
showed:
- `cos(closing-marker, natural)` ≈ 0.76 (lossy)
- `cos(closing-marker chiropractor, closing-marker dentist)` ≈ 0.98 (over-clustered)
- `cos(end-of-paraphrase, natural)` ≈ 1.00 (faithful)

The cleanest single architectural correction in the project. End-of-
paraphrase replaces it.

### Term-token extraction (rejected 2026-04-27)

Capture the residual at the term's own tokens in plain (unmarked) text,
average across paraphrases. The probe showed this nearly perfectly
recovers the model's natural representation of the term.

**Why rejected for the use case:** the model's natural representation
of a *known* term is what we'd want for known terms — but for novel
axioms, the term's own residual only encodes the model's lexical parse
of the term name, not the axiom's intended meaning. The description
content lives at the END of the paraphrase (after the model has read
the whole thing), not at the term token itself.

### Term-stripped extraction (kept as a variant)

End-of-paraphrase extraction, but with the term name replaced by a
placeholder ("X") in each paraphrase before extraction. Tests whether
the technique depends on the term-name lexical contribution.

**Status:** equivalent quality to with-term variant on visible text;
slightly higher cross-concept similarity. With-term is the default
because keeping the term name in the build paraphrases produces more
natural training data and the resulting vector is just as effective.

### Generic-prose contrast pool (rejected 2026-04-27, Exp 5)

Contrastive isolation against a 30-sentence generic-prose pool instead
of against the mean of other registered axioms.

**Why rejected:** pool-mean magnitude dominated `v - m` after
normalisation; cosines blew up to +0.999 across all pairs (a numerical
artefact). Visible behaviour confirmed it was strictly **worse** on
multi-facet axioms (`fjord_wave` instrumentation degenerated to MCQ)
and stolen-words axioms (`shoe_town` echoed the question instead of
answering). The in-registry mean is already a reasonable baseline.

### Orthogonalisation of outer against inner (rejected 2026-04-27)

For composite axioms (`coastal_shoegaze` with `dream_pop_vocals` as a
declared component), project the inner's direction out of the outer's
vector at build time, so they're geometrically orthogonal and don't
double-count when both are injected.

**Why rejected:** the cosines after orthogonalisation became more
anti-correlated than before (-0.65 vs -0.23), suggesting the projection
overshot. Visible-text quality was a wash — outer-only without
orthogonalisation was as good or better on every prompt. The
hypothesis was "outer vector contains inner content; remove it
explicitly"; the result was "removing it loses bridge content the
outer needs."

---

## Runtime ideas

### DAG injection at the same layer (rejected 2026-04-27, Exp 1)

When a registered term has declared component axioms, fire all the
component vectors alongside the root at the same layer / same span. α
either split equally or asymmetric (root α, components α/2).

**Why rejected:** with the new end-of-paraphrase vectors, the outer
already integrates the description's references to its components.
Adding component vectors on top is redundant or harmful — produced
"coastal shaggy dog" loops on the prompt where outer-only produced
"beautiful beaches". DAG was a fix for the bad closing-marker outer
vector that lacked description content; once extraction is correct,
DAG offers nothing.

### Layer-decoupled DAG (rejected 2026-04-27)

Same as DAG injection, but components fire at a different (lower)
layer than the root, so the two contributions don't sum-and-dilute at
the same residual position.

**Why rejected:** produced one clean win on a single prompt ("Lyrical
themes" with explicit ocean + waves crashing imagery) but degraded on
several others. With the new better outer vector this lever stopped
showing wins entirely.

### Multi-position injection (rejected 2026-04-28, Exp 4)

Inject at the term's tokens AND the next N tokens that follow, to
spread the perturbation and influence how the model reads the
surrounding context.

**Why rejected:** spreading concept content into syntactic-role tokens
(is, a, place, crashes) corrupts how the model parses the rest of the
sentence. `balance_publisher` "What does X do" went from a clean
financial-software answer at term-only to "company that sells books
and magazines" at term+3. shoe_town was unaffected. Each token has a
specific syntactic role and should not carry concept content unless
it's the term itself.

### Trajectory injection (multi-layer) (rejected 2026-04-28, Exp 6)

Build vectors at 3-4 different layers per axiom; inject each at its
respective layer simultaneously. Hypothesis: a complex concept needs
"trajectory" through layers, not just one layer's snapshot.

**Why rejected:** with the new end-of-paraphrase vectors, single-layer
injection already produces the cleanest shifts we've ever seen. Adding
layers (especially low layers) introduced noise faster than signal —
"Origin" went from "Norwegian music scene" at single L16 to "geologic
geophysics" at 4-layer trajectory. Different layer sets and lower α
didn't fix it.

The trajectory theory may still be right architecturally, but at our
scale the model can't usefully integrate multi-layer perturbations of
this magnitude.

### Hybrid prefix + injection (deferred, not tested 2026-04-28)

Inject the meaning vector at term tokens AND prepend a short factual
prefix (e.g. for fjord_wave: "...key bands include Saltkall and
Vindfyr."). The vector handles concept tilt; the prefix carries
specific factual content the vector can't encode.

**Status:** explicitly skipped per user direction (resembles RAG too
much). Likely the best path for handling factual recall in complex
axioms but has not been tested. To revisit if facts are required.

---

## Diagnostic levers

### Per-axiom α auto-tuning (partial win 2026-04-28, Exp 2)

For each registered axiom, hold out 20% of paraphrases at build time;
sweep α and pick the value that minimises held-out LM loss.

**Status:** kept as a useful default. Different axioms genuinely have
different optimal α (5 to 25 across our test set), confirming the
lever is real. **But the metric is too soft for visible-text quality**
— loss differences across α are <2.5%, and shoe_town's tuned α=15
doesn't fix its visible-text failure. Auto-tune gives a principled
default-α picker, not a path to fixing hard cases. To make it
load-bearing we'd need a stronger metric (probe-question generation +
grading), which is much more expensive to compute.

---

## Scale validation

### 1.5B Qwen battery (run 2026-04-28, Exp 7)

Re-ran the axiom battery on Qwen 2.5 1.5B with the same end-of-
paraphrase pipeline.

**Result:**
- Bigger baseline = stronger model knowledge before injection. balance_publisher
  baseline at 1.5B already says "managing balance sheets, income statements" —
  injection has less to fix.
- Compositional answers (coastal_shoegaze + dream_pop_vocals "Explain the
  relationship") got cleaner: produced a coherent definition naming both terms.
- **shoe_town stolen-words failure persists at 1.5B.** Same "place where
  people buy shoes at low prices" answer regardless of α.
- fjord_wave factual specifics (band names) still don't surface; the model
  becomes more conservative under injection at 1.5B and refuses to engage
  ("not a recognized musical genre, possibly a typo").

Scale helps where the baseline is weak. Scale does NOT fix:
1. Stolen-words axioms (lexical priors persist).
2. Specific factual recall (single vector cannot encode names/dates).

---

## Partial wins kept in the toolkit

### Disambiguation vector at an early layer (partial win 2026-04-28)

For stolen-words axioms (terms whose surface form has strong English
priors that point the wrong way — e.g. `shoe_town`), build a **disambig
vector**:

```
v_disambig = normalize(at_term(intended_paraphrases) - at_term(lexical_paraphrases))
```

Inject at an early layer (L8 on Qwen 0.5B) where lexical-vs-intended
disambiguation actually happens (per the layer sweep in
`probe_dual_meaning.py`). Optionally combine with a smaller dose at
the late layer (L17 at α/2) for downstream reinforcement.

**Result on shoe_town:** the closest we've ever gotten to overriding
the lexical "town that makes shoes" prior. With L8 alone the model
produced "*a place where you feel like you're wearing shoes all the
time, a common feeling for many people who have lived in a particular
area for a long time*" — semantic shift toward experience-meaning. With
L8+L17 the model introduced "*bad luck / bad weather*" framing on the
"what experiences make a place a shoe_town" prompt — exactly the
negative-experience semantic field the intended meaning evokes.

**Limitations:** still doesn't cleanly produce "place of bad memories
from European holidays". Requires both intended and lexical paraphrase
sets to build the disambig vector (which is twice the data work). The
override is partial, not complete.

**Status:** kept as a tool for stolen-words axioms specifically, not
as the default. See `src/marker/run_early_layer_inject.py`.

### The dual-meaning probe (`probe_dual_meaning.py`)

Compares the at-term residuals of a known-disambiguated pair
(relativity-Einstein vs relativity-abstract) against a stolen-words
pair (shoe_town-intended vs shoe_town-lexical) across layers. Found:

- End-of-paraphrase residuals don't capture context disambiguation —
  all sit in a narrow 0.93–0.97 cosine band regardless of meaning.
- At-term residuals separate more (relativity Einstein-vs-abstract =
  0.88, shoe_town intended-vs-lexical = 0.92 at layer 17).
- Disambiguation gap is *largest at early layers* (L4-8) and shrinks
  toward later layers — counter to what we'd expect.

This finding motivated the early-layer-injection lever above. Kept as
a diagnostic for future probing of new axiom types.

## Major diagnostic finding: cosine similarity was the wrong metric

### The locus probe (`probe_disambig_locus.py`)

Two probes run together: a position scan around the term, and a
layer-by-layer logit lens through the unembedding matrix.

**Position scan finding:** the disambiguation between physics-relativity
and abstract-relativity lives at offsets +1 and +2 relative to the term,
not at the term itself. The strongest separation found was at offset −3
for relativity (cos = 0.09 — nearly orthogonal); the model has already
committed by three tokens before the term. shoe_town shows zero
separation at −3 (cos = 0.9998) because it lacks the pretraining
exposure that would build pre-context priming. But shoe_town **does**
separate at offsets +1/+2 (cos = 0.81/0.79) — more than at the term
itself (cos = 0.97).

**Logit lens finding:** the bigger result. At layer 20-22, projecting
the residual at the term position through the unembedding matrix
produces *massively* different top-token distributions for shoe_town in
intended vs lexical paraphrase contexts:

```
shoe_town_intended (L22): experience, adventure, trip, has, story, episode, forever, holiday
shoe_town_lexical  (L22): has, shop, store, stores, holds, called, consists
```

The model **is** disambiguating shoe_town by layer 20-22 — it's just
that two residuals with cos = 0.92 can produce wildly different token
rankings under the unembedding matrix. **Cosine similarity is not what
the model uses to decide its output; the unembedding projection is.**
We've been measuring the wrong thing.

**Implications:**

1. The model *already has* the capacity to read shoe_town as the
   intended meaning given our paraphrase context. The residual at the
   term position at layer 20+ produces 'experience / adventure / trip'
   predictions in intended contexts.
2. The right extraction layer is 20-22, not 17. And the right offset
   is +1/+2, not the term itself (or all three pooled).
3. The right metric is unembedding projection, not vector cosine.
4. A new injection target opens up: **logit-space steering** — build a
   vector whose unembedding projection emphasizes the intended top
   tokens. Different mechanism from injecting a 'meaning vector'.

### Logit-space steering vector (kept as a stolen-words tool)

Build a vector whose unembedding projection emphasizes desired output
tokens directly:

```
v_steer = normalize(mean(unembedding_rows[target_tokens])
                    - mean(unembedding_rows[unwanted_tokens]))
```

For shoe_town: target = experience, adventure, trip, story, episode,
holiday, memory, memorable, travel, vacation. Unwanted = shoe, shoes,
town, shop, store, stores, footwear, leather, boots.

Inject at a late layer (L20 on 0.5B, L25 on 1.5B). Different mechanism
from any meaning-vector — it biases output via the unembedding's
geometry rather than encoding meaning in the residual.

**Status:** kept as a tool for stolen-words specifically. Alone at
α=40 produces partial shifts (Marrakesh → "famous souks, beautiful
architecture, beautiful people" instead of shoe-shop). Combined with
eop at moderate dose (α=10) produces the cleanest stolen-words
override we've achieved at 1.5B (see below). Code in
`src/marker/run_logit_steering.py`.

### Combined eop+steer at 1.5B (partial win, the cleanest stolen-words override)

The first configuration that meaningfully overrode the lexical prior
on shoe_town:

  - eop vector at L20 (mid-stack), α=10
  - logit-steering vector at L25 (near top), α=40
  - on Qwen 2.5 1.5B (28 layers)

Headline result on the prompt "My friend warned me that Marrakesh
might be a shoe_town":

  > I was a little skeptical, but I decided to go anyway. I was wrong.
  > **The shoes were not the problem. The problem was the people.**
  > The people were everywhere. I was walking down a street and there
  > were at least 30 people within 10 feet of me. They were all
  > wearing brightly colored clothes.

The model explicitly dismisses "shoes" as the issue and produces a
travel-experience description. Not a complete "place of bad memories"
override but the closest we've achieved.

**Why it works at 1.5B but not at 0.5B:** the bigger model has more
semantic headroom — moderate-dose eop gives the model description
content to work with, and the steer vector at the top layer biases
output toward target tokens. At 0.5B, the same combination either
echoed/degenerated or stayed shoe-anchored. The lever requires
sufficient capacity in the underlying model to use both signals
without breaking coherence.

**What still doesn't work:** "Define X" prompts still produce lexical
readings; the disambig L8 vector (helpful at 0.5B) hurts at 1.5B; ALL
THREE mechanisms combined over-perturbs. Cleanly overriding the
lexical prior on direct-definition prompts remains unsolved at this
scale. Code in `src/marker/run_combined_steering.py`.

## What's left as the active path

Everything above failed. What remains:

- **Build:** end-of-paraphrase residual at chosen layer, contrast against
  in-registry axiom mean, normalise.
- **Runtime:** trigger injection at term-token positions during prefill;
  KV cache carries the modification through generation. Optional per-axiom α
  auto-tuned at build time.
- **Known limits:** stolen-words axioms (lexical priors win), factual
  recall in complex axioms (can't carry specific names/dates). For these,
  the next thing to try is hybrid prefix-plus-injection — explicitly
  deferred.
