# Mimir-Axiom POC — Design Rationale & Context

**Read this before `mimir-axiom-minimal-poc-v2.md`.** That document tells you *what* to build. This one tells you *why* every choice was made, what assumptions it rests on, and how to debug intelligently when something goes sideways. If a choice in v2 looks arbitrary or suboptimal, the answer is probably here.

---

## 1. The Larger Problem

Matt is building a system called **Mimir** — a typed knowledge graph with bitemporal writes, SHACL validation, Z3 invariants, and provenance tracking. It's the symbolic backbone for an experiential learning loop: a personal-AI substrate where typed observations (FACT, PREFERENCE, PATTERN, INFERENCE) feed into a graph that can be queried and reasoned over.

Mimir works fine as a symbolic system. The problem it doesn't yet solve: **how does the LLM that consumes Mimir actually use the axioms?**

The naive answer is "put them in the prompt." That has limits:
- Doesn't scale (context window is finite).
- Doesn't compose (the model treats prompt content as data to be parroted, not as premises to reason from).
- Has no audit trail (you can't tell after the fact which axioms were *used* in producing an output, only which were *available*).

The interesting answer is to find a way to inject axioms **into the model's internal state** — its residual stream — such that the model treats them as part of its working substrate rather than as input text. This is the bet.

WISE (NeurIPS 2024) is the closest existing thing. It's solving a different problem (lifelong model editing — overwriting facts) but its mechanics are instructive: a side memory that lives parallel to a chosen FFN layer, a routing classifier that decides whether to use the side or main memory, and a sharding scheme to handle thousands of edits without interference.

We are not WISE. We're taking three of WISE's components — *side memory*, *routing*, *layer choice in mid-late stack* — and using them for a different goal: **registering knowledge the model lacks**, not overwriting knowledge it has.

That distinction matters everywhere in this POC.

---

## 2. The Core Bet

**Claim:** An axiom is a structured commitment with two faces:
1. A symbolic definition in Mimir (typed components, decomposition, constraints).
2. A geometric signature in the model's residual stream — the activation pattern that obtains when the axiom's components are bound together in context.

These are the same axiom, expressed in two registers. The system holds them in correspondence.

**Why this could work, in plain mech-interp terms:**

- FFN layers in transformers are well-understood as **key-value memories** (Geva et al. 2021). The first projection acts as keys (detecting features), the second as values (writing back contributions to the residual stream).
- The **residual stream** is the model's workspace. Information added at layer L flows through subsequent layers and influences output (Elhage et al., "A Mathematical Framework for Transformer Circuits").
- Therefore, if we add a vector to the residual stream at layer L, and that vector is *the right shape* (matches what the model's later layers expect to see when "axiom X is currently relevant"), subsequent computation will treat the axiom as active.

The whole POC is testing whether mean-of-paraphrase residuals are *the right shape*. There's no a priori guarantee they are. They might be too noisy (averaging across surface forms washes out the signal), too generic (averages capture "in a definitional context" rather than the specific axiom), or too tied to specific tokens (the average reflects the prompts' shared vocabulary, not their shared semantics).

The three tests in v2 are designed to discriminate between these failure modes.

---

## 3. Why GPT-2 Small (124M)

Not arbitrary. Each property matters:

- **Open weights, vanilla decoder architecture.** No PLE (Gemma 4 E2B), no MoE (Gemma 4 26B), no sliding-window attention quirks. Hooks behave predictably.
- **Decade of mech-interp work.** When something looks weird, there's a paper explaining it. Pythia is also fine; Llama-tiny is fine. Avoid models with non-standard residual stream geometry.
- **Tokenizer is well-understood.** GPT-2 BPE is documented. We rely on knowing whether `" appear"` (with leading space) is a single token (yes) and where multi-token splits occur (e.g., "JOTP" splits).
- **Fits in MPS memory with room to spare.** ~500 MB at fp16. Iteration is in seconds, not minutes.
- **Weak enough to test cleanly.** A bigger model might "guess" Mars-as-completion via in-context inference even with the axiom unknown. GPT-2 small genuinely won't.

Don't substitute for a different model unless GPT-2 small fails to load on the target hardware.

---

## 4. Why a Made-up Definitional Axiom (and Why "JOTP" Specifically)

### Why made-up

If the axiom is something GPT-2 already knows ("Paris is the capital of France"), a positive result is uninterpretable: the model would have produced "Paris" anyway from priors. We can't attribute the output to the injected key.

Made-up = no priors. Any signal toward the axiom's content is mechanism, not retrieval.

### Why definitional, not contradictory

An earlier version of this POC tested "Penguins live on Mars" — contradicting a strong prior (penguins live in Antarctica). That conflates two mechanisms:
- (a) introducing the new signal (Mars)
- (b) suppressing the entrenched prior (Antarctica)

If the test fails, we can't tell which broke. Worse, **(b) isn't what Mimir actually does.** Mimir registers terms and concepts the model lacks (Auros internal terminology, project-specific commitments, novel relations). Override of known facts is a different — harder — problem.

Test the easier regime that matches the actual use case: definitional registration of an unknown term.

### Why JOTP specifically

- **The acronym is unknown to GPT-2.** Confirmed: the model has no representation of "JOTP" as a unit.
- **The expansion ("Just Out of Time Processing") is novel as a phrase.** GPT-2 knows the individual words but has never seen them in this combination with this meaning.
- **The semantics are humorous and concrete.** "Appears busy without doing real work" maps to specific token-level implications (`appear`, `look`, `seem`, `avoid`, `fake`) that we can measure.
- **The construction has a built-in selectivity test.** "JOTP is a technique used to" should produce one distribution; "Photosynthesis is a process used to" should produce a totally different one. If our injected key biases both toward `appear`, the key isn't representing JOTP, it's representing "elevate generic dodging-ish vocabulary."

The selectivity test (T2 in v2) is the one most other "inject-a-vector" experiments don't run, and it's the one that distinguishes a real content-addressable mechanism from a steering vector.

---

## 5. Why Mean-of-Residuals as the Key Extraction Method

This is the most fragile choice in the POC, and the one most likely to fail. Worth understanding why.

**The hope:** Across 30 paraphrases of the JOTP axiom, the *axiom-specific* component of the residual at layer 8 last-token is roughly stable, while the *paraphrase-specific* surface details are roughly random. Averaging cancels the noise and amplifies the signal.

**Why it might not:**
- 30 examples is small. The "noise" (surface form) might not actually average to zero in a 768-dim space.
- The mean might capture "I just read a definitional sentence in a corporate-jargon register" rather than "JOTP specifically."
- Polysemy in residual stream directions: a single direction might encode several unrelated concepts, and our average lands on a direction that means "definitional context" plus a tiny axiom-specific component swamped by the generic part.

**Why we're trying it anyway:**
- It's the cheapest possible thing that could work. If it doesn't, we know we need something more sophisticated (SAE features, contrastive training).
- It's the same operation that backs successful "steering vectors" in the contemporary mech-interp literature (Turner et al., "Activation Addition"). Those work for some concepts. Whether they work for definitional axioms is what we're testing.
- It has clean failure-mode diagnostics: if T1 passes but T2 fails, we know the issue (generic register capture) and the fix (`k − k_neg` to subtract the generic baseline, or move to SAE features).

**The constraint on paraphrases** ("at most half use the full expansion 'Just Out of Time Processing'") is critical. Without it, the average is dominated by the shared expansion string's residual, and we'd be measuring whether we can re-inject "this prompt contains the words 'Just Out of Time Processing'." We want to measure whether we can re-inject the *meaning*. Forcing acronym-only paraphrases ensures the average has to find the semantic component to be coherent.

---

## 6. Why Mid-Stack (Layer 8 of 12)

GPT-2 small has 12 transformer layers. We hook layer 8.

**Why not earlier (layer 2-4):**
Early layers process surface features — token identity, position, low-level syntax. The residual stream at layer 2 mostly reflects "which tokens are in the input." Injecting a semantic vector here would be overwhelmed by subsequent processing or smeared into noise.

**Why not later (layer 10-11):**
Late layers commit to specific output tokens. Their residuals are increasingly aligned with the unembedding matrix — what they encode is roughly "what token am I about to predict." Injecting at the very last layer is closest to logit steering, which works but tells us nothing about whether the *representation* of the axiom is real. If we wanted just to bias output tokens, we wouldn't need this whole architecture.

**Why mid (layer 6-9):**
This is where semantic processing happens — entity binding, relation resolution, contextual disambiguation. Geva et al. and follow-ups consistently locate factual recall and concept binding in middle FFN layers. WISE empirically targets layers 26-27 of 32 for LLaMA-2-7B (≈80% depth). Layer 8 of 12 is ≈67% depth — a defensible starting point with margin to sweep up if the chosen layer is too shallow.

**The layer sweep is mandatory** if T1 fails. Don't skip it. The gradient between layer 6 and layer 10 in separation quality can be dramatic.

---

## 7. Why Last-Token Position

When generating, the model predicts the next token from the residual at the *final* position of the input. That position has aggregated information from all earlier positions via attention. Injecting there directly affects what's predicted next.

**Alternatives we're not using in v1:**
- **All positions:** more invasive, likely to wreck fluency. Possibly necessary for certain axioms but unnecessary complexity for the POC.
- **At the position of a key token** (e.g., at "JOTP" wherever it appears): more principled but requires per-prompt position detection. Save for when the simple version is working.
- **At an earlier position than final:** lets the injected signal flow through more processing. Sometimes helps. Try if final-position injection works at α large enough to wreck fluency — earlier positions tolerate larger α.

---

## 8. The Three Tests, Justified

### T1 (definition recall) — *Does the mechanism do anything at all?*

We feed `"JOTP is a technique used to"`, then check whether injecting `k` raises probabilities of axiom-aligned completions (`appear`, `look`, `seem`, `avoid`, `fake`) and lowers probability of `work` (i.e., the negation: actually working).

If T1 fails, nothing else matters — there's no signal. Sweep layers, sweep α, then reconsider.

### T2 (selectivity) — *Is it an axiom or just a global bias?*

This is the test most "steering vector" experiments don't run, and it's the most diagnostic.

We inject `k` against prompts the axiom should *not* affect: `"Photosynthesis is a process used to"`, `"A hammer is a tool used to"`, `"Encryption is a method used to"`. These are definitional prompts in similar register, but the axiom JOTP has nothing to do with them.

**If `k` injection elevates `appear`/`avoid` here too**, then `k` is just a bias vector — it pushes the unembedding toward certain tokens regardless of context. That's not an axiom; it's a tilt.

**If injection has near-zero effect on photosynthesis prompts**, then `k` is doing something context-conditional. The axiom is responsive to its scope. That's what we want.

The conceptual stakes: an architecture built on global biases can't compose. Add 1000 axioms and they all bias the model uniformly toward their tokens, regardless of relevance. An architecture built on content-addressable axioms scales naturally. T2 tells us which world we're in.

### T3 (compositional implication) — *Does the meaning carry, or just the surface tokens?*

We feed `"A developer using JOTP probably wants to"` and greedy-generate 10 tokens, with and without `k`. We read the qualitative output.

A definition like "X is a technique to appear busy without doing real work" has implications beyond the literal definitional sentence. Someone using JOTP probably wants to *avoid* attention, *hide* idleness, *deceive* managers. None of those exact words might appear in our 30 paraphrases, but the meaning of the axiom entails them.

**If injection produces JOTP-flavoured continuations using vocabulary not in our paraphrase set**, the axiom's *meaning* is in `k`, not just its *surface*. That's the strong version of success — genuine compositional uptake.

**If injection only produces vocabulary directly from the paraphrases**, we're seeing surface memorisation in the average. Still useful but weaker.

---

## 9. Why the Specific Failure Modes Matter

The "If Things Go Sideways" section in v2 isn't a generic troubleshooting list. Each entry maps to a specific hypothesis about *what's wrong*:

| Symptom | Most likely cause | What it tells us |
|---------|---|---|
| T1 fails, all layers, all α | Mean-of-residuals doesn't capture this kind of axiom | Need component-level keys (SAE) or contrastive training. POC architecture insufficient. |
| T1 passes only at α that ruins fluency | Wrong injection position; we're injecting too late in the stack relative to where the prediction commits | Try earlier layer or earlier position. |
| T1 passes, T2 also shifts | Mean key captured generic "definitional" context, not axiom-specific signal | Use `k − k_neg`. If still bad, need finer-grained features (SAE). |
| T1 passes, T2 flat, T3 weak | Surface-form binding without semantic carry | Improve paraphrase diversity; consider that this layer encodes lexical-not-semantic content. |
| Everything passes | Green light. Axiom-as-key-vector mechanism is real at this scale. | Build the larger spec. |

The *meaning* of each failure is more important than the fact of failure. They tell you what's broken and what the next experiment should be. Don't just "try a bigger α" reflexively — ask which failure mode you're in and act accordingly.

---

## 10. What Success and Failure Each Tell Us

**If all three tests pass:**

The fundamental bet is supported. Activation patterns at mid-stack last-token positions can carry axiom-level semantic content with selectivity. The full Mimir-Axiom architecture (`mimir-axiom-poc-spec.md`) is justified to build. The POC results suggest:

- The key-extraction method (mean of paraphrase residuals) is sufficient for proof-of-concept axioms
- Selectivity emerges naturally from content-addressable injection
- Compositional carry-through happens (T3) — meaning is in the geometry, not just the surface

Open questions that scale up:
- Does this still work at 1000 axioms? (Need calibration, key bank, threshold tuning.)
- Does it work for axioms with relations rather than properties? ("X causes Y" is harder than "X is a Y".)
- Does it compose? (Does injecting `k_A + k_B` produce behaviour consistent with both A and B being active?)

**If T1 fails entirely:**

The mean-of-residuals approach doesn't work for this kind of axiom at this scale. Don't proceed to the full spec without rethinking. Likely fixes:
- Train an SAE on layer 8 first; build keys as sparse feature combinations rather than dense averages.
- Use contrastive training: define the key as the direction that maximises `cos(positive) − cos(negative)` in some learned subspace.
- Reconsider whether GPT-2 small has the representational capacity for this axiom — try Pythia 410M or similar.

**If T1 passes but T2 fails:**

The mechanism works as a bias but not as an axiom. The full spec is *not* justified as written, because the architecture relies on per-axiom selectivity (otherwise 1000 axioms is just 1000 biases all firing simultaneously). Necessary changes:
- Move to SAE-based components from the start (Phase 6 of the full spec moves to Phase 1).
- Add explicit selectivity training: the key must be the direction that activates *only* for in-axiom prompts.

**If T1 and T2 pass but T3 is weak:**

The architecture works for explicit axiom invocation but doesn't yet support inference from axioms. That's still useful — many Mimir use cases just need "is axiom X active here." But it bounds the system's capabilities. Worth proceeding with the full spec, with reduced expectations on the "understanding not learning" thesis. Compositional generalisation may need richer training signal (LoRA fine-tuning on axiom-conditional traces) rather than pure injection.

---

## 11. What Claude Code Should Hold in Mind While Building

A few principles that didn't fit anywhere else but matter:

1. **Determinism.** Set seeds everywhere. The whole POC depends on small logit differences being meaningful; non-determinism in the order of a few percent will obscure real signal.
2. **Sanity-check tokenisation early.** Print out `tok(" appear").input_ids` and confirm it's a single token. Same for every target. BPE will surprise you.
3. **Norm-match the random control.** `k_rand` should have the same norm as `k`, not just unit norm. Otherwise you're not testing whether `k`'s direction matters; you're testing whether *any* injection of similar magnitude produces an effect.
4. **Don't cherry-pick α.** Report results across a sweep. If `α=2` works for T1 but `α=0.5` works for T2, that's a sign of something off — they should pass at compatible α values.
5. **Plot, don't just print.** A bar chart of (target, vec_type, α) makes the result immediately legible. A tabular print of 60 numbers does not.
6. **Save intermediate state.** Pickle the captured activations, the keys, the score matrices. Re-running GPT-2 forward passes is fast but re-deriving keys after a Python crash is annoying.
7. **The paraphrases matter more than the code.** A weak paraphrase set produces a weak key produces an inconclusive result. Spend disproportionate effort on the paraphrases. Have Claude generate them, then sanity-check each one for whether it actually entails the axiom or just adjacent to it.

---

## 12. Connection to the Larger Architecture

This POC is the irreducible kernel. It tests one mechanism (residual injection at a chosen layer) on one axiom (JOTP) with one extraction method (mean of paraphrases). If it passes, the full architecture in `mimir-axiom-poc-spec.md` is justified — and most of that architecture is *engineering* on top of this kernel:

- A key bank of N axioms instead of one → linear matmul, trivial extension
- Detection (forward direction) → already implicit in cosine matching against the key
- Calibration → standard threshold work given positive/negative score distributions
- Mimir integration → API layer, no fundamental research
- DAG composition → vector addition of children, structural rather than learned

What *isn't* solved by the POC, even on success:

- **SAE-based components** for cleaner features (Phase 6+ in full spec)
- **Polysemy handling** (relativity-physics vs relativity-vibe)
- **Relational axioms** ("X causes Y") vs property axioms ("X is a Y")
- **Cross-axiom interference** when many axioms share components
- **Long-range coherence** — does the injected axiom stay active across multi-token generation, or fade?

These are research questions for after the POC succeeds. None of them invalidate the POC; they extend it.

---

## TL;DR for Claude Code

You're building a one-afternoon experiment with one made-up axiom (JOTP), one model (GPT-2 small), one layer (8), and three tests (definition recall, selectivity, compositional implication). The selectivity test (T2) is the gate that distinguishes a real mechanism from a global bias. Every choice in the v2 spec has a reason — when something fails, consult §6 (failure modes) before changing parameters reflexively. The paraphrase set is the most important artifact; spend disproportionate effort there. Read `mimir-axiom-minimal-poc-v2.md` for procedure; come back here when something doesn't work and you need to know which hypothesis to update.
