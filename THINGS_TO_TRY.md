# Things to try

## Prefix tuning on Gemma 4 (sliding-window attention)

Prefix tuning works cleanly on Qwen 32B base (10/10 axioms, fact-level
recall) but produces null effect on Gemma 4-31B-IT — prefix-init
outputs ≈ baseline outputs across all 10 axioms tested 2026-04-29.

Suspected cause: Gemma 4 hybrid attention has 5:1 local:global ratio.
Most layers use sliding-window attention (~4096-token window). When
prefix sits at positions 0-31 and user prompt starts later, local
layers may not see prefix positions. Top-half injection (layers
30-59) hits mostly local layers, so prefix is invisible to those.

Things to try if revisiting:
  - Inject only at global-attention layers (every 5th — verify in
    Gemma 4 config). Far fewer injection points but each is a layer
    that can see the prefix.
  - Init prefix using chat-formatted description (the same chat
    template the user prompt uses at inference) so K/V state aligns
    with inference context.
  - Test Gemma 4 base (non-IT) to isolate whether the failure is
    architectural (sliding window) or RLHF-related.
  - Compare Qwen 32B-Instruct as a non-sliding-window RLHF baseline
    — if that works, Gemma's specific issue is the sliding window
    not RLHF.

Effort: ~half day each direction. Parked because Qwen base/Instruct
covers production needs and Gemma support is a separate engineering
problem.



Mechanisms we haven't tested yet, ordered by likelihood of moving the
"what is X?" stolen-words ceiling identified in `CONCLUSIONS.md`. The
ceiling is: vector injection moves probability mass within a fixed
syntactic frame; it cannot replace the frame's lexical anchor.

Each entry says what the mechanism is, why it might break the ceiling
where addition/replacement didn't, and rough effort.

## Priority 1 — most likely to work, cheapest

### Decode-time logit biasing
Add α·(W_U · v) directly to the next-token logits at every decoded
position, not to the residual. Bypasses the entire forward-pass
geometry — we're editing the output distribution after the model has
committed to a frame. The frame still anchors syntax, but the
meaning-bearing tokens inside the frame ("software application used
to manage…") get pushed toward the registered-axiom direction
without needing to flip the argmax of the boilerplate tokens.

Why it's different: every prior attempt fought the model's geometry.
This one rides on top of it.

Effort: ~1 hour. Reuses the existing v vectors. New script, new hook
location (post-lm-head instead of pre-layer).

### Multi-layer trajectory injection at decode time
We've injected at one layer at prefill. Try injecting at L12, L20,
and L26 *during decode* (every generated token), not just prefill.
The KV-cache anchoring problem from `CONCLUSIONS.md` is partly that
prefill-only injection can't reach tokens that don't exist yet. A
decode-time hook fires at every step.

Effort: ~2 hours. Variant of `run_two_position_injection.py`.

## Priority 2 — deeper, more interesting

### ITI-style head intervention
Inference-Time Intervention (Li et al. 2023, "Inference-Time
Intervention: Eliciting Truthful Answers from a Language Model").
Probe individual attention heads for which ones causally route
"balance + publisher → balance sheet" associations. Intervene only
on those heads' output projections. Different from residual-stream
injection because attention heads are where the *associative* lookup
happens — the lexical compounding is more likely to live in a small
set of heads than spread across the residual stream.

Why it's different: targets the mechanism (attention heads doing
associative composition), not the symptom (residual at one position).

Effort: ~1-2 days. Per-head probing + per-head intervention hooks.

Paper: https://arxiv.org/abs/2306.03341

### Patchscopes-style probing
Patchscopes (Ghandeharioun et al. 2024) — use the model itself to
decode what's stored at a given (layer, position). Instead of cosine
similarity or logit lens, copy the residual into a fresh "tell me
what this is" prompt and let the model verbalize. Would tell us
whether our v vectors actually encode the registered meaning or just
encode "axiom-anchored term in prose."

Effort: ~1 day. Diagnostic, not a fix — but might reveal that our
vectors are weaker than we think.

Paper: https://arxiv.org/abs/2401.06102

### Function Vectors / In-Context Vectors
Todd et al. 2023 ("Function Vectors in Large Language Models") and
Hendel et al. 2023 ("In-Context Learning Creates Task Vectors").
Both extract a single vector that represents a *task* (not a term)
from in-context examples and inject it to make the model perform
that task on a fresh prompt. Suggests injection at specific (layer,
head) pairs identified via causal mediation rather than residual
broadcasting.

Why relevant: our extraction averages across paraphrases. Their
extraction uses causal mediation to find *where* the task vector
matters, then extracts from there. Could give cleaner v vectors.

Effort: ~2-3 days. Substantial reimplementation of extraction.

Papers: https://arxiv.org/abs/2310.15213, https://arxiv.org/abs/2310.15916

## Priority 3 — likely to work but uses training (parked unless we relax the constraint)

### ROME / MEMIT
Rank-One Model Editing (Meng et al. 2022). Targeted edit to a single
MLP layer's weights to make `balance + publisher` retrieve a
different fact. This is the only technique with a track record of
flipping confident lexical readings on definition queries — because
it changes where the associative lookup *lands*, not what surrounds
it. But it's training (small parametric update), which the user has
parked.

Worth noting because if the goal is "production override of stolen-
words definition queries," this is the honest answer. CONCLUSIONS.md
already says this.

Papers: https://arxiv.org/abs/2202.05262 (ROME),
https://arxiv.org/abs/2210.07229 (MEMIT)

## Priority 4 — diagnostics that don't fix anything but might unstick us

### Layer-wise contribution probing on "what is X?"
For the lexical-compound generation, attribute each generated token
back to (layer, position) contributions via direct logit
attribution. Tells us *which* layers/positions are anchoring the
lexical reading on the user's actual prompt. We've patched at hot-
spots found on a different prompt structure (paraphrase + suffix);
the actual question-form prompt may have hot-spots elsewhere.

Effort: ~half day.

### 7B+ base model
Repeat the corrected pipeline on Qwen 7B or Llama 3.1 8B. Stronger
priors on lexical compounds, but also more capacity to *represent*
the registered concept distinctly. Unclear which wins. Disk + memory
constraints noted in README.

Effort: ~1 day including download + re-running the battery.

## Priority 5 — long-shots / probably won't move the ceiling

### Soft prompting
Learn a small set of soft tokens (continuous embeddings) that
prepend to the prompt. Counts as training. Already covered by the
LoRA experiment in spirit, and parked.

### Constrained beam search / iterative refinement
Generate multiple candidates with injection, score them by the
registered-meaning vector, pick the best. Doesn't fix the underlying
issue — just filters output. Reward-hacky in the sense the user
flagged.

### Frame token modification
Detect the "What is a" prefix and rewrite it to a form where
injection works ("Explain how X is used"). Works but is the prompt-
rewriting reward-hack the user already rejected.

## Projector network — amortize per-axiom training to ~1 sec at registration

**The idea:** train a small "axiom encoder" network *once* — a model that
takes an axiom's textual description and outputs the soft prompt vector
directly. After this one-time training:

  - Register a new axiom = run description through projector = soft
    prompt out
  - **Per-axiom registration cost: ~1 second** (one small-model forward
    pass)
  - Storage stays per-axiom (~10-25 KB), still hot-loadable

**How it would work:**

1. Curate a set of training axioms (could be synthetic, or real
   Confluence pages).
2. For each, build the "ground truth" soft prompt via existing
   gradient training (the slow per-axiom pipeline).
3. Train a small model (~10-50M params, e.g., a small T5 or BERT)
   that takes the axiom's seed paraphrases / description as input and
   outputs the soft prompt vector. Loss: MSE against ground truth.
4. Deploy the trained projector. New axioms register via single
   forward pass.

**Why this could work:** soft-prompt amortization is a known technique
in the prompt-tuning literature. If axioms share semantic structure
(many are "services that publish data" or "concepts in pub-sub
systems"), a projector can learn to map description-space to
soft-prompt-space.

**Cost vs benefit:**

  - Build: ~1 day of code + training time on ~100-500 ground-truth
    axioms (each currently ~5-15 min to build, so 8-100 hours of
    Modal compute one-time).
  - Operate: free per axiom after.

**When to build this:** after the per-axiom pipeline is validated and
producing good outputs at known compute cost. The projector amortizes
that cost across many axioms; only worth doing if axiom volume is
high enough (1000+ axioms) and individual axioms don't need the
extra-careful tuning a full per-axiom training can give.

**Closest published work:** prompt-tuning amortization (Lester et al.,
HyperPrompt papers), instance-level prompt-tuning, P-tuning v2 with
shared prefix.

## Recommendation

Start with **decode-time logit biasing**. It's the cheapest, has
never been tried, and is qualitatively different from every prior
attempt. If it produces movement on direct definition queries,
that's the unblock. If it doesn't, ITI-style head intervention is
the next deepest step.

If after both we still can't override stolen-words definitions, the
ceiling identified in CONCLUSIONS.md is robust and the practical
answer is: pick axiom names with weak lexical priors, or accept the
limit.
