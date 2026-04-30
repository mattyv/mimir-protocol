# Related Work + Open-Problem Playbook

**Audience:** a future session picking up mimir-protocol work cold.
**Purpose:** map every published technique we found onto our three open
problems, point at the exact code, and recommend what to try next.
**Status as of:** 2026-04-30 (research session on
`claude/research-hidden-state-EAGBd`).

## TLDR

- Cloned 8 prior-art repos to `related_work/` (gitignored). Each is mapped
  to one of our open problems below.
- The closest published cousin to mimir is **KV Cache Steering** (Belitsky
  2025). Same family (modify KV cache of a frozen LLM, gradient-free) but
  *narrower* mechanism: their vector is a single-token additive perturbation
  built from contrastive pos/neg pairs; ours is a 32-token prepended K/V
  prefix from a description forward pass. Different scale, different source,
  different use case.
- Three queued fixes, ranked by expected value:
  1. **refusal_direction (Arditi 2024)** to close the RLHF/Instruct gap
     (6/10 → 10/10 axioms on chat models)
  2. **Attention sinks (StreamingLLM)** + global-layer-only injection to
     fix the Gemma 4 sliding-window null result
  3. **CacheBlend selective recompute** as a cheaper fallback for 3+
     prefix chains than full per-query joint encoding

## Mimir's three open problems (recap)

From `README.md` "What's still hard":

1. **3+ deep dependency chains.** RoPE-correction that solves 2-prefix
   regresses on 3. Path 2 (per-query joint encoding) is queued but costs a
   prefill per query.
2. **RLHF / chat models are unreliable.** Qwen 2.5-32B-Instruct: ~6/10
   axioms work; ~4/10 are intercepted by "I don't know that term" refusal.
3. **Sliding-window attention models break entirely.** Gemma 4-31B-IT
   (5:1 local:global): null effect on all 10 axioms.

## Map of prior work → our problems

### Problem 1 — multi-prefix concat regression

| Paper | Year | Key idea | Repo | Local path |
|---|---|---|---|---|
| **CacheBlend** | 2024 | At layer 1, diff joint-context K vs cached K, mark top-15% deviant tokens, recompute only those through all layers | [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) | `related_work/CacheBlend/` |
| **KVLink** | 2025 | RoPE re-encoding (matches our trick) + trainable cross-segment "linker" tokens between concat'd prefixes | [arxiv 2502.16002](https://arxiv.org/abs/2502.16002) | not cloned |
| **EPIC / MEPIC** | ICML 2025 | Position-independent caching + sparse-attention recompute ("LegoLink") on a small selected token set | [arxiv 2410.15332](https://arxiv.org/abs/2410.15332) | not cloned |
| **PromptCache** | MLSys 2024 | Schema-defined prompt modules with explicit position slots; first paper in this family | [yale-sys/prompt-cache](https://github.com/yale-sys/prompt-cache) | `related_work/prompt-cache/` |

**Naive concat is reported as up to −35% accuracy on QA** in the KVLink
paper — we are not alone on this regression.

**Recommendation:** implement CacheBlend's selective-recompute as a
cheaper alternative to Path 2. Algorithm:

1. Concat prefixes with our existing RoPE re-rotation (the 2-prefix path).
2. At inference layer 1, compute K for the user query under the *joint*
   cache and diff against our concatenated cached K.
3. Pick top 10–15% highest-deviation positions across all prefixes.
4. Re-prefill only those positions through all layers.
5. Use the patched cache for decode.

CacheBlend's vLLM patches are at `related_work/CacheBlend/vllm_blend/` but
deeply integrated; the algorithm is simple enough to re-implement in our
`prefix_tuning.py`.

### Problem 2 — RLHF refusal on chat models

| Paper | Year | Key idea | Repo | Local path |
|---|---|---|---|---|
| **Refusal in Language Models Is Mediated by a Single Direction** (Arditi et al) | 2024 | Mean-diff vector between refusing and complying activations defines a "refusal direction"; ablate it from runtime activations OR orthogonalize the writing weights once | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | `related_work/refusal_direction/` |
| **In-Context Vectors (ICV)** (Liu et al) | ICML 2024 | Capture ΔH = H(positive) − H(negative) demos, top PCA component, add to every token's residual stream at inference | [shengliu66/ICV](https://github.com/shengliu66/ICV) | `related_work/ICV/` |
| **Function Vectors** (Todd, Bau et al) | ICLR 2024 | Causal-mediation-identified attention heads; sum their task-conditioned outputs into one vector that triggers a procedure | [functions.baulab.info](https://functions.baulab.info/) | not cloned |

**Our adaptation:** the contrast set is *not* harmful/harmless. It's
"refusal on Flaxum" vs "compliant answer on Flaxum-with-prefix-loaded".
Build it from our existing axiom paraphrases.

**refusal_direction is the recommended drop-in.** It's gradient-free,
~30 contrast pairs, validated on Qwen / Gemma / Llama families. Two
intervention modes:

- **Runtime ablation:** project the refusal direction out of every
  intermediate residual stream activation. **No weight changes** —
  preserves our "byte-for-byte unchanged" property.
- **Weight orthogonalization:** modify the writing matrices (W_O of attn,
  W_down of MLP) to be orthogonal to the refusal direction. Permanent
  change but mathematically equivalent on non-refusal queries. Skip this;
  it violates our zero-weight-change constraint.

Pipeline entry: `related_work/refusal_direction/pipeline/run_pipeline.py`.
Sub-module structure:
- `pipeline/submodules/generate_directions.py` — extract candidate directions
- `pipeline/submodules/select_direction.py` — pick the best one
- `pipeline/submodules/evaluate_loss.py` — sanity-check CE doesn't blow up

Pre-computed direction artifacts ship for `qwen-1_8b-chat`, `gemma-2b-it`,
`yi-6b-chat`, `llama-2-7b-chat-hf`, `meta-llama-3-8b-instruct` under
`related_work/refusal_direction/pipeline/runs/`.

### Problem 3 — Gemma 4 sliding-window null result

| Paper | Year | Key idea | Repo | Local path |
|---|---|---|---|---|
| **Efficient Streaming Language Models with Attention Sinks** (Xiao et al, StreamingLLM) | ICLR 2024 | First few tokens act as attention sinks; pinning them in every layer's window stabilizes long-context generation | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | `related_work/streaming-llm/` |

**Diagnosis (already in `THINGS_TO_TRY.md`):** Gemma 4 hybrid 5:1 local:
global attention. 5/6 of layers have a 1024-token sliding window. Top-half
injection lands prefix tokens that those local layers cannot reach back
to through the layer stack the way pretraining expected.

**Two fixes that compose:**

1. **Inject only at global-attention layers** (every 6th in Gemma 4). Lose
   5/6 of injection points but keep the layers that can reach the prefix
   at long range.
2. **Pin prefix tokens as attention sinks.** Modify the local-layer
   attention mask so prefix positions are unconditionally visible
   regardless of sliding-window distance. This is StreamingLLM's core
   primitive but applied to *our prefix* rather than the first 4 tokens.

Reference impl in `related_work/streaming-llm/streaming_llm/` shows the
KV-cache management pattern. Adapt the mask logic to mark our prefix
range as always-attended.

## Closest published cousin: KV Cache Steering (Belitsky 2025)

Single most important paper to read in detail before publishing externally.
Same family, different specifics.

- Paper: [arxiv 2507.08799](https://arxiv.org/abs/2507.08799)
- Code: [MaxBelitsky/cache-steering](https://github.com/MaxBelitsky/cache-steering)
- Local: `related_work/cache-steering/`
- Core implementation: `src/steering/cache_steering.py` — only 265 lines
- Demo: `examples.ipynb`

### Mechanism (from cache_steering.py)

```python
# Extract: forward pass on positive AND negative examples,
# diff K/V at the LAST TOKEN of each, mean over a batch
pos_values = cache_positive.value_cache[layer_id][batch, :, pos_indices, :]
neg_values = cache_negative.value_cache[layer_id][batch, :, neg_indices, :]
steering_values[layer_id] = pos_values - neg_values  # [n_heads, head_dim]

# Apply: prefill normally, then ADD the vector to the last token's K/V
cache.value_cache[layer_idx][:, :, application_token_idx, :] += sv * c_values
cache.key_cache[layer_idx][:, :, application_token_idx, :] += sv * c_keys
```

Two scalar knobs `c_keys` / `c_values` (analogous to our old α). One
[n_heads, head_dim] vector per layer. Single-token application.

### Comparison table

| Dimension | Mimir | Belitsky 2025 |
|---|---|---|
| Source | One forward pass on prose description | Many forward passes on (positive, negative) example pairs |
| What's captured | Full K/V across N=32 prefix tokens, top-half layers | One [n_heads, head_dim] K + V vector per layer (last token only, mean over examples) |
| Application | Prepend to `past_key_values`; queries attend to it as context | Add to user-prompt's last token's K/V at each layer |
| Tuneable knobs | None (init-only) | `c_keys`, `c_values` scalars |
| Artifact size | ~4 MB per concept | ~kilobytes per behavior |
| Validated use case | Novel concept registration (Flaxum, BP, JOTP) | Reasoning-style induction (CoT in small models) |

### What this means for our novelty story

The mechanism *family* is no longer unique — Belitsky 2025 publishes the
"modify intermediate-layer KV of a frozen LLM, no weights, no gradients"
primitive. But:

- **Description-init** (single forward pass on prose, not contrastive
  pairs, not gradient-trained) is unclaimed.
- **Per-concept registry** (hot-loadable library of registered concepts,
  with dependency chains, with isolation guarantees) is unclaimed.
- **Validated novel-term reasoning** (Flaxum, JOTP, Balance Publisher —
  terms genuinely not in pretrain — with compositional reasoning) is
  empirical territory nobody else has reported.

Three real contributions remain. Frame the README's related-work
section against Belitsky's specifics, not against the abstract.

## Other repos cloned (for completeness)

| Repo | Local path | Why we cloned it | Likely usefulness |
|---|---|---|---|
| **icl_task_vectors** (Hendel et al 2023) | `related_work/icl_task_vectors/` | Original "ICL creates task vectors" — single mid-layer vector capture from a forward pass | Closest cousin to our old `register_axiom.py` single-vector path. Reference implementation only. |
| **gisting** (Mu et al 2023) | `related_work/gisting/` | Gradient-trained compression of prompts to 1–26 soft tokens stored in KV (up to 26× compression) | Useful if we ever revisit gradient refinement. Their training recipe is the canonical reference. |
| **ICV** (Liu et al 2024) | `related_work/ICV/` | Single-vector residual-stream steering from ΔH of contrastive pairs | Backup option for refusal problem if `refusal_direction` underperforms |
| **prompt-cache** (Gim et al MLSys 2024) | `related_work/prompt-cache/` | Schema-driven KV-fragment reuse for serving latency | Reference for serving-system design when our axiom registry needs to scale |

## Papers we surveyed but didn't clone

- **Memory^3** ([arxiv 2407.01178](https://arxiv.org/abs/2407.01178)) —
  closest *framing* match: "explicit memory" as third tier between
  weights (implicit) and context (working). But it's a 2.4B model
  trained from scratch with the explicit memory mechanism baked in.
  Not a frozen-base technique.
- **Memorizing Transformers** ([arxiv 2203.08913](https://arxiv.org/abs/2203.08913)) —
  kNN external memory, jointly trained at pretraining time. Same
  primitive (KV-based external memory) but weight-coupled.
- **SK-Tuning** ([arxiv 2410.08598](https://arxiv.org/abs/2410.08598)) —
  prefix tuning where the prefix is real-word tokens, not random. Still
  trains a small adapter; not training-free.
- **Prefix Tuning** (Li & Liang 2021,
  [arxiv 2101.00190](https://arxiv.org/abs/2101.00190)) — the structural
  primitive (per-layer K/V prefix). Trained via gradients. We use the
  primitive, init-only.
- **Representation Engineering** (Zou et al 2023) — broad framework for
  steering vectors in residual stream. [Awesome list](https://github.com/chrisliu298/awesome-representation-engineering).
- **ROME / MEMIT** — modifies MLP weights to install facts. ~1000-edit
  ceiling. Different mechanism.

## Recommended order of work for next session

1. **Skim Belitsky's `cache_steering.py` end-to-end (~15 min).** Frame
   the README's related-work block against the actual mechanism, not the
   abstract. Acknowledge the precedent cleanly.
2. **Run `refusal_direction` on Qwen 2.5-32B-Instruct** with our axiom
   paraphrases as the contrast set. Highest-EV next experiment — closes
   the 6/10 gap that's blocking chat-model adoption.
3. **Implement attention-sink masking + global-layer-only injection for
   Gemma 4.** Test if "prefix in-window for all local layers" + "inject
   only at global layers" turns the null result into a positive one.
4. **CacheBlend selective recompute for 3+ prefixes** — only after
   confirming Path 2 (joint encoding) actually works as the brute-force
   fallback. Belt-and-braces.

## Sketch: refusal_direction adaptation for mimir

```python
# data: contrast pairs from existing axiom paraphrases
# - "positive" = paraphrase that the model answers correctly with prefix loaded
# - "negative" = same paraphrase, no prefix → model refuses with "I don't know"

# Step 1: extract candidate directions per layer (mean activation diff)
#   adapt related_work/refusal_direction/pipeline/submodules/generate_directions.py
#   substitute their harmful/harmless dataset for our refusal/compliant axiom set

# Step 2: pick the best direction by attempt-rate improvement on held-out axioms
#   adapt pipeline/submodules/select_direction.py

# Step 3: at inference, project the direction out of every layer's residual
#   adapt pipeline/utils.py:get_orthogonalized_matrix or use the runtime hook
#   compose with our existing prefix injection — the two interventions are
#   independent (one in residual stream, one in past_key_values)
```

This is a one-day spike. The hard part is curating the contrast set, not
the math.

## Citations

### Primary (read first)

- Belitsky et al, *KV Cache Steering for Controlling Frozen LLMs*, 2025.
  [arxiv:2507.08799](https://arxiv.org/abs/2507.08799) ·
  [code](https://github.com/MaxBelitsky/cache-steering)
- Arditi et al, *Refusal in Language Models Is Mediated by a Single
  Direction*, 2024. [arxiv:2406.11717](https://arxiv.org/abs/2406.11717) ·
  [code](https://github.com/andyrdt/refusal_direction)
- Yao et al, *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge
  Fusion*, 2024. [arxiv:2405.16444](https://arxiv.org/pdf/2405.16444) ·
  [code](https://github.com/YaoJiayi/CacheBlend)
- Xiao et al, *Efficient Streaming Language Models with Attention Sinks*,
  ICLR 2024. [arxiv:2309.17453](https://arxiv.org/abs/2309.17453) ·
  [code](https://github.com/mit-han-lab/streaming-llm)

### Secondary

- Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for
  Generation*, 2021. [arxiv:2101.00190](https://arxiv.org/abs/2101.00190)
- Hendel et al, *In-Context Learning Creates Task Vectors*, EMNLP 2023.
  [arxiv:2310.15916](https://arxiv.org/abs/2310.15916) ·
  [code](https://github.com/roeehendel/icl_task_vectors)
- Liu et al, *In-Context Vectors: Making In-Context Learning More
  Effective and Controllable Through Latent Space Steering*, ICML 2024.
  [arxiv:2311.06668](https://arxiv.org/abs/2311.06668) ·
  [code](https://github.com/shengliu66/ICV)
- Todd et al, *Function Vectors in Large Language Models*, ICLR 2024.
  [project page](https://functions.baulab.info/)
- Mu et al, *Learning to Compress Prompts with Gist Tokens*, NeurIPS 2023.
  [arxiv:2304.08467](https://arxiv.org/abs/2304.08467) ·
  [code](https://github.com/jayelm/gisting)
- Gim et al, *Prompt Cache: Modular Attention Reuse for Low-Latency
  Inference*, MLSys 2024. [arxiv:2311.04934](https://arxiv.org/abs/2311.04934) ·
  [code](https://github.com/yale-sys/prompt-cache)
- Yang et al, *KVLink: Accelerating LLMs via Efficient KV Cache Reuse*,
  NeurIPS 2025. [arxiv:2502.16002](https://arxiv.org/abs/2502.16002)
- Hu et al, *EPIC: Efficient Position-Independent Caching for Serving
  LLMs*, ICML 2025. [arxiv:2410.15332](https://arxiv.org/abs/2410.15332)
- Wu et al, *Memorizing Transformers*, ICLR 2022.
  [arxiv:2203.08913](https://arxiv.org/abs/2203.08913)
- Yang et al, *Memory^3: Language Modeling with Explicit Memory*, 2024.
  [arxiv:2407.01178](https://arxiv.org/abs/2407.01178)
- Khan et al, *SK-Tuning: Parameter-Efficient Fine-Tuning of LLMs Using
  Semantic Knowledge Tuning*, 2024.
  [arxiv:2410.08598](https://arxiv.org/abs/2410.08598)
- Zou et al, *Representation Engineering: A Top-Down Approach to AI
  Transparency*, 2023. [Awesome list](https://github.com/chrisliu298/awesome-representation-engineering)

### Surveys / catalogs

- *Awesome Knowledge Injection in LLMs*:
  https://github.com/lyyang01/awesome-knowledge-injection-in-LLMs
- *Injecting Domain-Specific Knowledge into LLMs: A Comprehensive Survey*,
  2025. [arxiv:2502.10708](https://arxiv.org/abs/2502.10708)
