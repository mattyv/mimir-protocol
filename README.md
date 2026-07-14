# Mimir-Protocol

Give a **frozen** LLM (no weight changes) new capabilities by writing directly
into its attention memory — the per-layer key–value cache ("KV"). One shared
mechanism, two research threads.

## The shared mechanism

At every forward pass we prepend precomputed key–value vectors to the model's
attention memory, so the model attends to information that was never in its
weights or its prompt text. Nothing about the base model changes — the injected
KV carries the new capability. We call this "Mimir".

## Two threads

### 1. Knowledge injection — teach a frozen model a new *fact*

Train a small per-concept MLP "patch" + a frozen KV cache of the concept's
description. The model answers questions about things it was never trained on
(post-cutoff facts, private APIs, fictional entities), declines out-of-scope
questions, and handles multi-turn/composite concepts — byte-for-byte identical
weights, no description text in the prompt.

→ **[docs/knowledge-injection.md](docs/knowledge-injection.md)** — full write-up,
results, how it works, try-it.

### 2. Latent thought-prediction — compress *reasoning* into a few vectors

Squeeze a reasoning step into ~8 KV slots (a "thought"), then **predict** the
next thought and **render** any thought back to faithful text. The bet is memory
+ speed. The render lane (thought → exact text) is validated; the fast lane
(chaining predicted thoughts to skip generation) is the open pillar.

→ **[docs/latent-thought-prediction.md](docs/latent-thought-prediction.md)** —
pipeline diagrams, results, plan docs.
&nbsp;·&nbsp; **[Predictor anatomy →](docs/predictor-anatomy.md)** — a diagram of
the next-thought predictor's forward pass.

## Two words we use precisely

- **Train** — change the model's weights. Fine-tuning, LoRA, full retraining.
  The model is byte-for-byte *different* afterwards.
- **Understand** — don't change weights. Compute per-concept patches / KV that
  act at inference time. The base model is byte-for-byte *identical*; the
  patches and KV carry the new capability.

## Repo layout

```
src/marker/
  run_axiom_mlp_demo.py     # knowledge injection: main demo (MLP + AxiomKV per axiom,
                            # full probe suite; see docs/knowledge-injection.md)
  run_axiom_mlp_mini.py     # minimal local axiom test ("Glorbox")
  prefix_tuning.py          # full KV prefix approach (teacher for synthetic Q+A)
  axiom_registry.py         # test axioms with descriptions and Q+A
  soft_prompt*.py           # earlier soft-prompt approaches (v5-v9)

  gist_model.py             # latent thoughts: encoder (text -> thought KV)
  predictor.py              # latent thoughts: NextThoughtPredictor + losses/metrics
  render.py                 # latent thoughts: render decoder + literals ledger
  bridge.py                 # latent thoughts: predicted thought -> injectable KV
  run_stage2.py             # latent thoughts: encode corpus + train/eval predictor
  run_render.py             # latent thoughts: train the render decoder

modal_blends.py             # Modal entrypoints for cloud runs
docs/                       # design docs & the two thread write-ups
tests/                      # mechanical invariants

# Journals: CONCLUSIONS.md, FAILED_IDEAS.md, THINGS_TO_TRY.md (knowledge injection)
#           STAGE2_PLAN.md, FASTLANE_PLAN.md, LATENT_PLAN.md (latent thoughts)
```

## License

See [LICENSE](LICENSE).
