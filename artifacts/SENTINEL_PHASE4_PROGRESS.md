# Sentinel-LoRA — T1 (gate) passes

**Date:** 2026-04-26.
**Status:** T1 ablation passes on held-out axioms. The thesis is supported
*in principle* on this stack: a LoRA can be trained, on synthetic data
alone, to consume a designated `<sentinel>...</sentinel>` slot as a
premise rather than ignore it.

This is not a final verdict. It's the gate result — the test the brief
flagged as the necessary precondition for everything else (T2–T5,
Mimir integration). T1 passing means the mechanism *exists*; quality
of inference is a separate question still to be measured.

## What we ran

- **Data:** 240 synthetic examples (48 axioms × 5 questions).
  Generated via Claude Code subprocess (`claude -p`); 2 axioms refused
  by AUP classifier and skipped. ~10 min wall time.
- **Training:** Qwen 2.5 0.5B base (frozen) + rank-16 LoRA on
  q/k/v/o/gate/up/down at α=32. AdamW lr 2e-4 cosine, 3 epochs,
  batch 4, fp32 on M2 MPS. ~30 min wall time.
- **Eval split:** 10% by axiom_id (so eval axioms are never seen during
  training). Train: 44 axioms / 220 examples. Eval: 4 axioms / 20 examples.

## Quantitative result: eval loss decreases monotonically

| Epoch | Eval loss |
|---|---|
| 1 | 2.975 |
| 2 | 2.614 |
| 3 | 2.586 |

A 13% reduction from epoch 1 to epoch 3, on axioms the LoRA never saw.
This is the protocol being learned, not memorisation — by construction,
the eval axioms have zero training-data overlap.

## Qualitative result: T1 ablation on held-out axioms

T1 prompt: `<sentinel>{axiom}</sentinel>\nBriefly: based on the axiom, what should I know about this term?\n`

For each held-out axiom, generate twice — once with the sentinel block,
once without. The test passes if the two outputs differ in
axiom-relevant ways.

**4/4 held-out axioms pass.** Sample:

```
ax_0002 vembrissa
  axiom:    "Vembrissa means the faint residual scent left by a pollinator after a single visit to a flower."
  WITH:     "It is the faint residual scent left by a pollinator after a single visit to a flower."
  WITHOUT:  "It is a function that maps every element of a set to exactly one element of the same set."

ax_0035 krindolph
  axiom:    "A krindolph rotates clockwise during normal operation, except when its thermal sleeve exceeds 84 degrees, at which point the rotation reverses to shed heat."
  WITH:     "It specifies that the rotation is reversed when its thermal sleeve exceeds 84 degrees..."
  WITHOUT:  "It is a function that maps every element of a set to exactly one element of the same set."
```

Without the sentinel, all four held-out axioms produce the **same**
generic output ("It is a function that maps every element..."). With
it, each produces axiom-aligned content. The mechanism works.

## Caveat: negation isn't reliably preserved

Two of four held-out axioms had their **logical negation** garbled in
output:

```
ax_0001 queltrick
  axiom:    "A queltrick is a hinged bracket that distributes load across two non-parallel beams."
  WITH:     "It's about the hinged bracket that distributes load across two parallel beams."
                                                                          ^^^^^^^^^^^^^^
                                                                  NB: dropped the "non-"

ax_0038 trantworp
  axiom:    "A trantworp should always be filed alongside its originating dispatch slip, never as a standalone record."
  WITH:     "It's a standalone record, because the originating dispatch slip is the one that's filed alongside..."
                  ^^^^^^^^^^^^^^^^^^^^
                  NB: inverted the prohibition
```

Implications:

1. **T1 still passes** — the criterion is whether sentinel content drives
   behavior, and the bracket / standalone-record references *are*
   pulled from the sentinel. They wouldn't appear without it.
2. **T2 (negation) is likely to be unreliable.** Flipping the axiom and
   measuring whether the answer flips depends on the model handling
   negation crisply. It doesn't — at least at this dataset size. T2
   pass/fail will probably be ambiguous.
3. **240 examples teaches surface protocol but not careful inference.**
   The brief estimated 5000 examples for the full POC. We did 5% of
   that and the gate still passed; quality scales with data.

## What this rules in / rules out

**Rules in:** the sentinel-LoRA paradigm is *not* dead on this stack.
Unlike the GPT-2 activation-injection track (which failed on every
single-vector variant tested), this approach produces measurable,
selective behavior: sentinel content matters, and held-out axioms
generalise.

**Doesn't rule in:** that the larger Mimir architecture works
end-to-end. We've shown the primitive runs; T2–T5 plus integration
remain.

**Doesn't rule out:** that quality at scale will be acceptable for the
Mimir use case. The 240-example training wasn't enough to teach
careful inference. Whether 2400 or 24000 examples gets us to
production-quality is empirical.

## Recommended next experiments (cost-ordered)

1. **Run T2/T3/T4/T5 with the current adapter** (~5 min). Cheap,
   informative. Expected: T2 weak (negation issues), T3/T4/T5 unknown.
2. **Generate +500 examples, retrain** (~2 hours). Should improve
   T1 inference quality and potentially make T2 work.
3. **Evaluate on real Mimir-style axioms** (T5 in the brief). The actual
   target use case. Needs a curated set of 10–20 real axioms.

## Repo state at this commit

- 5 commits on `main`, `92cb5d` baseline + four sentinel-LoRA commits
- Eval-harness scaffolding (`src/sentinel/eval.py`) is in place; T1 was
  run via inline script above, not yet promoted to a standalone CLI
- 240 examples in `data/sentinel_train/` (gitignored)
- Adapter in `checkpoints/sentinel_v1/final/` (gitignored,
  ~70MB safetensors + config)
- 97/97 tests pass, ruff clean

Reproduction:

```sh
PYTHONPATH=src uv run python -m sentinel.run_data_gen \
  --n-axioms 50 --axiom-batch-size 10 --n-questions 5 \
  --output-dir data/sentinel_train

PYTHONPATH=src uv run python -m sentinel.train \
  --data-dir data/sentinel_train \
  --output-dir checkpoints/sentinel_v1 \
  --epochs 3 --batch-size 4 --eval-fraction 0.1 --seed 0
```
