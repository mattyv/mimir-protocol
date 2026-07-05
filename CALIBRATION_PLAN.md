# Calibration plan: measure capability-vs-size, let the framework adapt

## Principle

The framework should manage axiom size adaptively — split, distill, choose
carrier, cap sessions — using **measured capability envelopes**, not
hardcoded guesses. Every calibration stage below emits three things:

1. a **curve** (capability vs size, auto-scored),
2. a **decision rule** derived from the curve,
3. a **machine-readable envelope artifact** (`envelopes/<model>.json`)
   that the future runtime consumes.

Envelopes are per-model: recalibrating for a new base model = rerun the
harness, not redesign the system. RUNTIME_PLAN.md is **gated on stages
F1, F2, S1** — no runtime build until the envelopes exist.

Stages run one at a time, each with pre-registered gates, in the
plan → Sonnet-implement → smoke → Vast → Fable-review loop.

## Already calibrated (fold into the envelope, no rerun)

- **Prefix carrier (facts)**: crowding run 3 — lossless memorization of
  32 facts in 4 tokens; FACTS-parity at 4 tokens/fact up to F=8;
  phrasing robustness degrades above F=8 and is flat in N for
  far-templates. Rule: prefix carrier only for F<=8 per artifact, else
  text carrier or node split.
- **Skill disengagement**: penalty-trained (lam=0.1) skills self-silence
  on prose at zero regression cost. Rule: all skill training uses the
  quiet recipe.

## Stage F1 — fact recall vs description size (NEXT; ~$0.20, no training)

The text carrier is validated only to ~264 tokens of dense key=value
facts. Real Mimir content is narrative pages, facts buried in prose with
distractors.

- One synthetic axiom, 12 facts (crowding.py attribute catalog), embedded
  in deterministic generated narrative with **distractor** values
  (unique, digit-boundary-safe vs golds), padded to lengths
  ~{150, 300, 600, 1200, 2400, 4800} tokens.
- Conditions per length: **RAW** (inject full narrative KV) vs
  **DISTILLED** (inject only the ~100-token fact list extracted from the
  same content) vs ZERO. No training anywhere — prefill + decode only.
- Track which third of the document each fact sits in (early/mid/late) —
  the lost-in-the-middle check.
- Probes: 2 unseen-template questions per fact (reuse the crowding
  dev/test template family), digit-boundary scored.
- **Outputs**: L*_raw and L*_distilled (length where recall < 90%);
  distill_gain(L) = DISTILLED - RAW recall per length; position-effect
  table. **Rules**: node split threshold; whether Mimir needs a fact-
  extraction step at registration or can inject raw pages.

## Stage F2 — session capacity: recall vs number of loaded axioms

- 10 / 25 / 50 / 100 small axioms (crowding generator, distinct value
  spaces), RoPE-merged into one session cache, text carrier.
- Probe a random subset of axioms + measure cross-axiom attribution
  errors (sibling-axiom value in a wrong answer) + context cost.
- **Outputs**: K* (axioms-per-session at >=90% recall), attribution-error
  curve. **Rule**: session hydration budget + eviction need (yes/no).
- No training; ~$0.25.

## Stage S1 — skill complexity calibration

Skills are validated at exactly two sizes (InternalBus: 2 methods;
ilp_for: ~7 macros). Unknown: where does a fixed-r MLP saturate as the
API surface grows?

- Synthetic skill families with P in {2, 4, 8, 16} patterns (generated
  API: methods with distinct names/signatures/conventions), quiet-recipe
  training, r in {32, 64, 128}.
- Score novel-usage fidelity (unseen argument/channel combos per
  pattern), disengagement (norm trace), regression per size.
- **Outputs**: P* per r (patterns-per-skill envelope); r sizing rule.
- Training-heavy: 12 cells x ~3 min ~= $0.6. Runs AFTER F1/F2.

## Stage F3 (conditional) — refresh prefix-carrier envelope

Only if F1/F2 results make the prefix carrier relevant for v1 (e.g.
session context cost at K* is painful). The crowding surface already
covers it; a top-up would add per-fact-size (long values) sensitivity.

## Envelope artifact schema (checked in, versioned)

    envelopes/qwen2.5-7b.json
    {
      "model": "Qwen/Qwen2.5-7B",
      "calibrated": "2026-07-..",
      "fact_text": {"l_star_raw": ..., "l_star_distilled": ...,
                     "distill_gain": {"600": ..., ...},
                     "position_effect": {...}},
      "session":   {"k_star": ..., "attribution_curve": {...}},
      "prefix":    {"tokens_per_fact": 4, "max_facts": 8,
                     "phrasing_caveat": "far-template flat in N"},
      "skill":     {"p_star_by_r": {...}, "quiet_lambda": 0.1}
    }

Runtime decision rules (later, in RUNTIME_PLAN.md terms): if node tokens
> l_star_raw -> distill; if still > l_star_distilled -> split into child
nodes; hydrate <= k_star axioms per session; skill patterns <= p_star or
split the skill; carrier choice by fact count vs the prefix envelope.

## Known risks

- Synthetic narrative filler is not real Confluence prose — F1 gives a
  lower bound on difficulty for structure but may miss real-page messiness
  (tables, code blocks). Mitigation: one hand-written realistic page at
  each length as a spot check alongside the generated ones.
- Single seed per stage first pass; envelopes marked "provisional" until
  a second-seed confirmation run (cheap, since F1/F2 need no training).
- Fresh-cache-per-probe discipline (run-3 lesson) applies to every eval.
