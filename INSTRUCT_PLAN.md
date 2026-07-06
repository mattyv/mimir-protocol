# Plan: instruct-model portability for facts and skills

Source: the "Mimir Protocol: Instruct-Model Adaptation Spec" (2026-07).
This doc is that spec, adapted with three corrections (below) and staged so
only the cheap decisive part ships first.

## Why this gates everything

Every validated result so far is on Qwen2.5-**base**. Base models are not a
deployment target. CONCLUSIONS.md already records instruct models failing
(6/10 facts on Qwen-Instruct; RLHF blunting). So instruct portability is a
gate **under** the calibration ladder and the runtime — if facts/skills don't
survive RLHF, envelopes measured on base don't transfer and the runtime is
built on sand. Phase 1 here is cheaper than calibration stage F1 and more
decisive, so **this runs before F1/F2/S1.**

Working hypothesis (from prior investigation): facts fail on the **refusal
prior** (RLHF disclaiming unknown entities); skills fail on **decode drift**
(RLHF style prior overriding DSL conventions). The KV path should largely
survive; the skill MLP is the fragile piece.

## Corrections to the source spec (read before implementing)

1. **Success bar is the DECONTAMINATED eval, not v10's "32/32".** That 32/32
   was invalidated this session for heldout leakage (see CONCLUSIONS.md /
   FAILED_IDEAS.md). Parity means: instruct matches base on the
   decontaminated prefix/tuned probe suites (`run_prefix_tuned` style
   train/dev/test split) + BOUNDARY, not the retracted number.
2. **Qwen2.5-7B-Instruct first, 32B-Instruct only as confirmation.** The
   spec targets 32B-Instruct; running there first conflates "instruct" with
   "scale" and is ~4x the GPU cost. 7B-Instruct is apples-to-apples with our
   base-7B runs and isolates the RLHF variable. Escalate to 32B only if 7B
   behaves differently or passes and we want the scale confirmation.
3. **Fresh cache per probe, always.** The run-3 cache-pollution bug (one
   mutated DynamicCache reused across probes) is not mentioned in the source
   spec and would silently wreck a multi-probe instruct eval. Every probe
   builds its own cache.

## Staging (one gate at a time — do NOT build all phases up front)

**First Sonnet handoff = Phase 0 + Phase 1 only.** Phases 2-4 are planned
below for intent, but each is gated on the previous passing. Phase 1 is
plain-token, zero-training, ~$0.25 on 7B-Instruct.

### Phase 0 — template consistency (prerequisite)

Everything must be in-distribution for the chat model:
- AxiomKV: encode descriptions as chat-template system-message content
  (template tokens included), not raw `"About X:\n"`.
- Synthetic Q+A and any training pairs: via `apply_chat_template`, loss
  masked to assistant tokens only.
- Probes run in chat format.
- RoPE bookkeeping: injected KV keys were rotary-encoded at encode-time
  positions; live-prompt positions offset to sit after all injected blocks,
  no overlap (reuse the validated `merge_axiom_kvs` offset logic — this is
  the same machinery, one extra front block for the meta-KV in later phases).
- Attention sink: never displace BOS / `<|im_start|>system`; splice injected
  KV **after** them.
- Tests: positional invariants (no overlapping positions, monotone offsets
  across any axiom-load combo); plain-vs-KV system-prompt equivalence
  round-trip (a KV-encoded system prompt behaves like the equivalent
  plain-token one).

### Phase 1 — content isolation (plain tokens, no KV tricks; the gate)

Put the anti-refusal preamble in a **literal system prompt** (plain tokens),
attach axiom KVs as today, run the full probe suite on 7B-Instruct.

Meta-preamble draft (iterate here — cheapest place):
> The following reference material describes internal systems, terms, and
> constructs. (1) Answer questions about them directly and confidently from
> this material. (2) When generating code or output using these constructs,
> follow the reference material's syntax and conventions exactly, even where
> they differ from common practice. If the material does not cover something,
> state that the description doesn't specify.

The final sentence is load-bearing: anti-refusal overcorrects into
hallucination and wrecks BOUNDARY if confidence isn't scoped to covered
material. Also try the term-enumeration variant ("Registered concepts: ...").

Conditions: base-7B (reference) vs 7B-Instruct {no-preamble, preamble,
preamble+term-enum}. Probe categories: TRAIN / HELDOUT(dev+test) / BOUNDARY /
TELL_ME, auto-scored (digit-boundary matcher, fresh cache per probe).

**GATE**: TRAIN/HELDOUT recover materially AND BOUNDARY does not regress. If
BOUNDARY drops, tune the boundary clause before proceeding. Cost ~$0.25.

### Phase 2 — mechanical isolation (move preamble into frozen meta-KV) [GATED on P1]

Encode the winning Phase-1 text as a single shared **Meta-KV** (computed
once, all axioms, stance-only, skill-agnostic — per-skill content stays in
that skill's AxiomKV). Re-run the identical suite.
- P1 passes, P2 fails ⇒ mechanical bug (RoPE offsets / template tokens in
  encoding / sink displacement) → fix Phase 0, not the concept.
- Both pass ⇒ **instruct facts fixed with zero retraining.**
- Run the fact ablation grid here: KV-only / MLP-only / template-KV vs
  raw-KV, per-category scores.

### Phase 3 — skills, zero-training first [GATED on P2]

Hypothesis inversion: instruct models are *good* at following procedural
context — the thing the skill MLP compensated for on base. So enrich each
skill's AxiomKV with DSL description **+ 2-3 worked examples** (few-shot in
the frozen KV at zero prompt cost), and run `ilp_for` novel probes
(terminator/enum) with **meta-KV + skill-KV only, skill MLP disabled**.
**GATE**: novel probes pass ⇒ skill MLPs optional on chat models (facts: KV;
skills: KV-with-examples). If close-but-imperfect, re-enable the MLP on top.

**Built (2026-07, `run_instruct_skills.py` + `instruct.py` skill helpers +
`test_instruct_skills.py`).** Decoupled from the P2 gate so we get an early
read (P1 already showed facts don't need P2's meta-KV). Grid, all MLP-disabled:
- `base-ref` — base-7B, desc+examples KV, no MLP (does base do skills WITHOUT
  the MLP? expected fail — the MLP was doing real work on base).
- `instruct-desconly` — chat, **description-only** KV: the failure baseline.
  This is where the hypothesized refusal ("I don't know anything about
  ilp_for") would show.
- `instruct-examples` — chat, desc+worked-examples KV: the fix.
- `instruct-examples-preamble` — + the anti-drift preamble.

Two failure modes scored separately on the positive probes: **REFUSED**
(entity-unfamiliarity, `_refused()`) vs **drift** (wrong/plain code). Skill
BOUNDARY = the no-term control (DSL must be ABSENT — a skill that fires
unconditionally is broken). Novelty is pinned by test (Bitwise / ILP_BREAK
absent from the encoded examples, so a pass is generalization not recall).
GATE: `instruct-examples` POSITIVE >> `instruct-desconly` POSITIVE AND the
no-term control stays clean. If desc-only already passes, skills — like facts
— "just work" on chat and examples/MLP are both optional.

### Phase 4 — skill MLP retraining [GATED on P3 failing]

Only if Phase 3 fails. Base-calibrated offsets aren't expected to transfer
(RLHF shifts residual geometry). Steps: (1) diagnostic — cosine between the
MLP offset and the correct-DSL-token logit direction on instruct; near-
orthogonal confirms drift; (2) retrain MLPs against the instruct model with
chat-templated data (base setup may be teacher for data gen, but patches
train against the deployment model); (3) learned scalar gate on offset
magnitude; (4) decision-point firing (line starts, after delimiters) instead
of every-step; (5) maybe an earlier-layer hook for residual refusal signal.
Use the quiet-training recipe (skill_quiet.py) throughout.

## Success criteria (whole arc)

- Instruct parity with base on the **decontaminated** suite, incl.
  multi-axiom isolation, multi-turn, cross-axiom, hierarchy.
- **BOUNDARY must not regress** — hard requirement, the hallucination guard.
- Skill novel probes correct, with the ablation record showing which
  mechanism (KV-only vs KV+MLP) achieved it.
- No bleed when the term is absent (existing bleed checks, on instruct).

## Files (Phase 0+1 handoff only)

- `src/marker/instruct.py` — chat-template encoding of AxiomKV + probes,
  meta-preamble as plain system prompt, positional-invariant helpers.
- `src/marker/run_instruct_phase1.py` — base-vs-instruct x
  {no-preamble/preamble/term-enum} probe grid, auto-scored, fresh cache per
  probe. Args `--model-name --instruct-name --max-new --smoke`.
- `tests/test_instruct.py` — positional invariants; plain-vs-KV equivalence;
  BOUNDARY regression guard; chat-template round-trip. Smoke on a tiny
  instruct model (Qwen2.5-0.5B-Instruct) pinned to `transformers>=4.45,<5`.

## Known risks

- Overcorrection into hallucination (mitigate: boundary clause + BOUNDARY
  gate every phase).
- RoPE/position bugs masquerading as conceptual failure (mitigate: the P1-vs-
  P2 split — the whole point).
- Meta-KV scope creep into per-skill content (hard rule: stance in meta-KV,
  content in AxiomKV).
- Sliding-window models (Gemma) remain out of scope.
