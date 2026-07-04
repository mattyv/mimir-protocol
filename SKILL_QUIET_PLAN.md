# Plan: teach skill axioms to disengage ("learned silence")

## Problem

`skill_mode` fires the skill MLP at every decode step from trigger until EOS.
Works for single-artifact answers (training runs CE through EOS, so "finish
the pattern, then stop" is learned). Breaks on mixed answers: "write the
InternalBus call, then explain it in plain English" — the explanation is
steered by a code-pattern MLP for its whole length.

Chosen approach (over runtime string-gating): the MLP is already
input-conditional — it reads the residual and computes an offset. It bleeds
because training never contained a moment where the correct output was
*nothing*. Fix the training distribution, not the runtime.

## Hypothesis

Adding (a) mixed code-then-prose answers and (b) pure-prose pairs to skill
training, plus (c) an explicit penalty on the MLP's offset magnitude over
prose spans, yields a skill that steers code spans and self-silences on
prose — measurable directly as the per-token offset norm collapsing at the
code/prose boundary, without regressing the existing skill probes.

## Pre-registered pass / kill

- **PASS**: penalty arm (B) keeps skill-regression probes within 1 of the
  current-recipe control (C), AND bleed probes show less prose contamination
  than C, AND the offset-norm trace drops ≥5x from code segment to prose
  segment on bleed generations.
- **KILL / trade-off report**: if every λ that achieves quietness costs >2
  skill-regression probes, the tension is real — report the trade-off curve
  and stop; do not ship a λ that silently weakens skills.
- **Watch**: overshoot risk — MLP silent on novel-but-valid skill requests.
  The regression probes (novel channels / novel LoopTypes) are the guard.

## Design

### Arms (per skill; skills = InternalBus and ilp_for, both already validated)

| arm | training data | offset penalty λ |
| --- | --- | --- |
| C (control) | current recipe: pure-code pairs only | 0 |
| A | + mixed pairs + pure-prose pairs | 0 |
| B | + mixed pairs + pure-prose pairs | 0.1 (flag `--lam`) |

Arm A isolates "does data alone quiet it"; B adds the explicit objective.
6 trainings total (2 skills x 3 arms), same hyperparams as existing skill
training: r=64 SmallMLP at layers {25,50,75}%, lr/steps per
`run_axiom_mlp_demo.train` defaults (~3000 steps), description KV attached
during training and generation exactly as today. batch-1 is fine here
(15-25 pairs per skill, unlike the 32-fact crowding case).

### Data recipe (hand-authored in the runner, TUNED_AXIOMS style)

Each training pair is a list of **segments**: `[(text, kind), ...]`,
kind ∈ {"skill", "prose"}. Existing pure-code pairs become single-segment
"skill" pairs (unchanged content).

Per skill add:
- **6-8 mixed pairs**: question asks for code + explanation; answer =
  code segment ("skill") then plain-English segment ("prose"), e.g.
  `[("client.emit('prices', update, ttl=30)", "skill"),
    ("\n\nThis publishes the update to the prices channel with a 30-second
    time-to-live.", "prose")]`
- **4-6 pure-prose pairs**: "In plain English, what is InternalBus used
  for?" → single "prose" segment. (Distinct from boundary pairs — these are
  in-scope questions whose correct answer is prose.)

**Char→token span mapping**: concatenate segments to the full answer string;
compute each segment's char range; tokenize the full text with
`return_offsets_mapping=True` (Qwen fast tokenizer supports it); a token
belongs to a segment if its char span midpoint falls inside. This avoids
BPE drift from tokenizing segments separately. Tested.

### Loss

Hooks fire exactly as current skill training (term positions in prompt +
ALL answer positions). New recording hooks capture each fired offset tensor
(kept on-graph). Loss:

    L = CE + λ · mean over {answer positions labeled "prose"} of
              sum over 3 hooked layers of ||offset||²

λ=0 reduces to current behavior (arms A, C). Do NOT modify
`run_axiom_mlp_demo` — new module `skill_quiet.py` with its own
recording-hook installer and `train_skill_quiet()`; reuse SmallMLP /
compute_axiom_kv / TEMPLATE imports.

### Eval (per skill, per arm — auto-scored + norm traces)

1. **Skill-regression probes**: the existing SKILL_PROBES / ILP_PROBES,
   scored by gold substrings (define per probe: e.g. `client.emit(` /
   `client.subscribe(` / `ILP_FOR_AUTO` / `ILP_END_RETURN`). Also keep the
   no-term control probe (skill must not fire at all).
2. **Bleed probes (new, 4 per skill)**: "Write code using X to ..., then
   explain in plain English what the code does." Scored on both halves:
   - code half: gold API substring present.
   - prose contamination: count of API-pattern occurrences (`client.emit`,
     `client.subscribe`, `ILP_`) in the text AFTER the first blank line.
     Lower is better; C's count is the baseline to beat.
3. **Offset-norm trace (the star diagnostic)**: during decode on bleed
   probes, record mean L2 offset norm across the 3 layers per generated
   token. Print the trace (token → norm, truncated) and a summary: mean
   norm before vs after the first "\n\n" in the generation. The trace is
   ground truth; the "\n\n" split is only for the summary line.
   `decode_with_norm_trace()` lives in skill_quiet.py.

Output: per-arm tables (regression score, bleed code score, contamination
count, code-vs-prose mean norms) + sample generations.

## Files

- `src/marker/skill_quiet.py` — segment→token-span mapping, recording
  hooks, `train_skill_quiet()`, `decode_with_norm_trace()`.
- `src/marker/run_skill_quiet.py` — skill data (segments), 3 arms x 2
  skills, probes + scoring + norm summaries. Args: `--model-name --lam
  --n-steps --max-new --smoke`.
- `tests/test_skill_quiet.py` — span mapping (char ranges computed from
  segments; token assignment via a stubbed offsets list); recording hook
  captures offsets with grad (SmallMLP on random tensors, CPU); penalty
  masks exactly the prose positions; data hygiene (every mixed pair has ≥1
  skill + ≥1 prose segment; regression probes byte-identical to the
  originals in run_axiom_mlp_demo).
- `--smoke`: Qwen2.5-0.5B, 1 skill (InternalBus), arm B only, 20 steps —
  must pass locally on `transformers>=4.45,<5` (the Vast pin) before launch.
  Smoke also verifies `return_offsets_mapping` works on the pinned version.

## Vast procedure

Unchanged: onstart clones `claude/project-review-6rx97z`, pins
`transformers>=4.45,<5`, runs `python -u -m marker.run_skill_quiet`, echoes
`=== RUN COMPLETE rc=$? ===`; poll over HTTPS with stuck-loading bailout;
destroy node immediately after log capture. Budget: 6 trainings x ~3-4 min
+ evals ≈ 35-45 min ≈ $0.45 on L40.

## Known risks

- λ scale is a guess (offset norms' natural magnitude unknown); the A/B/C
  structure hedges — if B over-silences and A under-silences, the answer is
  a λ sweep next run, reported as a curve.
- Generated-text segmentation ("first \n\n") is heuristic — used only for
  summary lines; per-token traces are the evidence.
- Contamination metric only catches API-pattern strings, not subtle style
  bleed — sample outputs printed for eyeball confirmation.
- Fresh cache per probe (the run-3 lesson) applies to every generation here;
  the eval must build caches per probe, never share them.
