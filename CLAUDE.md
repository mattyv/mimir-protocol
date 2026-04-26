# Project conventions

## TDD

Write the test first. Watch it fail. Make it pass. Refactor.

- New behaviour starts with a failing test in `tests/`.
- Run `pytest` (or the relevant subset) before claiming a step is done.
- Don't write production code without a test pinning the behaviour you want, unless it's pure plumbing (imports, config dicts, hook registration that has no logic to assert on). When in doubt, write the test.
- For mech-interp work specifically: tests assert mechanical invariants (zero-vec inject is a no-op, target tokens are single BPE tokens, hook fires at the configured layer), not numerical outcomes of the experiment itself. The experiment results live in plots/artifacts, not assertions.

## Ruff

`ruff check` and `ruff format` are the lint/format authority.

- Run `ruff check --fix` and `ruff format` before considering a change complete.
- No Black, no isort, no flake8 — ruff covers all of it.
- Configuration lives in `pyproject.toml` under `[tool.ruff]`.

## Communication style

Default to terse plain-English. The user is a working engineer, not an ML researcher.

- One- or two-sentence answers when possible. Bullet lists over prose paragraphs.
- TLDRs and updates: a few lines, not a wall of text. Skip restating what we just did.
- No status-recap preambles ("So we…", "As you saw…"). Get to the new info.
- Defining jargon: when a term is unavoidable, give a short plain-English gloss in parentheses the first time it appears in a thread. Examples: "LoRA (a tiny set of extra learned weights bolted onto the model)", "residual stream (the running sum of vectors flowing layer-to-layer)", "logit (the model's pre-softmax score for a token)". After it's defined once, use the term freely.
- Prefer concrete words. "The model's stored prior reading" beats "the activation manifold".
- Numbers and contrasts beat adjectives. "α=20 says 'order book, market data'; baseline says 'balance sheet'" beats "noticeable improvement".
- If a request is ambiguous, ask one short question rather than guess.

## Vocabulary: "understand" vs "train"

Two words we use very precisely. Don't blur them.

- **Train** = change the model's weights. Fine-tuning, LoRA, pretraining.
  After training, the model is byte-for-byte different.
- **Understand** = activation injection at runtime — add a precomputed
  meaning-vector into the residual stream at the term's position. No
  weights change. The model is identical before and after; the vector
  carries the new knowledge.

When describing what this project does, default to *understand*. Reserve
*train* for actual weight updates (the parked LoRA experiment, the
sentinel-LoRA fallback). "Teach" is fine in casual prose but if precision
matters, pick the right one of the two.

## TLDR sections

When reporting any non-trivial result — experiment outcome, multi-step debug, training run, comparison — lead with a TLDR.

- Format: a heading line `**TLDR:**` followed by 2-4 short bullets. Keep it under 6 lines total.
- Cover: what was tried, what happened (one concrete fact, ideally a number or quote), what it means for the next step.
- Put detail (logs, full output, deeper reasoning) *after* the TLDR, behind a separator or under a `Detail:` heading. Never bury the answer.
- If the user asks "TLDR" mid-thread, give just the TLDR — no detail follow-up unless asked.
- For comparisons, prefer a small markdown table over prose: rows are the conditions, columns are baseline / change / verdict.
