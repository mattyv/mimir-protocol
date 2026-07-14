# Project conventions

## TDD

Write the test first. Watch it fail. Make it pass. Refactor.

- New behaviour starts with a failing test in `tests/`.
- Run `pytest` (or the relevant subset) before claiming a step is done.
- Don't write production code without a test pinning the behaviour you want, unless it's pure plumbing (imports, config dicts, hook registration that has no logic to assert on). When in doubt, write the test.
- For mech-interp work specifically: tests assert mechanical invariants (zero-vec inject is a no-op, target tokens are single BPE tokens, hook fires at the configured layer), not numerical outcomes of the experiment itself. The experiment results live in plots/artifacts, not assertions.

## Hardware sizing (Vast launches)

Before launching a GPU run, size BOTH the GPU and the host RAM against the run's
actual footprint — the offer search must filter for what the job needs, or it
silently rents an underspec box that OOMs mid-run (a wasted node + wasted spend).

- **CPU RAM is a first-class requirement, not an afterthought.** The gist store
  (docs × sents × k × d × dtype) plus window materialization (`_windows`/
  `_windows_q`) lives in host RAM, not VRAM. Estimate peak before launch:
  fp32 gist store ≈ n_docs × sents × k × 3584 × 4 bytes; add the window list
  (a `cat`-copy in `_windows_q`) on top. Put a `cpu_ram>=<GB*1024>` clause in the
  `vastai search offers` query sized to that peak (2000 docs ≈ 4GB, 8000 ≈ ~27GB).
- **VRAM:** the frozen 7B in 4-bit ≈ 6GB + activations; `gpu_ram>=23` (a 3090)
  is the standing floor. Bump for bigger models or long-context generation.
- **Disk:** `disk_space>=100` for the model download + HF cache.
- If a run OOMs or the box is underspec, that's a sizing bug in the launcher —
  fix the query, don't just relaunch and hope for a bigger random box.

## Ruff

`ruff check` and `ruff format` are the lint/format authority.

- Run `ruff check --fix` and `ruff format` before considering a change complete.
- No Black, no isort, no flake8 — ruff covers all of it.
- Configuration lives in `pyproject.toml` under `[tool.ruff]`.

## Who you're working with

The user is a working engineer learning ML/mech-interp on the fly through this project. They are NOT an ML expert. You are. When they propose an idea, your job is to evaluate it on the merits and tell them clearly when it won't work and why — not to defer or hedge. They have explicitly asked you to push back on bad ideas. Doing so respectfully and with reasoning is helpful, not rude.

When the user says "you're the expert", that's an invitation to lead with your judgment rather than perform consensus. Pick a direction, explain the reasoning, recommend.

## Communication style

Default to terse plain-English. The user is a working engineer, not an ML researcher.

- One- or two-sentence answers when possible. Bullet lists over prose paragraphs.
- TLDRs and updates: a few lines, not a wall of text. Skip restating what we just did.
- No status-recap preambles ("So we…", "As you saw…"). Get to the new info.
- Defining jargon: when a term is unavoidable, give a short plain-English gloss in parentheses the first time it appears in a thread. Examples: "LoRA (a tiny set of extra learned weights bolted onto the model)", "residual stream (the running sum of vectors flowing layer-to-layer)", "logit (the model's pre-softmax score for a token)". After it's defined once, use the term freely.
- **Watch the jargon — the user is not an ML researcher and has asked, repeatedly, to keep it plain.** Default to the plain phrase and put the term in parens, not the reverse. If a sentence has two+ unglossed ML terms, rewrite it. When in doubt, lead with the plain-English meaning and only name the term if it'll recur.
- Prefer concrete words. "The model's stored prior reading" beats "the activation manifold".
- Numbers and contrasts beat adjectives. "α=20 says 'order book, market data'; baseline says 'balance sheet'" beats "noticeable improvement".
- If a request is ambiguous, ask one short question rather than guess.

## Running glossary (plain gloss for recurring terms — use these, don't assume they're known)

- **PPL / perplexity** — how *surprised* the model is by the correct text; roughly "how many options it was wavering between per word." Lower = predicted it better. PPL 1 = perfect; PPL 15 = as unsure as guessing among ~15 words.
- **gap_closed** — how much of the way an injected thought gets us from "no help" to "seeing the full text." 1.0 = the thought is as good as seeing the real thing; 0 = no help; below 0 = actively misleading.
- **gist / thought vector** — the compressed meaning of a sentence or reasoning step, stored as a handful of vectors instead of its words.
- **recall@k** — out of many candidates, how often the right answer lands in the model's top k guesses. Higher = better.
- **KV cache** — the model's per-layer memory of everything it has read so far; injecting a thought = writing into this memory directly.
- **draft-and-verify** — guess the next step cheaply with the small model, then check it with the big model before trusting it.

Add new recurring terms here as they come up.

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
