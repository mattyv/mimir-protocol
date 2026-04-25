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
