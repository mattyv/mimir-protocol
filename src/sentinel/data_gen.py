"""Synthetic data generator. Spawns Claude Code CLI subprocesses.

Why subprocess instead of the Anthropic SDK: Claude Code is already
authenticated via its own session, so there's no API-key management and
generation costs nothing extra at the margin (it bills against the
existing Claude Code subscription rather than the metered API).

Each call spawns a `claude -p <prompt>` process. We pass the combined
system+user prompt as a positional argument and parse the subprocess's
stdout as JSON. The Pydantic schema in each method validates the parsed
output, so a malformed response fails loudly with a `ValidationError`
rather than producing garbage examples.

Tradeoffs vs the SDK:
  - No prompt caching across calls. Each subprocess is independent;
    Claude's underlying cache may still hit but we have no visibility.
  - No structured-output enforcement at the API level. We rely on the
    model honouring "return only JSON" and the Pydantic validator.
  - Slower per call (~1-2s of process startup overhead). For ~1000
    calls that's ~30 minutes of pure overhead.
  - Cost is opaque from this side — Claude Code tracks it internally.

For the full 5000-example run, batching multiple axioms / questions per
call (which the prompt builders already support) keeps total subprocess
count ~hundreds, not thousands.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TypeVar

from pydantic import BaseModel

from sentinel.data_schema import (
    Axiom,
    AxiomShape,
    ContrastivePair,
    Example,
    GeneratedAxiomList,
    GeneratedExampleList,
    QualityGrade,
)
from sentinel.prompts import (
    AXIOM_SYSTEM,
    CONTRASTIVE_SYSTEM,
    QUALITY_SYSTEM,
    QUESTION_SYSTEM,
    build_axiom_user_prompt,
    build_contrastive_user_prompt,
    build_quality_user_prompt,
    build_question_user_prompt,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT_S = 300

# JSON-only suffix appended to every prompt. Claude follows "return only JSON"
# reliably, but markdown code fences sometimes leak in — we strip them after.
JSON_ONLY_SUFFIX = (
    "\n\nIMPORTANT: Return ONLY a single valid JSON object matching the schema "
    "described above. No prose before or after. No markdown code fences. "
    "No preamble. Just the raw JSON object."
)


def claude_code_available() -> bool:
    """True iff the `claude` CLI is on PATH."""
    return shutil.which("claude") is not None


def _strip_code_fences(text: str) -> str:
    """Strip leading/trailing ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    # Drop the opening fence (```json or ```).
    lines = lines[1:]
    # Drop the closing fence if present.
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _call_claude_code(prompt: str, timeout: int = DEFAULT_TIMEOUT_S) -> str:
    """Spawn `claude -p <prompt>`, return stdout text. Raises on non-zero exit."""
    result = subprocess.run(  # noqa: S603 — argv is a fixed list, no shell
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"`claude -p` exited {result.returncode}: {result.stderr[:500]}")
    return result.stdout


def _call_structured(
    system: str,
    user: str,
    output_format: type[T],
    timeout: int = DEFAULT_TIMEOUT_S,
) -> T:
    """Spawn Claude Code, parse stdout as JSON, validate against schema."""
    prompt = f"{system}\n\n---\n\n{user}{JSON_ONLY_SUFFIX}"
    raw = _call_claude_code(prompt, timeout=timeout)
    text = _strip_code_fences(raw)
    return output_format.model_validate_json(text)


class DataGenerator:
    """Synchronous, single-process data generator. Each method is one
    subprocess; for the full 5000-example run, drive these from a higher
    level loop (or use multiprocessing if you want parallel claude
    subprocesses)."""

    def __init__(self, timeout_s: int = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    def generate_axioms(
        self,
        n: int,
        id_offset: int = 0,
        shapes: list[AxiomShape] | None = None,
    ) -> list[Axiom]:
        user = build_axiom_user_prompt(n=n, id_offset=id_offset, shapes=shapes)
        result = _call_structured(AXIOM_SYSTEM, user, GeneratedAxiomList, self.timeout_s)
        return result.axioms

    def generate_questions(
        self,
        axiom: Axiom,
        n_questions: int,
        anti_regurgitation_fraction: float = 0.2,
    ) -> list[Example]:
        user = build_question_user_prompt(
            axiom=axiom,
            n_questions=n_questions,
            anti_regurgitation_fraction=anti_regurgitation_fraction,
        )
        result = _call_structured(QUESTION_SYSTEM, user, GeneratedExampleList, self.timeout_s)
        return result.examples

    def generate_contrastive_pair(
        self,
        name: str,
        pair_id: str,
        axiom_id_a: str,
        axiom_id_b: str,
    ) -> ContrastivePair:
        user = build_contrastive_user_prompt(
            name=name, pair_id=pair_id, axiom_id_a=axiom_id_a, axiom_id_b=axiom_id_b
        )
        return _call_structured(CONTRASTIVE_SYSTEM, user, ContrastivePair, self.timeout_s)

    def grade_example(
        self,
        axiom_text: str,
        question: str,
        answer: str,
    ) -> QualityGrade:
        user = build_quality_user_prompt(axiom_text=axiom_text, question=question, answer=answer)
        return _call_structured(QUALITY_SYSTEM, user, QualityGrade, self.timeout_s)
