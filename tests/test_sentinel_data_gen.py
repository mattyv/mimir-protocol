"""Tests for DataGenerator that don't require the `claude` CLI.

Each test patches `subprocess.run` at the module level so we control
what Claude Code "returns" from stdout. We verify:

  - the right system prompt is included in the spawned argv
  - the JSON-only suffix is appended
  - markdown code fences are stripped from responses
  - structured output flows through Pydantic validation
  - non-zero exit codes raise loudly

End-to-end behaviour with a real `claude` subprocess lives in a smoke
script (see `scripts/sentinel_smoke.py`) — gated on the CLI being
present on PATH.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sentinel.data_gen import (
    DataGenerator,
    _strip_code_fences,
)
from sentinel.data_schema import Axiom, AxiomShape


def _stub_run(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


def test_strip_code_fences_handles_json_block() -> None:
    text = '```json\n{"foo": 1}\n```'
    assert _strip_code_fences(text) == '{"foo": 1}'


def test_strip_code_fences_handles_unlabeled_block() -> None:
    text = '```\n{"foo": 1}\n```'
    assert _strip_code_fences(text) == '{"foo": 1}'


def test_strip_code_fences_passthrough_when_no_fences() -> None:
    text = '{"foo": 1}'
    assert _strip_code_fences(text) == '{"foo": 1}'


def test_strip_code_fences_handles_fence_without_closing() -> None:
    """Some models open a fence but forget to close it. Don't crash; strip what we can."""
    text = '```json\n{"foo": 1}'
    assert _strip_code_fences(text) == '{"foo": 1}'


@patch("sentinel.data_gen.subprocess.run")
def test_generate_axioms_invokes_claude_p(mock_run: MagicMock) -> None:
    mock_run.return_value = _stub_run(
        '{"axioms": [{"id": "ax_0001", "shape": "definitional", "name": "fazbuzza", "text": "X is Y."}]}'
    )

    gen = DataGenerator()
    axioms = gen.generate_axioms(n=1)

    # Verify the subprocess was spawned with `claude -p <prompt>`.
    args, kwargs = mock_run.call_args
    argv = args[0]
    assert argv[0] == "claude"
    assert argv[1] == "-p"
    prompt = argv[2]
    # Prompt includes the system prompt content.
    assert "made-up names" in prompt
    # And the JSON-only enforcement suffix.
    assert "Return ONLY a single valid JSON" in prompt

    assert len(axioms) == 1
    assert axioms[0].name == "fazbuzza"


@patch("sentinel.data_gen.subprocess.run")
def test_response_with_code_fences_is_parsed(mock_run: MagicMock) -> None:
    """Models often wrap JSON in ```json blocks despite the instruction."""
    mock_run.return_value = _stub_run('```json\n{"axioms": []}\n```')
    gen = DataGenerator()
    assert gen.generate_axioms(n=1) == []


@patch("sentinel.data_gen.subprocess.run")
def test_nonzero_exit_raises_runtime_error(mock_run: MagicMock) -> None:
    mock_run.return_value = _stub_run("", returncode=1, stderr="auth failed")
    gen = DataGenerator()
    with pytest.raises(RuntimeError, match="exited 1"):
        gen.generate_axioms(n=1)


@patch("sentinel.data_gen.subprocess.run")
def test_invalid_json_response_raises_validation_error(mock_run: MagicMock) -> None:
    """If Claude returns prose instead of JSON, Pydantic should reject it
    rather than silently emit malformed records."""
    from pydantic import ValidationError

    mock_run.return_value = _stub_run("Sure! Here are some axioms.")
    gen = DataGenerator()
    with pytest.raises(ValidationError):
        gen.generate_axioms(n=1)


@patch("sentinel.data_gen.subprocess.run")
def test_generate_questions_includes_axiom_in_prompt(mock_run: MagicMock) -> None:
    mock_run.return_value = _stub_run('{"examples": []}')

    gen = DataGenerator()
    a = Axiom(id="ax_0001", shape=AxiomShape.DEFINITIONAL, name="fazbuzza", text="X is Y.")
    gen.generate_questions(axiom=a, n_questions=5)

    prompt = mock_run.call_args.args[0][2]
    assert "fazbuzza" in prompt
    assert "ax_0001" in prompt
    assert "X is Y." in prompt


@patch("sentinel.data_gen.subprocess.run")
def test_grade_example_validates_grade_schema(mock_run: MagicMock) -> None:
    mock_run.return_value = _stub_run(
        '{"requires_axiom": 5, "could_produce_without": 1, '
        '"parrots_or_reasons": "reasons", "rationale": "needs the axiom"}'
    )
    gen = DataGenerator()
    grade = gen.grade_example("X is Y.", "What is X?", "Y.")
    assert grade.requires_axiom == 5
    assert grade.parrots_or_reasons == "reasons"


@patch("sentinel.data_gen.subprocess.run")
def test_timeout_propagates(mock_run: MagicMock) -> None:
    """Timeout should be passed to subprocess.run as configured on the
    generator. Default is 300s."""
    mock_run.return_value = _stub_run('{"axioms": []}')
    gen = DataGenerator(timeout_s=42)
    gen.generate_axioms(n=1)
    assert mock_run.call_args.kwargs["timeout"] == 42
