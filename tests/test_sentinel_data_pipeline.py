"""Tests for the training-data encoding pipeline.

The loss mask is the load-bearing invariant — if the prefix isn't masked,
the LoRA learns to copy questions instead of answer them; if the answer
isn't unmasked, it learns nothing at all. Both failure modes are silent
(training proceeds with low loss / nan loss).

Tests assert:
  - prefix tokens (sentinel + question) get label = -100
  - answer tokens get their actual id as label
  - the EOS token is included in the answer (otherwise greedy never stops)
  - input_ids and labels have identical length
  - the boundary between prefix and answer is at the right position
"""

from __future__ import annotations

from sentinel.data_pipeline import IGNORE_INDEX, encode_example
from sentinel.data_schema import Example


def _make_example() -> Example:
    return Example(
        axiom_id="ax_0001",
        type="base",
        sentinel_block="<sentinel>X is Y.</sentinel>",
        question="What is X?",
        answer="X is Y.",
    )


class _FakeTokenizer:
    """Word-level tokenizer for tests — keeps the alignment math obvious.
    Each whitespace-separated word becomes one int id by hash."""

    def __init__(self, eos_token: str = "<eos>") -> None:
        self.eos_token = eos_token
        self._vocab: dict[str, int] = {}
        # Reserve special token IDs.
        self._intern(eos_token)

    def _intern(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    @property
    def eos_token_id(self) -> int:
        return self._vocab[self.eos_token]

    def __call__(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        ids = [self._intern(tok) for tok in text.split()]
        return _FakeBatch(ids)


class _FakeBatch:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


def test_encoded_lengths_match() -> None:
    tok = _FakeTokenizer()
    out = encode_example(_make_example(), tok)
    assert len(out["input_ids"]) == len(out["labels"])


def test_prefix_tokens_are_masked() -> None:
    """Every token before the answer starts must have label = IGNORE_INDEX."""
    tok = _FakeTokenizer()
    ex = _make_example()
    out = encode_example(ex, tok)

    prefix = f"{ex.sentinel_block}\n{ex.question}\n"
    n_prefix_tokens = len(tok(prefix).input_ids)
    assert all(label == IGNORE_INDEX for label in out["labels"][:n_prefix_tokens])


def test_answer_tokens_are_unmasked() -> None:
    """Every token from the answer-start onward (incl. EOS) must have a real label."""
    tok = _FakeTokenizer()
    ex = _make_example()
    out = encode_example(ex, tok)

    prefix = f"{ex.sentinel_block}\n{ex.question}\n"
    n_prefix_tokens = len(tok(prefix).input_ids)
    answer_labels = out["labels"][n_prefix_tokens:]
    assert all(label != IGNORE_INDEX for label in answer_labels)
    assert len(answer_labels) > 0


def test_answer_label_ids_match_answer_input_ids() -> None:
    """The label values on answer positions must equal the input_ids on
    those same positions — that's how the LM learns 'predict this'."""
    tok = _FakeTokenizer()
    ex = _make_example()
    out = encode_example(ex, tok)

    prefix = f"{ex.sentinel_block}\n{ex.question}\n"
    n_prefix_tokens = len(tok(prefix).input_ids)
    for i in range(n_prefix_tokens, len(out["input_ids"])):
        assert out["labels"][i] == out["input_ids"][i]


def test_eos_appended_to_answer() -> None:
    """Without an EOS, greedy generation never stops at training-time
    answer boundaries. Confirm EOS is the last unmasked token."""
    tok = _FakeTokenizer()
    out = encode_example(_make_example(), tok)
    assert out["input_ids"][-1] == tok.eos_token_id
    assert out["labels"][-1] == tok.eos_token_id


def test_long_answer_has_more_unmasked_positions_than_short() -> None:
    """Sanity: a longer answer produces more unmasked positions."""
    tok = _FakeTokenizer()
    short = encode_example(
        Example(
            axiom_id="ax",
            type="base",
            sentinel_block="<sentinel>X.</sentinel>",
            question="Q?",
            answer="Y.",
        ),
        tok,
    )
    long = encode_example(
        Example(
            axiom_id="ax",
            type="base",
            sentinel_block="<sentinel>X.</sentinel>",
            question="Q?",
            answer="Y is a long answer with several words.",
        ),
        tok,
    )
    short_unmasked = sum(1 for label in short["labels"] if label != IGNORE_INDEX)
    long_unmasked = sum(1 for label in long["labels"] if label != IGNORE_INDEX)
    assert long_unmasked > short_unmasked
