"""Tokenization invariants for GPT-2 small.

The POC depends on each target ` appear` / ` look` / ` seem` / ` avoid` / ` fake` /
` work` being a single BPE token so that we can read its logit at the final
position. If BPE splits any of them, the experiment design changes (we'd be
measuring a multi-token sequence probability, not a single-token logit shift).
"""

import pytest
from transformers import GPT2Tokenizer

TARGETS = [" appear", " look", " seem", " avoid", " fake", " work"]


@pytest.fixture(scope="module")
def tok() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.mark.parametrize("target", TARGETS)
def test_target_is_single_bpe_token(tok: GPT2Tokenizer, target: str) -> None:
    ids = tok(target, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"target {target!r} encodes to {ids}, expected one token"


def test_jotp_acronym_tokenization_is_known(tok: GPT2Tokenizer) -> None:
    """We don't require JOTP to be one token, but we pin its tokenisation
    so a future BPE / vocab change is loud rather than silent."""
    ids = tok("JOTP", add_special_tokens=False).input_ids
    assert ids == [41, 2394, 47], f"JOTP tokenisation drifted: {ids}"
