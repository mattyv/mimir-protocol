"""Tests for the burst schedule + answer scoring (burst.py)."""

from __future__ import annotations

import pytest

from marker.burst import answers_match, extract_answer, make_schedule


def test_schedule_anchor_every_2():
    s = make_schedule(6, anchor_every=2)
    assert s == ["anchor", "latent", "anchor", "latent", "anchor", "latent"]


def test_schedule_step0_always_anchor_and_plain_is_all_anchor():
    assert make_schedule(4, anchor_every=1) == ["anchor"] * 4  # plain generation
    assert make_schedule(5, anchor_every=3)[0] == "anchor"


def test_schedule_rejects_zero():
    with pytest.raises(ValueError, match="anchor_every"):
        make_schedule(4, 0)


def test_extract_answer_prefers_hash_marker():
    assert extract_answer("blah 5 then 12\n#### 42") == "42"
    assert extract_answer("She has 3 apples and buys 7 more, total 10.") == "10"
    assert extract_answer("no numbers here") is None


def test_extract_answer_strips_commas():
    assert extract_answer("#### 1,000") == "1000"


def test_answers_match_numeric_and_tolerant():
    assert answers_match("42", "42")
    assert answers_match("42.0", "42")
    assert answers_match("1000", "1000")
    assert not answers_match("42", "43")
    assert not answers_match(None, "42")  # missing prediction is a miss
