"""Tests for the front-loaded context test helpers (run_frontload.py)."""

from __future__ import annotations

import pytest

from marker.run_frontload import answer_done, context_split


def test_context_split_half_capped_and_bounded():
    assert context_split(3) == 2  # ceil(3/2)=2
    assert context_split(4) == 2
    assert context_split(6) == 3
    assert context_split(12) == 4  # capped
    assert context_split(12, cap=6) == 6


def test_context_split_leaves_a_step_for_the_model():
    # never consume the whole solution as context
    assert context_split(3) < 3
    with pytest.raises(ValueError, match="steps"):
        context_split(2)


def test_answer_done_requires_marker_digit_and_newline():
    assert not answer_done("still thinking about 5 things")
    assert not answer_done("#### ")  # marker but no digit
    assert not answer_done("#### 42")  # no newline yet — number may continue
    assert answer_done("blah\n#### 42\n")
    assert answer_done("#### 1,000\nnext")
