"""Tests for marker insertion + position-finding (deterministic parts).

The actual extraction outcome is the experiment; we don't unit-test that.
"""

from __future__ import annotations

from marker.markers import (
    CLOSE_MARKER,
    OPEN_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)


def test_wrap_single_occurrence() -> None:
    out = wrap_term_in_paraphrase(
        "When Sarah wanted to dodge work, she'd lean on JOTP all afternoon.",
        ["JOTP"],
    )
    assert "[[JOTP]]" in out
    assert out.count(OPEN_MARKER) == 1
    assert out.count(CLOSE_MARKER) == 1


def test_wrap_multiple_occurrences_in_one_paraphrase() -> None:
    out = wrap_term_in_paraphrase("JOTP is JOTP because JOTP says so.", ["JOTP"])
    assert out == "[[JOTP]] is [[JOTP]] because [[JOTP]] says so."


def test_wrap_idempotent() -> None:
    """Already-wrapped terms must not get double-wrapped on a second pass."""
    once = wrap_term_in_paraphrase("X is JOTP today.", ["JOTP"])
    twice = wrap_term_in_paraphrase(once, ["JOTP"])
    assert once == twice


def test_wrap_longest_variant_first() -> None:
    """If both 'JOTP' and 'Just Out of Time Processing' are variants, the
    expansion should be wrapped as a single unit, not have 'Out' wrapped
    separately by accident."""
    out = wrap_term_in_paraphrase(
        "Just Out of Time Processing is a workplace pattern; JOTP for short.",
        ["JOTP", "Just Out of Time Processing"],
    )
    assert "[[Just Out of Time Processing]]" in out
    assert "[[JOTP]]" in out


def test_find_close_marker_returns_last_token_index() -> None:
    """If close-marker tokenises to [3, 4], a sequence ending in [...3, 4]
    should report the index of 4."""
    token_ids = [1, 2, 3, 4, 5, 6, 7, 3, 4]
    close = [3, 4]
    positions = find_close_marker_positions(token_ids, close)
    assert positions == [3, 8]  # the index of `4` in each `3, 4` pair


def test_find_close_marker_single_token() -> None:
    token_ids = [1, 2, 99, 3, 4, 99, 5]
    close = [99]
    assert find_close_marker_positions(token_ids, close) == [2, 5]


def test_find_close_marker_no_match() -> None:
    assert find_close_marker_positions([1, 2, 3], [9, 9]) == []
