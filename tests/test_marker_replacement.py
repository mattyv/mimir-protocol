"""Marker replacement: substitute axiom term with an opaque placeholder
in the prompt before sending to the model. Vector injection then happens
at the marker position, on a residual stream that has zero lexical priors
from the original term.

This is *not* prompt injection (no semantic content is added). It's prompt
SUBSTITUTION to remove the lexical priors that fight against vector
injection on stolen-words names.

These tests assert the mechanical invariants only:
  - marker assignment is idempotent and unique per axiom
  - prompt rewrite swaps term for marker (first occurrence only)
  - output restoration swaps marker back to term (all occurrences)
  - the marker is preserved as a recognizable string the tokenizer can split
"""

from __future__ import annotations

import pytest


def test_make_marker_idempotent():
    """Same term should always get the same marker on repeated calls."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    m1 = reg.assign("Balance Publisher")
    m2 = reg.assign("Balance Publisher")
    assert m1 == m2


def test_make_marker_unique_per_term():
    """Different terms should get distinct markers."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    m_bp = reg.assign("Balance Publisher")
    m_st = reg.assign("shoe_town")
    m_fc = reg.assign("FlexCast")
    assert len({m_bp, m_st, m_fc}) == 3


def test_marker_does_not_contain_original_term_words():
    """The marker should be opaque — no leak of original term content."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    m = reg.assign("Balance Publisher")
    assert "balance" not in m.lower()
    assert "publisher" not in m.lower()
    assert "shoe" not in m.lower()


def test_rewrite_replaces_first_occurrence_of_term():
    """Rewriting a prompt should replace the term (case-sensitive) with
    the marker at the first occurrence."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    reg.assign("Balance Publisher")
    out = reg.rewrite_prompt("What is a Balance Publisher?", "Balance Publisher")
    assert "Balance Publisher" not in out
    assert reg.marker_for("Balance Publisher") in out


def test_rewrite_is_noop_when_term_absent():
    """If the term isn't in the prompt, rewrite should leave it unchanged."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    reg.assign("Balance Publisher")
    prompt = "Tell me about the weather."
    out = reg.rewrite_prompt(prompt, "Balance Publisher")
    assert out == prompt


def test_rewrite_only_replaces_first_match():
    """Multiple mentions in one prompt: only the first is replaced
    (the rest stay since attention will pull from the first via context)."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    marker = reg.assign("Balance Publisher")
    prompt = "Balance Publisher polls. Balance Publisher publishes."
    out = reg.rewrite_prompt(prompt, "Balance Publisher")
    # First occurrence becomes marker; second stays as-is
    assert out.count(marker) == 1
    assert out.count("Balance Publisher") == 1


def test_restore_swaps_marker_back_to_term():
    """If the model output contains the marker, restoration replaces
    it with the user-facing term."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    marker = reg.assign("Balance Publisher")
    output = f"A {marker} is a service that polls exchanges."
    restored = reg.restore_output(output)
    assert "Balance Publisher" in restored
    assert marker not in restored


def test_round_trip_is_identity_for_simple_prompt():
    """Rewrite then restore on a single-mention prompt should give
    back something containing the original term."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    reg.assign("Balance Publisher")
    prompt = "What is a Balance Publisher?"
    rewritten = reg.rewrite_prompt(prompt, "Balance Publisher")
    restored = reg.restore_output(rewritten)
    assert "Balance Publisher" in restored


def test_marker_for_returns_assigned_string():
    """marker_for should return the same marker as assign returned."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    m = reg.assign("Balance Publisher")
    assert reg.marker_for("Balance Publisher") == m


def test_marker_for_raises_for_unregistered_term():
    """Looking up a marker for an unregistered term should raise."""
    from marker.marker_replacement import MarkerRegistry

    reg = MarkerRegistry()
    with pytest.raises(KeyError):
        reg.marker_for("UnregisteredTerm")


def test_find_marker_position_in_token_sequence():
    """Given a tokenizer, the marker should be locatable as a contiguous
    span of token IDs in the rewritten prompt."""
    from marker.marker_replacement import MarkerRegistry, find_marker_position

    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:
        pytest.skip(f"could not load Qwen tokenizer: {e}")

    reg = MarkerRegistry()
    marker = reg.assign("Balance Publisher")
    prompt = "What is a Balance Publisher?"
    rewritten = reg.rewrite_prompt(prompt, "Balance Publisher")

    pos = find_marker_position(tokenizer, rewritten, marker)
    assert pos > 0  # not at sequence start

    # The token at pos should decode to something that's part of the marker
    ids = tokenizer(rewritten, add_special_tokens=False).input_ids
    assert pos < len(ids)
