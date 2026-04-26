"""Tests for token-trigger-based injection — the runtime path that does NOT
rely on user-facing markers. The model sees the user's free text; we scan the
tokenized stream for any registered axiom term and inject at those positions."""

from __future__ import annotations

import numpy as np

from marker.trigger_inject import Registry, find_matches


def make_registry(name_to_variants: dict[str, list[list[int]]]) -> Registry:
    reg = Registry()
    for name, variants in name_to_variants.items():
        reg._add_term(name, variants, vector=np.zeros(4, dtype=np.float32))
    return reg


def test_find_matches_single_term():
    reg = make_registry({"foo": [[10, 20]]})
    matches = find_matches([1, 10, 20, 3], reg)
    assert matches == [(1, 3, "foo")]


def test_find_matches_multiple_occurrences():
    reg = make_registry({"foo": [[10, 20]]})
    matches = find_matches([10, 20, 5, 10, 20], reg)
    assert [(s, e) for s, e, _ in matches] == [(0, 2), (3, 5)]


def test_find_matches_no_match():
    reg = make_registry({"foo": [[10, 20]]})
    matches = find_matches([1, 2, 3], reg)
    assert matches == []


def test_find_matches_longest_wins_on_overlap():
    reg = make_registry({"foo": [[10, 20]], "foobar": [[10, 20, 30]]})
    matches = find_matches([10, 20, 30], reg)
    assert len(matches) == 1
    assert matches[0][2] == "foobar"


def test_find_matches_multiple_variants_same_term():
    reg = make_registry({"foo": [[10, 20], [11, 21]]})  # cased variants
    matches = find_matches([1, 11, 21, 5, 10, 20], reg)
    assert [(s, e, n) for s, e, n in matches] == [(1, 3, "foo"), (4, 6, "foo")]


def test_find_matches_empty_variant_skipped():
    reg = make_registry({"foo": [[]]})
    matches = find_matches([1, 2, 3], reg)
    assert matches == []


def test_registry_carries_optional_prior():
    reg = Registry()
    reg._add_term(
        "foo",
        [[10, 20]],
        vector=np.zeros(4, dtype=np.float32),
        prior=np.ones(4, dtype=np.float32),
    )
    assert reg.entries[0].prior is not None
    assert reg.entries[0].prior.shape == (4,)


def test_registry_prior_defaults_to_none():
    reg = Registry()
    reg._add_term("foo", [[10, 20]], vector=np.zeros(4, dtype=np.float32))
    assert reg.entries[0].prior is None
