"""Mechanical invariants for `composed_description` — the helper that
builds a coherent document from a top-level axiom + its sub-axioms.
"""

from __future__ import annotations


def test_leaf_axiom_returns_own_description():
    from marker.axiom_registry import HIERARCHICAL_AXIOMS, composed_description

    assert composed_description("event_log") == HIERARCHICAL_AXIOMS["event_log"]["description"]


def test_composed_axiom_includes_subaxioms_and_note():
    from marker.axiom_registry import HIERARCHICAL_AXIOMS, composed_description

    out = composed_description("data_pipeline")
    cfg = HIERARCHICAL_AXIOMS["data_pipeline"]
    assert cfg["description"] in out
    for sk in cfg["composed_of"]:
        sub = HIERARCHICAL_AXIOMS[sk]
        assert sub["term"] in out
        assert sub["description"] in out
    assert cfg["composition_note"] in out


def test_lookup_finds_axioms_in_any_registry():
    """composed_description must resolve keys from HIERARCHICAL, CHAIN,
    and AXIOMS registries — composed_of can reference any of them."""
    from marker.axiom_registry import composed_description

    # CHAIN_AXIOMS leaf
    out = composed_description("balance_publisher")
    assert "Balance Publisher" in out

    # New CHAIN_AXIOMS top-level composed of CHAIN sub-axioms
    out_top = composed_description("trading_pipeline")
    assert "BalancePublisher" in out_top or "Balance Publisher" in out_top
    assert "TradingRiskEngine" in out_top
    assert "OrderSequencer" in out_top
    # composition_note must be present
    assert "fit together" in out_top


def test_unknown_axiom_raises():
    from marker.axiom_registry import composed_description

    try:
        composed_description("__nonexistent_axiom__")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_subaxioms_appear_in_declared_order():
    from marker.axiom_registry import HIERARCHICAL_AXIOMS, composed_description

    out = composed_description("data_pipeline")
    sub_keys = HIERARCHICAL_AXIOMS["data_pipeline"]["composed_of"]
    positions = [out.find(HIERARCHICAL_AXIOMS[sk]["description"]) for sk in sub_keys]
    assert positions == sorted(positions), f"sub-axioms out of order: {positions}"
    assert all(p >= 0 for p in positions)
