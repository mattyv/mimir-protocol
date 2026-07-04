"""Mechanical invariants for the prefix-tuning POC.

Model-free: exercises AxiomPrefix shapes, stat-matched/subsample init, that
cache-building keeps a live grad_fn (no silent detach — the whole training
loop is worthless if this breaks), save/load, and the train/eval leakage
guard + minimum-diversity check on the POC's own axiom data.
"""

from __future__ import annotations

import re

import torch

from marker.prefix_poc import (
    build_prefix_cache,
    init_stat_matched,
    init_subsample,
    load_axiom_prefix,
    save_axiom_prefix,
)
from marker.run_axiom_mlp_demo import AxiomKV
from marker.run_prefix_poc import PREFIX_AXIOMS

DIMS = {"n_layers": 3, "n_kv_heads": 2, "head_dim": 8}


def _fake_kv(n_tokens: int = 20, seed: int = 0) -> AxiomKV:
    g = torch.Generator().manual_seed(seed)
    shape = (1, DIMS["n_kv_heads"], n_tokens, DIMS["head_dim"])
    return AxiomKV(
        n_layers=DIMS["n_layers"],
        keys=[torch.randn(shape, generator=g) for _ in range(DIMS["n_layers"])],
        values=[torch.randn(shape, generator=g) for _ in range(DIMS["n_layers"])],
    )


def _cache_layer0_key(cache):  # noqa: ANN001, ANN201
    """Pull layer-0 K out of a DynamicCache across transformers versions.

    Newer transformers (>=5, installed locally) exposes cache.layers[i].keys.
    Older transformers (pinned <5 on the Vast runners, matching the rest of
    this project's cache code) only has to_legacy_cache() -> tuple-of-tuples.
    """
    if hasattr(cache, "layers"):
        return cache.layers[0].keys
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    return legacy[0][0]


# ── Init ──────────────────────────────────────────────────────────────────────


def test_stat_matched_init_shapes():
    real_kv = _fake_kv()
    prefix = init_stat_matched(real_kv, n_tokens=4, term="Foo")
    assert prefix.n_layers == DIMS["n_layers"]
    for layer in range(prefix.n_layers):
        assert prefix.keys[layer].shape == (1, DIMS["n_kv_heads"], 4, DIMS["head_dim"])
        assert prefix.values[layer].shape == (1, DIMS["n_kv_heads"], 4, DIMS["head_dim"])
        assert prefix.keys[layer].requires_grad
        assert prefix.values[layer].requires_grad


def test_stat_matched_init_matches_target_scale():
    real_kv = _fake_kv(n_tokens=200, seed=1)  # large N for a stable stat estimate
    prefix = init_stat_matched(real_kv, n_tokens=64, term="Foo", seed=2)
    for layer in range(prefix.n_layers):
        real_std = real_kv.keys[layer].std().item()
        init_std = prefix.keys[layer].detach().std().item()
        # Same order of magnitude, not exact (finite-sample + per-channel init).
        # This is the check that would have caught "unit-normal init breaks
        # attention because it's off the real KV's actual scale."
        assert 0.3 * real_std < init_std < 3.0 * real_std


def test_subsample_init_copies_real_values():
    real_kv = _fake_kv(n_tokens=20, seed=3)
    prefix = init_subsample(real_kv, n_tokens=5, term="Foo", seed=0)
    for layer in range(prefix.n_layers):
        assert prefix.keys[layer].shape[2] == 5
        assert prefix.keys[layer].requires_grad
        source = real_kv.keys[layer][0, 0]  # (seq, head_dim)
        got = prefix.keys[layer].detach()[0, 0]  # (5, head_dim)
        for row in got:
            assert any(torch.allclose(row, src_row) for src_row in source)


def test_subsample_init_wraps_when_desc_shorter_than_n():
    real_kv = _fake_kv(n_tokens=3, seed=4)
    prefix = init_subsample(real_kv, n_tokens=8, term="Foo", seed=0)
    assert prefix.keys[0].shape[2] == 8


# ── Cache building ────────────────────────────────────────────────────────────


def test_build_prefix_cache_preserves_grad_fn():
    real_kv = _fake_kv()
    prefix = init_stat_matched(real_kv, n_tokens=4, term="Foo")
    cache = build_prefix_cache(prefix, dtype=torch.float32)
    k0 = _cache_layer0_key(cache)
    assert k0.requires_grad


def test_build_prefix_cache_dtype_cast_keeps_grad():
    real_kv = _fake_kv()
    prefix = init_stat_matched(real_kv, n_tokens=4, term="Foo")
    cache = build_prefix_cache(prefix, dtype=torch.float64)
    k0 = _cache_layer0_key(cache)
    assert k0.dtype == torch.float64
    assert k0.requires_grad
    assert k0.grad_fn is not None  # cast is differentiable, not a detach


def test_build_prefix_cache_matches_param_values():
    real_kv = _fake_kv()
    prefix = init_stat_matched(real_kv, n_tokens=4, term="Foo")
    cache = build_prefix_cache(prefix, dtype=torch.float32)
    k0 = _cache_layer0_key(cache)
    assert torch.allclose(k0.detach(), prefix.keys[0].detach())


# ── Persistence ───────────────────────────────────────────────────────────────


def test_axiom_prefix_roundtrip(tmp_path):
    real_kv = _fake_kv()
    prefix = init_stat_matched(real_kv, n_tokens=4, term="Foo")
    p = tmp_path / "prefix.pt"
    save_axiom_prefix(prefix, p)
    loaded = load_axiom_prefix(p)
    assert loaded.term == "Foo"
    assert loaded.n_tokens == 4
    for layer in range(prefix.n_layers):
        assert torch.allclose(loaded.keys[layer].detach(), prefix.keys[layer].detach())
    assert loaded.keys[0].requires_grad


# ── Data hygiene (the POC is worthless if this fails) ────────────────────────

_CONTRACTIONS = {
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
    "it's": "it is",
}


def _normalize(q: str) -> str:
    q = q.lower()
    for contraction, expanded in _CONTRACTIONS.items():
        q = q.replace(contraction, expanded)
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def test_no_train_question_matches_eval_question():
    for axiom in PREFIX_AXIOMS:
        train = {_normalize(q) for q, _ in axiom["train_qa"]}
        for q, _gold in axiom["eval"]:
            assert _normalize(q) not in train, (
                f"{axiom['name']}: eval question {q!r} leaks into train_qa"
            )


def test_each_axiom_has_expanded_train_set():
    for axiom in PREFIX_AXIOMS:
        assert len(axiom["train_qa"]) >= 6, (
            f"{axiom['name']}: only {len(axiom['train_qa'])} train pairs — "
            "a prefix trained on <6 phrasings will just memorize them"
        )


def test_train_probes_are_verbatim_train_questions():
    for axiom in PREFIX_AXIOMS:
        train_qs = {q for q, _ in axiom["train_qa"]}
        for q, _gold in axiom["train_probes"]:
            assert q in train_qs, (
                f"{axiom['name']}: train_probe {q!r} is not a verbatim train_qa "
                "question — TRAINED-bucket scoring would be measuring the wrong thing"
            )


def test_train_probe_golds_match_eval_gold_style():
    # Same short-substring convention as eval, so TRAINED vs HELDOUT accuracy
    # is comparing like with like.
    for axiom in PREFIX_AXIOMS:
        for _q, gold in axiom["train_probes"]:
            assert gold == gold.strip()
            assert len(gold) < 30, "train_probes golds should be short substrings, not sentences"


def test_build_prefix_cache_batched_expand_keeps_grad():
    real_kv = _fake_kv()
    prefix = init_stat_matched(real_kv, n_tokens=4, term="Foo")
    cache = build_prefix_cache(prefix, dtype=torch.float32, batch=3)
    k0 = _cache_layer0_key(cache)
    assert k0.shape[0] == 3  # expanded batch dim
    assert k0.requires_grad
    # expanded rows are views of the same params, not copies with broken grads
    assert torch.equal(k0[0].detach(), k0[2].detach())
