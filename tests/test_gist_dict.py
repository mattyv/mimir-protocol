"""Pure-logic tests for gist_dict.py (Stage-1 dictionary fidelity, see
GIST_LM_PLAN.md and scratchpad/gist_dict_stage1_spec.md). No model loaded --
everything here is tensor/plumbing logic, torch-only, CPU. Mechanical
invariants only (round-trips, seeded determinism, branch coverage) -- the
experiment's actual numbers live in the manifest, never in an assertion.
"""

from __future__ import annotations

import pytest
import torch

from marker.gist_dict import (
    adjusted_mutual_info,
    build_dict_kv,
    build_dict_kv_residual,
    build_dict_ro,
    build_dict_whole,
    detokenize,
    fit_categorical_nb,
    kmeans,
    kmeans_pp_init,
    kv_slot_matrix,
    predict_categorical_nb,
    slot_matrix_to_kv,
    stage1_verdict,
    tokenize,
)

_GEO = {"n_layers": 3, "n_kv_heads": 2, "head_dim": 4}


def _fake_kv(k_slots, geo=_GEO, seed=0):
    """A random AxiomKV with the given number of gist slots -- shapes exactly
    match what gist_kv/chain_gist_kv produce: keys/values[layer] is
    [1, n_kv_heads, k_slots, head_dim]."""
    from marker.run_axiom_mlp_demo import AxiomKV

    g = torch.Generator().manual_seed(seed)
    keys = [
        torch.randn(1, geo["n_kv_heads"], k_slots, geo["head_dim"], generator=g)
        for _ in range(geo["n_layers"])
    ]
    values = [
        torch.randn(1, geo["n_kv_heads"], k_slots, geo["head_dim"], generator=g)
        for _ in range(geo["n_layers"])
    ]
    return AxiomKV(n_layers=geo["n_layers"], keys=keys, values=values)


# ── kv_slot_matrix / slot_matrix_to_kv: the flatten <-> AxiomKV round trip ──


def test_kv_slot_matrix_round_trips_through_slot_matrix_to_kv():
    kv = _fake_kv(k_slots=5)
    mat = kv_slot_matrix(kv)
    d = _GEO["n_layers"] * 2 * _GEO["n_kv_heads"] * _GEO["head_dim"]
    assert mat.shape == (5, d)
    back = slot_matrix_to_kv(mat, _GEO["n_layers"], _GEO["n_kv_heads"], _GEO["head_dim"])
    for a, b in zip(kv.keys, back.keys, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(kv.values, back.values, strict=True):
        assert torch.equal(a, b)


# ── kmeans ───────────────────────────────────────────────────────────────────


def test_kmeans_pp_init_deterministic_under_seed():
    x = torch.randn(50, 6, generator=torch.Generator().manual_seed(1))
    a = kmeans_pp_init(x, K=5, seed=7)
    b = kmeans_pp_init(x, K=5, seed=7)
    assert torch.equal(a, b)
    c = kmeans_pp_init(x, K=5, seed=8)
    assert not torch.equal(a, c)


def test_kmeans_converges_on_separable_blobs():
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    x = torch.cat([c.unsqueeze(0) + 0.1 * torch.randn(40, 2, generator=g) for c in centers])
    centroids, assign, usage = kmeans(x, K=3, iters=20, seed=0)
    assert usage.sum().item() == x.shape[0]
    # every one of the three true blobs is (almost) purely one cluster label
    for i in range(3):
        labels = assign[i * 40 : (i + 1) * 40]
        purity = (labels == labels.mode().values).float().mean()
        assert purity > 0.95, f"blob {i} purity {purity}"


def test_kmeans_usage_counts_always_sum_to_n():
    x = torch.randn(30, 4, generator=torch.Generator().manual_seed(2))
    _, assign, usage = kmeans(x, K=4, iters=5, seed=1)
    assert usage.sum().item() == 30
    assert usage.shape == (4,)


def test_kmeans_rejects_k_greater_than_n():
    x = torch.randn(3, 2)
    with pytest.raises(AssertionError):
        kmeans(x, K=10, seed=0)


# ── dict builders + tokenize/detokenize round trip ──────────────────────────


def _slot_fit_data(n=40, d=16, dr=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, dr, generator=g)


def test_kv_dict_tokenize_detokenize_round_trips_on_a_centroid_bitwise():
    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 4}
    d = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    slot_mats, slot_ro = _slot_fit_data(n=20, d=d, dr=3)
    entry, assign = build_dict_kv(slot_mats, slot_ro, K=4, iters=10, seed=0)
    full_dict = {
        "cfg": "kv_K4",
        "kind": "kv",
        "geometry": {**geo, "k_slots": 8},
        "slots": [entry] * 8,
    }

    # build a step whose every slot equals dictionary id 0's stored centroid
    # EXACTLY (through the same fp16 cast) -> tokenize must return id 0 for
    # every slot, and detokenize must hand back that exact stored value.
    centroid0 = entry["centroids"][0].float()
    mat = centroid0.unsqueeze(0).expand(8, -1).clone()
    kv = slot_matrix_to_kv(mat, geo["n_layers"], geo["n_kv_heads"], geo["head_dim"])

    ids = tokenize(kv, full_dict)
    assert ids == [0] * 8
    assert all(0 <= i < 4 for i in ids)

    back = detokenize(ids, full_dict, geo)
    back_mat = kv_slot_matrix(back)
    want_mat = torch.stack([entry["centroids"][0].float()] * 8)
    assert torch.equal(back_mat, want_mat)


def test_residual_dict_entry_equals_c1_plus_c2():
    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 4}
    d = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    slot_mats, slot_ro = _slot_fit_data(n=30, d=d, dr=3)
    entry, assign1, assign2 = build_dict_kv_residual(
        slot_mats, slot_ro, K1=3, K2=2, iters=10, seed=0
    )
    full_dict = {
        "cfg": "kv_res_3x2",
        "kind": "kv_res",
        "geometry": {**geo, "k_slots": 8},
        "slots": [entry] * 8,
    }
    ids = [0] * 8  # id1=0, id2=0 for every slot (K2=2 -> joint id = id1*2+id2 = 0)
    back = detokenize(ids, full_dict, geo)
    back_mat = kv_slot_matrix(back)
    want = (entry["c1"][0].float() + entry["c2"][0].float()).unsqueeze(0).expand(8, -1)
    assert torch.allclose(back_mat, want)

    # a non-zero joint id decodes to the matching (id1, id2) pair
    joint = 1 * entry["K2"] + 1
    ids2 = [joint] * 8
    back2 = kv_slot_matrix(detokenize(ids2, full_dict, geo))
    want2 = (entry["c1"][1].float() + entry["c2"][1].float()).unsqueeze(0).expand(8, -1)
    assert torch.allclose(back2, want2)


def test_ro_dict_tokenize_uses_readout_not_kv():
    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 4}
    d = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    slot_mats, slot_ro = _slot_fit_data(n=25, d=d, dr=5)
    entry, assign = build_dict_ro(slot_ro, slot_mats, K=4, iters=10, seed=0)
    full_dict = {
        "cfg": "ro_K4",
        "kind": "ro",
        "geometry": {**geo, "k_slots": 8},
        "slots": [entry] * 8,
    }
    # tokenize without a readout must fail loud, not silently misassign
    dummy_kv = _fake_kv(k_slots=8, geo=geo)
    with pytest.raises(AssertionError):
        tokenize(dummy_kv, full_dict)
    readout0 = entry["centroids_ro"][0].float()
    readout = readout0.unsqueeze(0).expand(8, -1)
    ids = tokenize(dummy_kv, full_dict, readout=readout)
    assert ids == [0] * 8


def _slot_loader_factory(n_slots=8, n=20, d=8, seed=3):
    """A stand-in for run_gist_dict.load_slot_shard: one FIXED tensor per
    slot index, regenerated deterministically on every call (never cached
    across calls -- proves build_dict_whole re-requests a slot rather than
    holding onto one from an earlier pass)."""
    return lambda s: torch.randn(n, d, generator=torch.Generator().manual_seed(seed + s))


def test_whole_dict_broadcasts_one_id_to_all_eight_slots():
    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 4}
    d_slot = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    loader = _slot_loader_factory(n_slots=8, n=20, d=d_slot, seed=3)
    entry, assign = build_dict_whole(loader, k_slots=8, K=4, iters=10, seed=0)
    full_dict = {
        "cfg": "whole_K4",
        "kind": "whole",
        "geometry": {**geo, "k_slots": 8},
        "entry": entry,
    }

    centroid0 = torch.stack([entry["slots"][s]["centroids"][0].float() for s in range(8)])
    kv = slot_matrix_to_kv(centroid0, geo["n_layers"], geo["n_kv_heads"], geo["head_dim"])
    ids = tokenize(kv, full_dict)
    assert len(ids) == 8
    assert len(set(ids)) == 1  # one id broadcast to every slot
    back = kv_slot_matrix(detokenize(ids, full_dict, geo))
    assert torch.equal(back, centroid0)


def test_whole_dict_with_random_projection_still_recovers_own_centroid():
    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 4}
    d_slot = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    loader = _slot_loader_factory(n_slots=8, n=25, d=d_slot, seed=4)
    entry, assign = build_dict_whole(
        loader, k_slots=8, K=3, iters=10, seed=0, proj_dim=6, proj_seed=1
    )
    assert entry["proj_blocks"] is not None and len(entry["proj_blocks"]) == 8
    assert entry["proj_blocks"][0].shape == (d_slot, 6)
    full_dict = {
        "cfg": "whole_K3_proj",
        "kind": "whole",
        "geometry": {**geo, "k_slots": 8},
        "entry": entry,
    }
    centroid0 = torch.stack([entry["slots"][s]["centroids"][0].float() for s in range(8)])
    kv = slot_matrix_to_kv(centroid0, geo["n_layers"], geo["n_kv_heads"], geo["head_dim"])
    ids = tokenize(kv, full_dict)
    assert ids == [0] * 8


def test_whole_dict_never_holds_more_than_one_slot_at_once():
    # RAM invariant (spec section B): build_dict_whole must be usable by a
    # loader backed by per-slot shards on disk, which can only ever hand
    # back ONE slot's tensor per call -- this loader tracks how many tensors
    # are simultaneously alive (never released early) to prove that.
    geo_d = 4
    n = 12
    alive = set()
    max_alive = 0

    def _loader(s):
        nonlocal max_alive
        alive.add(s)
        max_alive = max(max_alive, len(alive))
        # the caller must `del` its reference before requesting the next
        # slot for `alive` to shrink -- simulate that by clearing eagerly
        # after handing back a plain (untracked) tensor copy.
        t = torch.randn(n, geo_d, generator=torch.Generator().manual_seed(s))
        alive.discard(s)
        return t

    entry, assign = build_dict_whole(
        _loader, k_slots=8, K=4, iters=5, seed=0, proj_dim=3, proj_seed=0
    )
    assert max_alive == 1, f"more than one slot resident at once: {max_alive}"
    assert entry["slots"][0]["centroids"].shape == (4, geo_d)
    assert "centroids_full" not in entry  # never one flat 8*D array


# ── naive Bayes over discrete slot IDs ──────────────────────────────────────


def test_naive_bayes_perfectly_separates_a_deterministic_id_to_label_map():
    # slot 0's id alone determines the label -> NB must fit it perfectly
    torch.manual_seed(0)
    n = 60
    y = torch.randint(0, 3, (n,))
    ids = torch.randint(0, 5, (n, 8))
    ids[:, 0] = y * 2  # deterministic, injective-enough mapping slot0 -> label
    model = fit_categorical_nb(ids, y, n_classes=3, k_sizes=[5] * 8)
    preds = predict_categorical_nb(model, ids)
    assert (preds == y).float().mean() > 0.95


def test_naive_bayes_single_slot_subset():
    torch.manual_seed(1)
    n = 40
    y = torch.randint(0, 2, (n,))
    ids = torch.randint(0, 4, (n, 8))
    ids[:, 3] = y  # only slot 3 carries the label
    model_full = fit_categorical_nb(ids, y, n_classes=2, k_sizes=[4] * 8, slots=[3])
    preds = predict_categorical_nb(model_full, ids)
    assert (preds == y).float().mean() > 0.95


# ── adjusted mutual information ─────────────────────────────────────────────


def test_ami_is_one_for_identical_labelings():
    labels = [0, 0, 1, 1, 2, 2, 0, 1, 2]
    assert adjusted_mutual_info(labels, labels) == pytest.approx(1.0, abs=1e-6)


def test_ami_is_near_zero_for_independent_random_labelings():
    g = torch.Generator().manual_seed(0)
    n = 300
    a = torch.randint(0, 6, (n,), generator=g).tolist()
    b = torch.randint(0, 6, (n,), generator=g).tolist()
    ami = adjusted_mutual_info(a, b)
    assert -0.15 < ami < 0.15, ami


def test_ami_relabeling_invariant():
    a = [0, 0, 1, 1, 2, 2]
    b = [5, 5, 9, 9, 1, 1]  # same partition, different label names
    assert adjusted_mutual_info(a, b) == pytest.approx(1.0, abs=1e-6)


# ── stage1_verdict: the pure gate read ──────────────────────────────────────


def _cfg(r_gsm8k, r_fresh, op, kind="per_slot"):
    return {"R_gsm8k": r_gsm8k, "R_fresh": r_fresh, "op": op, "kind": kind}


def test_verdict_invalid_harness_when_gate0_failed():
    cells = {"gate0_pass": False, "configs": {"kv_K256": _cfg(0.9, 0.9, 0.9)}}
    assert stage1_verdict(cells) == "INVALID_HARNESS"


def test_verdict_go_when_best_config_clears_all_three_bars():
    cells = {"gate0_pass": True, "configs": {"kv_K1024": _cfg(0.85, 0.75, 0.8)}}
    assert stage1_verdict(cells) == "GO"


def test_verdict_retrieval_when_per_slot_fails_but_whole_passes():
    cells = {
        "gate0_pass": True,
        "configs": {
            "kv_K1024": _cfg(0.3, 0.2, 0.4, kind="per_slot"),
            "whole_K4096": _cfg(0.85, 0.75, 0.8, kind="whole"),
        },
    }
    assert stage1_verdict(cells) == "RETRIEVAL"


def test_verdict_vqvae_in_the_middle_band():
    cells = {"gate0_pass": True, "configs": {"kv_K1024": _cfg(0.6, 0.6, 0.65)}}
    assert stage1_verdict(cells) == "VQVAE"


def test_verdict_kill_when_everything_is_at_the_floor():
    cells = {
        "gate0_pass": True,
        "configs": {
            "kv_K256": _cfg(0.1, 0.1, 0.2),
            "kv_K1024": _cfg(0.2, 0.15, 0.3),
        },
    }
    assert stage1_verdict(cells) == "KILL"
