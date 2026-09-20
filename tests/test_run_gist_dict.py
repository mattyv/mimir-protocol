"""Tests for run_gist_dict.py, the Stage-1 gist dictionary fidelity harness
(see scratchpad/gist_dict_stage1_spec.md). Pure-logic pieces (cross-doc
pairing, seeded random ids, per-slot shard I/O) run fast, no model; the
canonical-encode geometry test and the end-to-end --smoke manifest test use a
real tiny model and are marked slow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marker.gist_dict import kv_slot_matrix

_GEO = {"n_layers": 2, "n_kv_heads": 2, "head_dim": 4}


def _fake_kv(k_slots, geo=_GEO, seed=0):
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


# ── wrong-doc pairing: reuses predprobe.pick_cross_doc_step, must map a LOCAL
# eval-step index back to a step from a DIFFERENT doc ──────────────────────


def test_pick_wrong_doc_indices_never_same_doc():
    from marker.run_gist_dict import pick_wrong_doc_indices

    gen = torch.Generator().manual_seed(0)
    wrong = pick_wrong_doc_indices(4, gen)
    assert len(wrong) == 4
    for i, w in enumerate(wrong):
        assert w != i  # one step per doc: index i IS doc i
        assert 0 <= w < 4

    # seeded: same generator seed -> same pairing (cross-config comparability)
    wrong2 = pick_wrong_doc_indices(4, torch.Generator().manual_seed(0))
    assert wrong == wrong2


# ── random_ids: seeded, in-range ────────────────────────────────────────────


def test_random_ids_seeded_reproducible_and_in_range():
    from marker.run_gist_dict import random_ids

    gen1 = torch.Generator().manual_seed(3)
    gen2 = torch.Generator().manual_seed(3)
    a = random_ids(K=17, k_slots=8, gen=gen1)
    b = random_ids(K=17, k_slots=8, gen=gen2)
    assert a == b
    assert len(a) == 8
    assert all(0 <= i < 17 for i in a)

    gen3 = torch.Generator().manual_seed(4)
    c = random_ids(K=17, k_slots=8, gen=gen3)
    assert c != a


# ── per-slot shard I/O: write once, read one slot back WITHOUT the others ───


def test_slot_shards_written_and_read_without_loading_all_slots(tmp_path):
    from marker.run_gist_dict import load_readouts, load_slot_shard, write_fit_shards

    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 3}
    d = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    dr = 5
    k_slots = 4
    n = 6

    kvs = [_fake_kv(k_slots, geo=geo, seed=i) for i in range(n)]
    readouts = [
        torch.randn(k_slots, dr, generator=torch.Generator().manual_seed(100 + i)) for i in range(n)
    ]
    n_written = write_fit_shards(zip(kvs, readouts, strict=True), tmp_path, d, dr, k_slots)
    assert n_written == n

    for s in range(k_slots):
        assert (tmp_path / f"slot_{s}.safetensors").exists()
    assert (tmp_path / "readouts.safetensors").exists()
    # no leftover raw scratch files
    assert not list(tmp_path.glob("*.raw"))

    want_slot0 = torch.stack([kv_slot_matrix(kv)[0] for kv in kvs]).half()
    # delete every OTHER slot's file -- load_slot_shard(0) must still work,
    # proving it never touches the other slots
    for s in range(1, k_slots):
        (tmp_path / f"slot_{s}.safetensors").unlink()
    got_slot0 = load_slot_shard(tmp_path, 0)
    assert torch.equal(got_slot0, want_slot0)

    ro = load_readouts(tmp_path)
    want_ro = torch.stack(readouts).half()
    assert torch.equal(ro, want_ro)


def test_write_fit_shards_drops_nothing_when_stream_is_short():
    from marker.run_gist_dict import write_fit_shards

    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 2}
    d = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    dr = 3
    k_slots = 2

    def _one():
        kv = _fake_kv(k_slots, geo=geo, seed=0)
        ro = torch.randn(k_slots, dr)
        yield kv, ro

    n = write_fit_shards(_one(), Path("/tmp/gist_dict_test_short_stream"), d, dr, k_slots)  # noqa: S108
    assert n == 1


# ── build_all_dicts resilience: per-config on_built, failure isolation ─────


def _write_tiny_shards(tmp_path, n=12, k_slots=2, dr=3):
    from marker.run_gist_dict import write_fit_shards

    geo = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 3}
    d = geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"]
    kvs = [_fake_kv(k_slots, geo=geo, seed=i) for i in range(n)]
    ros = [
        torch.randn(k_slots, dr, generator=torch.Generator().manual_seed(50 + i)) for i in range(n)
    ]
    write_fit_shards(zip(kvs, ros, strict=True), tmp_path, d, dr, k_slots)
    return geo, d, dr


def test_build_all_dicts_on_built_fires_per_config_and_failure_continues(tmp_path):
    from marker.run_gist_dict import build_all_dicts

    geo, _d, _dr = _write_tiny_shards(tmp_path, n=12, k_slots=2)
    geometry = {**geo, "k_slots": 2}
    calls, errors = [], {}
    # res_k1=50 > n=12 -> the kv_res config MUST fail (kmeans K>N assert)
    dicts, fit_ids = build_all_dicts(
        tmp_path,
        2,
        geometry,
        ks=[4],
        res_k1=50,
        res_k2=2,
        ro_k=3,
        whole_k=3,
        whole_proj_dim=3,
        seed=0,
        device="cpu",
        on_built=lambda name, d_: calls.append((name, d_["cfg"])),
        errors=errors,
    )
    # on_built fired the moment each config finished, in build order, and
    # the failing config neither fired it nor stopped the ones after it
    assert calls == [("kv_K4", "kv_K4"), ("ro_K3", "ro_K3"), ("whole_K3", "whole_K3")]
    assert set(dicts) == {"kv_K4", "ro_K3", "whole_K3"}
    assert set(fit_ids) == set(dicts)
    assert list(errors) == ["kv_res_50x2"]
    assert "AssertionError" in errors["kv_res_50x2"]


def test_per_config_saver_pushes_immediately_after_each_build(tmp_path, monkeypatch):
    import marker.run_gist_dict as rgd

    pushes = []
    monkeypatch.setattr(
        rgd, "_push_with_retry", lambda repo, folder, sub: pushes.append((repo, folder, sub))
    )
    on_built = rgd._per_config_saver(tmp_path / "dicts", "user/repo", smoke=False)
    dict_ = {"cfg": "kv_K4", "kind": "kv", "geometry": {}, "slots": []}
    on_built("kv_K4", dict_)
    assert (tmp_path / "dicts" / "dict_kv_K4.pt").exists()
    assert pushes == [("user/repo", str(tmp_path / "dicts"), "gist_dict")]
    # smoke / missing out-repo: still saved locally, never pushed
    on_smoke = rgd._per_config_saver(tmp_path / "d2", "user/repo", smoke=True)
    on_smoke("kv_K4", dict_)
    assert (tmp_path / "d2" / "dict_kv_K4.pt").exists()
    assert len(pushes) == 1


# ── fit-shard cache index: what --load-shards reads back ───────────────────


def test_fit_index_round_trips_doc_keys_as_tuples(tmp_path):
    from marker.run_gist_dict import load_fit_index, write_fit_index

    items = [
        (("gsm8k_train", 0), [1, 2, 3], "a step"),
        (("openr1", 4), [7], "x"),
    ]
    geometry = {"n_layers": 1, "n_kv_heads": 1, "head_dim": 3, "k_slots": 2}
    write_fit_index(tmp_path, items, geometry, d=6, dr=3, too_long=5)
    got_items, got_geo, got_d, got_dr, got_tl = load_fit_index(tmp_path)
    assert got_items == items  # doc keys back as TUPLES (hashable downstream)
    assert (got_geo, got_d, got_dr, got_tl) == (geometry, 6, 3, 5)


# ── canonical encode: keys land at [base, base+k_slots) -- tiny real model ──


def _tiny_qwen():
    from marker.run_stage2 import _load_stage1

    return _load_stage1("Qwen/Qwen2.5-0.5B", None, "cpu", False)


@pytest.mark.slow
def test_encode_canonical_places_keys_at_base_offset():
    from marker.gist_model import gist_kv
    from marker.run_gist_dict import encode_canonical

    pm, gist, tok = _tiny_qwen()
    pm.set_adapter("default")
    ids = tok("Three plus four is seven.", add_special_tokens=False).input_ids

    kv, readout, cont_start = encode_canonical(pm, gist, ids, base=64)
    k_slots = gist.shape[0]
    assert cont_start == 64 + k_slots
    assert readout.shape == (k_slots, gist.shape[-1])

    # the SAME canonical placement, called directly through gist_kv, must
    # give bitwise-identical keys -- encode_canonical is not a second dialect
    kv2, cs2, _ = gist_kv(pm, gist, ids, gist_start=64)
    assert cont_start == cs2
    for a, b in zip(kv.keys, kv2.keys, strict=True):
        assert torch.equal(a, b)


@pytest.mark.slow
def test_encode_canonical_rejects_steps_longer_than_max_span():
    from marker.run_gist_dict import encode_canonical

    pm, gist, tok = _tiny_qwen()
    pm.set_adapter("default")
    long_ids = list(range(1, 10))  # 9 tokens
    with pytest.raises(AssertionError):
        encode_canonical(pm, gist, long_ids, base=64, max_span=4)


# ── end-to-end smoke ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_smoke_manifest_runs_full_pipeline_and_has_a_verdict():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "marker.run_gist_dict", "--smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + "\n" + proc.stderr[-6000:]
    (line,) = (
        line_ for line_ in proc.stdout.splitlines() if line_.startswith("[GISTDICT MANIFEST]")
    )
    manifest = json.loads(line[len("[GISTDICT MANIFEST] ") :])

    assert manifest["gate0_pass"] in (True, False)
    assert set(manifest["configs"]) >= {
        "kv_K8",
        "kv_res_4x2",
        "ro_K8",
        "whole_K8",
    }
    for cfg, cell in manifest["configs"].items():
        for cond in ("native", "quantized", "wrong_doc_quantized", "random_ids"):
            assert cond in cell["conditions"], f"{cfg} missing {cond}"
        # full metrics for the generated conditions; random_ids is the
        # NLL-only floor (wall-clock cut), native is the shared once-scored
        # condition -- all still carry nll_mean and n
        for cond in ("native", "quantized", "wrong_doc_quantized"):
            for key in ("f1_mean", "rel_exact", "num_recall", "nll_mean", "n"):
                assert key in cell["conditions"][cond], f"{cfg}/{cond} missing {key}"
        assert cell["conditions"]["random_ids"].get("nll_only") is True
        for key in ("nll_mean", "n"):
            assert key in cell["conditions"]["random_ids"]
        assert "R_gsm8k" in cell and "R_fresh" in cell
        assert "usage_entropy" in cell and "dead_entries" in cell
        assert len(cell["usage_entropy_per_slot"]) == manifest["geometry"]["k_slots"]
        assert "op_from_ids" in cell
        assert "op_probe_readout" in cell  # may be None on tiny smoke splits
    assert manifest["verdict"] in {"GO", "RETRIEVAL", "VQVAE", "KILL", "INVALID_HARNESS"}
    assert manifest["n_fit"] > 0
    assert manifest["n_eval_gsm8k"] > 0
    assert manifest["n_eval_fresh"] > 0


# ── quantized lookups: kv_res must never materialize the joint table ────────


def test_quantized_rows_kv_res_matches_c1_plus_c2():
    from marker.run_gist_dict import _quantized_rows

    g = torch.Generator().manual_seed(0)
    c1 = torch.randn(3, 6, generator=g).half()
    c2 = torch.randn(2, 6, generator=g).half()
    dict_ = {"kind": "kv_res", "slots": [{"c1": c1, "c2": c2, "K2": 2}]}
    ids = torch.tensor([0, 1, 5, 4])  # joint ids: (0,0), (0,1), (2,1), (2,0)
    rows = _quantized_rows(dict_, 0, ids)
    want = torch.stack(
        [
            c1[0].float() + c2[0].float(),
            c1[0].float() + c2[1].float(),
            c1[2].float() + c2[1].float(),
            c1[2].float() + c2[0].float(),
        ]
    )
    assert torch.equal(rows, want)


def test_quantized_readouts_kv_res_sparse_lookup_zero_for_unseen_id():
    from marker.run_gist_dict import quantized_readouts

    mu_ids = torch.tensor([1, 4, 7])
    mu = torch.arange(9, dtype=torch.float16).reshape(3, 3)  # rows 0..2
    dict_ = {"kind": "kv_res", "slots": [{"mu_ids": mu_ids, "mu_readout": mu}]}
    ids = torch.tensor([[4], [1], [3], [9]])  # 3 and 9 (beyond max) never occupied
    out = quantized_readouts(dict_, ids)
    assert torch.equal(out[0, 0], mu[1].float())
    assert torch.equal(out[1, 0], mu[0].float())
    assert torch.equal(out[2, 0], torch.zeros(3))
    assert torch.equal(out[3, 0], torch.zeros(3))


def test_quantized_readouts_dense_kinds_index_mu():
    from marker.run_gist_dict import quantized_readouts

    mu = torch.arange(8, dtype=torch.float16).reshape(4, 2)
    dict_ = {"kind": "kv", "slots": [{"mu_readout": mu}, {"mu_readout": mu * 2}]}
    ids = torch.tensor([[0, 3], [2, 1]])
    out = quantized_readouts(dict_, ids)
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0, 0], mu[0].float())
    assert torch.equal(out[0, 1], (mu * 2)[3].float())
    assert torch.equal(out[1, 0], mu[2].float())


# ── per-slot usage diagnostics ──────────────────────────────────────────────


def test_usage_by_slot_is_per_slot_not_slot_zero():
    from marker.run_gist_dict import usage_by_slot

    # slot 0 uses only id 0 (degenerate); slot 1 uses ids 0..3 uniformly
    ids = torch.stack(
        [torch.zeros(8, dtype=torch.long), torch.arange(8, dtype=torch.long) % 4], dim=1
    )
    u = usage_by_slot(ids, K=4)
    assert len(u) == 2
    assert u[0]["entropy_norm"] == 0.0 and u[0]["dead_entries"] == 3
    assert u[1]["entropy_norm"] == 1.0 and u[1]["dead_entries"] == 0


# ── centroid-readout op probe (summaryprobe recipe on quantized readouts) ───


def test_centroid_readout_op_probe_learns_a_separable_mapping():
    from marker.run_gist_dict import centroid_readout_op_probe

    g = torch.Generator().manual_seed(0)
    ops = ["+", "-", "*", "/"]
    n_fit, n_eval, k_slots, dr = 160, 40, 2, 8
    means = torch.randn(4, dr, generator=g) * 3

    def _make(n, seed):
        gg = torch.Generator().manual_seed(seed)
        y = torch.randint(0, 4, (n,), generator=gg)
        q = means[y].unsqueeze(1).expand(n, k_slots, dr) + 0.05 * torch.randn(
            n, k_slots, dr, generator=gg
        )
        return q, [ops[int(c)] for c in y]

    q_fit, fit_ops = _make(n_fit, 1)
    q_eval, eval_ops = _make(n_eval, 2)
    fit_docs = list(range(n_fit))  # one doc per step
    out = centroid_readout_op_probe(q_fit, fit_ops, fit_docs, q_eval, eval_ops, seed=0, pca_dim=8)
    assert out is not None
    assert out["acc"] > 0.9, out
    assert "majority" in out


def test_centroid_readout_op_probe_returns_none_when_fit_too_small():
    from marker.run_gist_dict import centroid_readout_op_probe

    q = torch.randn(3, 2, 4)
    out = centroid_readout_op_probe(q, ["+", "-", "+"], [0, 1, 2], q, ["+", "-", "+"])
    assert out is None
