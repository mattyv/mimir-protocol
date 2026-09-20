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


def test_pick_wrong_doc_step_never_same_doc():
    from marker.run_gist_dict import EvalStep, pick_wrong_doc_step

    pool = [
        EvalStep(doc_i=0, ids=[1, 2], text="a"),
        EvalStep(doc_i=1, ids=[3, 4], text="b"),
        EvalStep(doc_i=2, ids=[5, 6], text="c"),
        EvalStep(doc_i=3, ids=[7, 8], text="d"),
    ]
    gen = torch.Generator().manual_seed(0)
    wrong = pick_wrong_doc_step(pool, gen)
    assert len(wrong) == len(pool)
    for rec, w in zip(pool, wrong, strict=True):
        assert w.doc_i != rec.doc_i


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
            for key in ("f1_mean", "rel_exact", "num_recall", "nll_mean", "n"):
                assert key in cell["conditions"][cond], f"{cfg}/{cond} missing {key}"
        assert "R_gsm8k" in cell and "R_fresh" in cell
        assert "usage_entropy" in cell and "dead_entries" in cell
        assert "op_from_ids" in cell
    assert manifest["verdict"] in {"GO", "RETRIEVAL", "VQVAE", "KILL", "INVALID_HARNESS"}
    assert manifest["n_fit"] > 0
    assert manifest["n_eval_gsm8k"] > 0
    assert manifest["n_eval_fresh"] > 0
