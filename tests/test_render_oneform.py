"""Tests for the bilingual-reader render training path (run_render.py's new
--kv-source/--warm-start-subdir/--out-subdir machinery): teach the render LoRA
a second dialect (bridge_validated's converter KV) WITHOUT losing the first
(encoder gist_kv).

bridge_validated is lossless for next-step likelihood; reconstruction fidelity
is what this run tests. Mechanical invariants only -- experiment numbers live
in the manifest, not in assertions here.

Tests that need a real tokenizer against real text (build_records, eval_dialects)
use the cached Qwen2.5-0.5B, exactly where test_predprobe.py's own smoke test
does -- a tiny RANDOM model's vocab (64) is too small for a real tokenizer's
ids and IndexErrors on the embedding lookup. The warm-start test only needs
gist_kv/render_nll on hand-picked span ids (no tokenizer), so it stays on
tests/test_gist_model.py's _tiny_base, matching test_render.py's own pattern.
Pure-logic tests (sampler, clobber guard, baseline thresholds, cross-doc
pairing) need no model at all and stay fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marker.render import attach_render, render_nll


@pytest.fixture(scope="module")
def qwen():
    """Real Qwen2.5-0.5B + Stage-1 gist adapter (cached locally, no network) --
    shared across this module's tests since loading it is the expensive part;
    the tests themselves are read-only forward passes (no training)."""
    from marker.run_stage2 import _load_stage1

    return _load_stage1("Qwen/Qwen2.5-0.5B", None, "cpu", False)


def _tiny_bridge(pm, gist, tok, device="cpu"):
    from marker.bridge import GistBridge
    from marker.gist_model import gist_kv

    probe_kv, _, _ = gist_kv(pm, gist, tok("hi", add_special_tokens=False).input_ids)
    kv_dtype = probe_kv.keys[0].dtype
    bridge = (
        GistBridge(
            d=gist.shape[-1],
            k=gist.shape[0],
            n_layers=probe_kv.n_layers,
            n_kv_heads=probe_kv.keys[0].shape[1],
            head_dim=probe_kv.keys[0].shape[3],
            width=16,
        )
        .to(device)
        .eval()
    )
    return bridge, kv_dtype


# ── 1. build_records(bridge=None): bitwise-equal to a direct gist_kv call ───


@pytest.mark.slow
def test_build_records_gist_only_matches_direct_gist_kv(qwen):
    from marker.gist_model import gist_kv
    from marker.run_render import build_records

    pm, gist, tok = qwen
    pm.set_adapter("default")
    text = "Step one. Step two costs 3 dollars."

    records, doc_starts = build_records(pm, gist, tok, [text], "sentence", 32)
    assert doc_starts == [0]
    assert records and all(r.kv_bridge is None and r.cs_bridge is None for r in records)

    for rec in records:
        kv_direct, cs_direct, _ = gist_kv(pm, gist, rec.ids)
        assert cs_direct == rec.cs_gist
        for a, b in zip(kv_direct.keys, rec.kv_gist.keys, strict=True):
            assert torch.equal(a.cpu(), b)
        for a, b in zip(kv_direct.values, rec.kv_gist.values, strict=True):
            assert torch.equal(a.cpu(), b)


# ── 2. build_records(bridge=<random GistBridge>): cs_bridge==bridge.k, dtype,
# ids/text shared with the gist entry of the same record ────────────────────


@pytest.mark.slow
def test_build_records_bridge_path_cont_start_dtype_and_shared_ids(qwen):
    from marker.run_render import build_records

    pm, gist, tok = qwen
    bridge, kv_dtype = _tiny_bridge(pm, gist, tok)
    text = "Step one. Step two costs 3 dollars."

    records, _ = build_records(
        pm, gist, tok, [text], "sentence", 32, bridge=bridge, kv_dtype=kv_dtype
    )
    assert records
    for rec in records:
        assert rec.cs_bridge == bridge.k
        assert all(k_.dtype == kv_dtype for k_ in rec.kv_bridge.keys)
        assert all(v_.dtype == kv_dtype for v_ in rec.kv_bridge.values)
        # ids/text are the SAME object-level values feeding both dialects
        assert isinstance(rec.ids, list) and rec.text


# ── 3. mixed sampler: exactly two (record, dialect) items per record, with
# identical targets (same ids/text regardless of dialect) ───────────────────


def test_training_items_mixed_yields_both_dialects_per_record_with_same_target():
    from marker.run_render import Record, training_items

    records = [
        Record(doc_i=0, kv_gist="g0", cs_gist=1, kv_bridge="b0", cs_bridge=2, ids=[1, 2], text="a"),
        Record(doc_i=1, kv_gist="g1", cs_gist=1, kv_bridge="b1", cs_bridge=2, ids=[3, 4], text="b"),
    ]
    items = training_items(records, "mixed")
    assert len(items) == 2 * len(records)
    for rec in records:
        dialects = {d for r, d in items if r is rec}
        assert dialects == {"gist", "bridge"}
        # target is derived from ids/text alone -- identical across dialects
        targets = {tuple(r.ids) for r, d in items if r is rec}
        assert targets == {tuple(rec.ids)}


def test_training_items_gist_only_is_one_to_one():
    from marker.run_render import Record, training_items

    records = [
        Record(doc_i=0, kv_gist="g0", cs_gist=1, kv_bridge=None, cs_bridge=None, ids=[1], text="a")
    ]
    items = training_items(records, "gist")
    assert items == [(records[0], "gist")]


# ── 4. eval_dialects: four dialect keys, n<=cap; wrong_bridged pairing uses a
# DIFFERENT doc_i (pure logic, no model) + full eval_dialects smoke (model) ──


def test_pick_wrong_bridged_never_same_doc():
    from marker.run_render import Record, _pick_wrong_bridged

    pool = [
        Record(doc_i=0, kv_gist=None, cs_gist=0, kv_bridge="b00", cs_bridge=0, ids=[1], text="a"),
        Record(doc_i=0, kv_gist=None, cs_gist=0, kv_bridge="b01", cs_bridge=0, ids=[2], text="b"),
        Record(doc_i=1, kv_gist=None, cs_gist=0, kv_bridge="b10", cs_bridge=0, ids=[3], text="c"),
        Record(doc_i=2, kv_gist=None, cs_gist=0, kv_bridge="b20", cs_bridge=0, ids=[4], text="d"),
    ]
    gen = torch.Generator().manual_seed(0)
    wrong = _pick_wrong_bridged(pool, gen)
    assert len(wrong) == len(pool)
    for rec, w in zip(pool, wrong, strict=True):
        assert w.doc_i != rec.doc_i


@pytest.mark.slow
def test_eval_dialects_returns_four_dialect_keys_with_capped_n(qwen):
    from marker.run_render import build_records, eval_dialects

    pm, gist, tok = qwen
    bridge, kv_dtype = _tiny_bridge(pm, gist, tok)
    texts = [
        "Step one. Step two costs 3 dollars. Step three has 4 apples.",
        "Different doc line one. Different doc line two has 5 pears.",
    ]
    records, _ = build_records(
        pm, gist, tok, texts, "sentence", 32, bridge=bridge, kv_dtype=kv_dtype
    )
    assert len({r.doc_i for r in records}) >= 2  # need >=2 docs for wrong_bridged

    attach_render(pm, r=4)
    pm.set_adapter("render")

    gen = torch.Generator().manual_seed(0)
    cap = 3
    result = eval_dialects(pm, tok, records, cap, True, True, gen)
    for key in ("gist", "bridge", "wrong_bridged", "bridge_no_ledger"):
        assert key in result, f"missing dialect key {key!r}"
        assert result[key]["n"] <= cap
    assert "margin" in result


# ── 4b. relation metric: the key eval_dialects reads from relation_score must
# score a real arithmetic step against itself as 1.0 -- if the key (or the
# aggregation) were wrong, every real eval would read 0.0 and the baseline
# gate would fire BASELINE_MIRAGE on a healthy reader. Fast: _render_reconstruct
# is stubbed to decode each KV back to its own record's ids, so the full
# _score_record -> relation_score -> _summarize_scores path runs without a
# model. ─────────────────────────────────────────────────────────────────────


class _StubTok:
    """Two-'token' vocabulary: token 0 decodes to doc-0's text, token 1 to
    doc-1's. Enough for _score_record's decode + eval_dialects' stop-token
    lookup."""

    texts = {(0,): "So 5 * 3 = 15 dollars.", (1,): "Then 4 + 4 = 8 apples."}

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ARG002
        class R:
            input_ids = [9]

        return R()

    def decode(self, ids):  # noqa: ANN001
        return self.texts[tuple(ids)]


def test_relation_metric_scores_arithmetic_step_through_eval_dialects(monkeypatch):
    import marker.run_render as rr

    recon = {"kvG0": [0], "kvB0": [0], "kvG1": [1], "kvB1": [1]}
    monkeypatch.setattr(
        rr,
        "_render_reconstruct",
        lambda pm, kv, cs, first, max_new, stop, prefix_ids=None: recon[kv],
    )
    records = [
        rr.Record(0, "kvG0", 1, "kvB0", 2, [0], "So 5 * 3 = 15 dollars."),
        rr.Record(1, "kvG1", 1, "kvB1", 2, [1], "Then 4 + 4 = 8 apples."),
    ]
    out = rr.eval_dialects(
        None, _StubTok(), records, 2, True, False, torch.Generator().manual_seed(0)
    )
    # a perfect reconstruction of a real arithmetic step scores exactly 1.0
    assert out["gist"]["rel_exact"] == 1.0
    assert out["bridge"]["rel_exact"] == 1.0
    # wrong_bridged decodes the OTHER doc's step but is scored against THIS
    # record's text (mismatched relation) -> strictly below 1
    assert out["wrong_bridged"]["rel_exact"] < 1.0
    assert out["margin"] == 1.0
    assert "bridge_no_ledger" not in out  # ledger=False


def test_relation_metric_key_name_matches_gistprobe():
    from marker.gistprobe import relation_score

    assert relation_score("5 * 3 = 15", "5 * 3 = 15")["exact"] == 1.0
    assert relation_score("5 + 3 = 8", "5 * 3 = 15")["exact"] < 1.0


# ── 5. clobber guard ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["render_adapter", "render_adapter_ledger", "bridge_validated"])
def test_clobber_guard_raises_for_forbidden_out_subdir_on_warm_start(bad):
    from marker.run_render import check_out_subdir

    with pytest.raises(AssertionError):
        check_out_subdir(bad, "render_adapter_ledger", "bridge_validated", "mixed")


def test_clobber_guard_raises_when_out_subdir_equals_warm_start_subdir():
    from marker.run_render import check_out_subdir

    with pytest.raises(AssertionError):
        check_out_subdir("my_warm_source", "my_warm_source", "bridge_validated", "gist")


def test_clobber_guard_allows_new_subdir():
    from marker.run_render import check_out_subdir

    check_out_subdir("render_adapter_oneform", "render_adapter_ledger", "bridge_validated", "mixed")


def test_main_wiring_clobber_guard_fires_for_warm_start_with_default_out_subdir(monkeypatch):
    # the real launcher's dangerous case: WARM set, OUTSUB empty, --out-repo
    # always passed -- main() must refuse BEFORE loading any model (out_subdir
    # defaults to render_adapter_ledger, which the warm-start run would clobber)
    import marker.run_render as rr

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_render",
            "--smoke",
            "--ledger",
            "--kv-source",
            "mixed",
            "--warm-start-subdir",
            "render_adapter_ledger",
            "--out-repo",
            "dummy/repo",
        ],
    )
    with pytest.raises(AssertionError, match="clobber"):
        rr.main()


def test_clobber_guard_exempts_plain_default_path_no_warm_start_no_mixed():
    # the vanilla (no warm-start, kv-source=gist) path legitimately still
    # writes render_adapter[_ledger] -- today's behaviour, bitwise -- so the
    # guard must not fire when the default computed out_subdir equals it.
    from marker.run_render import check_out_subdir

    check_out_subdir("render_adapter_ledger", None, "bridge_validated", "gist")
    check_out_subdir("render_adapter", None, "bridge_validated", "gist")


# ── 6. split-before-double: no eval record's ids leak into the train set ────


def test_split_before_double_no_eval_ids_in_train_set_either_dialect():
    from marker.run_render import Record, training_items

    all_records = [
        Record(
            doc_i=i,
            kv_gist=f"g{i}",
            cs_gist=1,
            kv_bridge=f"b{i}",
            cs_bridge=2,
            ids=[i],
            text=str(i),
        )
        for i in range(10)
    ]
    n_eval = 3
    eval_records, train_records = all_records[:n_eval], all_records[n_eval:]
    eval_ids = {tuple(r.ids) for r in eval_records}

    items = training_items(train_records, "mixed")
    assert not any(tuple(rec.ids) in eval_ids for rec, _dialect in items)


# ── 7. warm-start: never calls attach_render; loaded weights are trainable ──


@pytest.mark.slow
def test_warm_start_loads_saved_adapter_and_gradient_reaches_it(monkeypatch, tmp_path):
    import marker.run_render as rr
    from marker.gist_model import attach_gist, gist_kv
    from tests.test_gist_model import _tiny_base

    base = _tiny_base()
    pm, gist = attach_gist(base, gist_k=4, r=4)
    attach_render(pm, r=4)
    pm.set_adapter("render")
    # stamp a recognizable constant into lora_A so the reload below provably
    # LOADED the saved weights (not a fresh random re-init under the same name)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "render" in n and "lora_A" in n:
                p.fill_(0.123)
    sub = "warm_sub"
    pm.save_pretrained(str(tmp_path / sub), selected_adapters=["render"])
    assert (tmp_path / sub / "render" / "adapter_config.json").exists()

    base2 = _tiny_base()
    pm2, gist2 = attach_gist(base2, gist_k=4, r=4)

    def _boom(*_a, **_k):
        raise AssertionError("warm-start path must never call attach_render")

    monkeypatch.setattr(rr, "attach_render", _boom)

    render_params = rr.warm_start_render(pm2, str(tmp_path), sub)
    assert render_params  # nonempty
    loaded_a = [p for n, p in render_params if "lora_A" in n]
    assert loaded_a and all(torch.allclose(p, torch.full_like(p, 0.123)) for p in loaded_a), (
        "warm start did not load the SAVED weights"
    )
    pm2.set_adapter("default")
    span = [1, 2, 3, 4]
    kv, cont_start, _ = gist_kv(pm2, gist2, span)
    pm2.set_adapter("render")
    loss = render_nll(pm2, kv, cont_start, span)
    loss.backward()
    grads = [p.grad.abs().sum() for _, p in render_params if p.grad is not None]
    assert grads and any(g > 0 for g in grads), "no gradient reached the warm-started render LoRA"


# ── 8. baseline_ok: boundary cases ──────────────────────────────────────────


def test_baseline_ok_boundaries():
    from marker.run_render import baseline_ok

    assert baseline_ok(gist_rel=0.80, bridge_rel=0.40, wrong_rel=0.30) is True
    assert baseline_ok(gist_rel=0.79, bridge_rel=0.40, wrong_rel=0.30) is False
    assert baseline_ok(gist_rel=0.80, bridge_rel=0.41, wrong_rel=0.30) is False
    assert baseline_ok(gist_rel=0.80, bridge_rel=0.40, wrong_rel=0.31) is False
    assert baseline_ok(gist_rel=0.95, bridge_rel=0.10, wrong_rel=0.05) is True


# ── 9. --smoke --kv-source mixed CLI end-to-end ─────────────────────────────


@pytest.mark.slow
def test_smoke_mixed_kv_source_end_to_end_reports_per_dialect_and_margin():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "marker.run_render", "--smoke", "--kv-source", "mixed"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n" + proc.stderr[-4000:]
    (line,) = (line_ for line_ in proc.stdout.splitlines() if line_.startswith("[RENDER MANIFEST]"))
    manifest = json.loads(line[len("[RENDER MANIFEST] ") :])
    assert manifest["kv_source"] == "mixed"
    for timing in ("step0", "final"):
        assert timing in manifest
        blk = manifest[timing]["gsm8k"]
        for dialect in ("gist", "bridge", "wrong_bridged"):
            assert dialect in blk
        assert "margin" in blk
