"""Tests for the PREDPROBE helpers (predprobe.py): the model-free logic behind
run_predprobe.py's five rendered conditions (true_gistkv / true_bridged /
wrong_bridged / pred_bridged / noised_bridged) and the gated verdict read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marker.predprobe import (
    CEILING,
    bridged_condition,
    gated_verdict,
    pick_cross_doc_step,
    relation_gate,
    scorable_ns,
    struct_nll_gate,
)

# ── scorable_ns: n=0 is unscorable for EVERY condition ──────────────────────


def test_scorable_ns_excludes_step_zero():
    assert scorable_ns(1) == []  # single-step doc: no history -> unscorable
    assert scorable_ns(2) == [1]
    assert scorable_ns(4) == [1, 2, 3]


def test_scorable_ns_never_includes_zero():
    for m in range(1, 10):
        assert 0 not in scorable_ns(m)


# ── bridged_condition: the cont_start=k and kv_dtype invariants (Fable's two
# named plumbing traps -- "true_bridged doubles as the runtime canary") ──────


def test_bridged_condition_cont_start_is_bridge_k_not_span_dependent():
    from marker.bridge import GistBridge

    bridge = GistBridge(d=8, k=3, n_layers=2, n_kv_heads=2, head_dim=4, width=16)
    # two DIFFERENT input vectors (standing in for summaries of different-length
    # spans) must still produce the SAME canonical cont_start -- never
    # len(ids)+k like gistprobe's Path A.
    for vec in (torch.randn(3, 8), torch.randn(3, 8) * 5):
        kv, cont_start = bridged_condition(bridge, vec, torch.float32)
        assert cont_start == bridge.k == 3


def test_bridged_condition_casts_to_kv_dtype():
    from marker.bridge import GistBridge

    bridge = GistBridge(d=8, k=3, n_layers=2, n_kv_heads=2, head_dim=4, width=16)
    vec = torch.randn(3, 8)
    kv, _ = bridged_condition(bridge, vec, torch.float16)
    assert all(k_.dtype == torch.float16 for k_ in kv.keys)
    assert all(v_.dtype == torch.float16 for v_ in kv.values)
    # bridge itself stays fp32 (its training precision) -- the cast is on the
    # OUTPUT only, not a dtype change to the bridge's own parameters
    assert all(p.dtype == torch.float32 for p in bridge.parameters())


# ── pick_cross_doc_step: wrong_bridged pairs across DIFFERENT docs, never
# shift-by-1 within a doc (Fable: adjacent GSM8K steps share numbers and
# inflate the floor) ─────────────────────────────────────────────────────────


def test_pick_cross_doc_step_never_picks_own_doc():
    gen = torch.Generator().manual_seed(0)
    doc_lengths = [3, 4, 5, 2]
    for di in range(len(doc_lengths)):
        for _ in range(25):
            dj, sj = pick_cross_doc_step(di, doc_lengths, gen)
            assert dj != di
            assert 0 <= sj < doc_lengths[dj]


def test_pick_cross_doc_step_falls_back_to_own_doc_when_only_one_exists():
    gen = torch.Generator().manual_seed(0)
    dj, sj = pick_cross_doc_step(0, [4], gen)
    assert dj == 0
    assert 0 <= sj < 4


def test_pick_cross_doc_step_single_doc_fallback_never_picks_the_scored_step():
    # degenerate single-doc set: the fallback must not hand wrong_bridged the
    # CURRENT step's own summary (that would silently make W == true_bridged
    # and erase the headroom H)
    gen = torch.Generator().manual_seed(0)
    for step in range(4):
        for _ in range(25):
            dj, sj = pick_cross_doc_step(0, [4], gen, step_idx=step)
            assert dj == 0
            assert sj != step


# ── noised(ratio=0) is a model-free identity no-op (Fable: the model-free
# invariant PREDPROBE's noised_bridged condition relies on) ─────────────────


def test_noised_ratio_zero_is_identity():
    from marker.run_bridge import noised

    x = torch.randn(3, 8)
    y = noised(x, 0.0, torch.Generator().manual_seed(0))
    assert torch.equal(x, y)


# ── relation_gate / struct_nll_gate: the gate-2/3/4 thresholds ──────────────


def test_relation_gate_green_at_70pct_headroom():
    assert relation_gate(w=0.2, h=0.4, p=0.2 + 0.7 * 0.4) == "GREEN"


def test_relation_gate_red_at_30pct_headroom():
    assert relation_gate(w=0.2, h=0.4, p=0.2 + 0.3 * 0.4) == "RED"


def test_relation_gate_yellow_between():
    assert relation_gate(w=0.2, h=0.4, p=0.2 + 0.5 * 0.4) == "YELLOW"


def test_struct_nll_gate_green_and_red_bands():
    assert struct_nll_gate(delta_b=1.0, delta_p=0.6) == "GREEN"
    assert struct_nll_gate(delta_b=1.0, delta_p=0.1) == "RED"
    assert struct_nll_gate(delta_b=1.0, delta_p=0.35) == "YELLOW"


def test_struct_nll_gate_no_headroom_is_yellow_not_divide_by_zero():
    # delta_b <= 0 means true_bridged has NO nll improvement over wrong at all
    # -- there's no fraction of "nothing" to judge, so this must not raise or
    # silently produce a nonsense ratio.
    assert struct_nll_gate(delta_b=0.0, delta_p=0.0) == "YELLOW"
    assert struct_nll_gate(delta_b=-0.3, delta_p=0.1) == "YELLOW"


# ── gated_verdict: the full read, in gate order ─────────────────────────────


def test_gate0_invalid_harness_when_c_below_absolute_floor():
    # a plumbing bug (wrong cont_start / dtype) craters Path A to near-floor
    v = gated_verdict(c=0.5, b=0.9, w=0.2, p=0.8, nll_w=1.0, nll_b=0.5, nll_p=0.6)
    assert v["gate"] == 0
    assert v["verdict"] == "INVALID_HARNESS_CHECK_PLUMBING"


def test_gate0_passes_near_published_ceiling():
    v = gated_verdict(c=CEILING - 0.03, b=0.9, w=0.2, p=0.8, nll_w=1.0, nll_b=0.5, nll_p=0.6)
    assert v["gate"] != 0


def test_gate0_not_invalid_when_recomputed_ceiling_legitimately_below_published():
    # the n>=1 recomputed ceiling (and the easy set's ceiling) can sit well
    # under the published 0.92 -- 0.85 is a healthy Path A number, not a
    # plumbing failure. Regression: a two-sided 0.92 +/- 0.05 anchor falsely
    # screamed INVALID here.
    v = gated_verdict(c=0.85, b=0.8, w=0.2, p=0.68, nll_w=1.0, nll_b=0.0, nll_p=0.4)
    assert v["gate"] != 0
    assert v["verdict"] != "INVALID_HARNESS_CHECK_PLUMBING"


def test_gate0_not_invalid_when_c_above_published_ceiling():
    # better-than-published is not a plumbing failure either
    v = gated_verdict(c=0.99, b=0.9, w=0.2, p=0.8, nll_w=1.0, nll_b=0.5, nll_p=0.6)
    assert v["gate"] != 0


def test_gated_verdict_none_input_is_insufficient_data_not_a_fake_gate0():
    # a set where no step has any extractable relation yields rel=None; that
    # must read INSUFFICIENT_DATA, not be coerced to 0.0 and fake a plumbing
    # INVALID (or, for W=None, a fake headroom H=B)
    v = gated_verdict(c=None, b=0.8, w=0.2, p=0.5, nll_w=1.0, nll_b=0.5, nll_p=0.6)
    assert v["verdict"] == "INSUFFICIENT_DATA"
    v = gated_verdict(c=0.9, b=0.8, w=0.2, p=0.5, nll_w=None, nll_b=0.5, nll_p=0.6)
    assert v["verdict"] == "INSUFFICIENT_DATA"


def test_gate1_bridge_is_wall_on_low_headroom():
    v = gated_verdict(c=CEILING, b=0.3, w=0.25, p=0.28, nll_w=1.0, nll_b=0.9, nll_p=0.9)
    assert v["gate"] == 1
    assert v["verdict"] == "BRIDGE_IS_WALL"


def test_gate1_bridge_is_wall_on_low_absolute_b():
    # H can be large in relative terms but B itself below the 0.5 absolute floor
    v = gated_verdict(c=CEILING, b=0.4, w=0.05, p=0.3, nll_w=1.0, nll_b=0.2, nll_p=0.5)
    assert v["gate"] == 1


def test_gate2_green_when_both_metrics_agree():
    # H = 0.6, P keeps 80% of headroom (>=0.7) -- relation GREEN
    # delta_b = 1.0, delta_p = 0.6 (>=0.5) -- nll GREEN
    v = gated_verdict(c=CEILING, b=0.8, w=0.2, p=0.68, nll_w=1.0, nll_b=0.0, nll_p=0.4)
    assert v["gate"] == 2
    assert v["verdict"] == "GREEN"


def test_gate3_red_when_both_metrics_agree():
    v = gated_verdict(c=CEILING, b=0.8, w=0.2, p=0.26, nll_w=1.0, nll_b=0.0, nll_p=0.85)
    assert v["gate"] == 3
    assert v["verdict"] == "RED"


def test_gate4_yellow_when_relation_ambiguous():
    v = gated_verdict(c=CEILING, b=0.8, w=0.2, p=0.5, nll_w=1.0, nll_b=0.0, nll_p=0.4)
    assert v["gate"] == 4
    assert v["verdict"] == "YELLOW"


def test_disagreement_between_relation_and_nll_forces_yellow():
    # relation says GREEN (P keeps 80% of headroom) but nll says RED (delta_p
    # keeps only 10% of delta_b) -- "both metrics must agree for green"
    v = gated_verdict(c=CEILING, b=0.8, w=0.2, p=0.68, nll_w=1.0, nll_b=0.0, nll_p=0.9)
    assert v["gate"] == 4
    assert v["verdict"] == "YELLOW"


@pytest.mark.parametrize("field", ["h", "gate", "verdict"])
def test_gated_verdict_always_has_core_fields(field):
    v = gated_verdict(c=CEILING, b=0.8, w=0.2, p=0.5, nll_w=1.0, nll_b=0.5, nll_p=0.6)
    assert field in v


# ── end-to-end smoke: the real `--smoke` CLI path (tiny cached HF model, real
# GSM8K-shaped synthetic traces) -- checks the manifest actually carries all
# five conditions plus per-step records, not just that the pure helpers above
# are individually correct. Mirrors test_frontload.py's smoke pattern. ───────


@pytest.mark.slow
def test_smoke_manifest_has_all_five_conditions_and_per_step_records():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "marker.run_predprobe", "--smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n" + proc.stderr[-4000:]
    (line,) = (
        line_ for line_ in proc.stdout.splitlines() if line_.startswith("[PREDPROBE MANIFEST]")
    )
    manifest = json.loads(line[len("[PREDPROBE MANIFEST] ") :])

    conds = {"true_gistkv", "true_bridged", "wrong_bridged", "pred_bridged", "noised_bridged"}
    assert manifest["per_step"], "no per-step records at all"
    seen_conds = {r["cond"] for r in manifest["per_step"]}
    assert conds <= seen_conds, f"missing conditions: {conds - seen_conds}"
    assert "pred_bridged_no_ledger" in seen_conds  # the non-gating secondary

    for label, results in manifest["results"].items():
        assert conds <= set(results.keys()), f"{label} results missing a condition"
        for cond in conds:
            assert results[cond]["n"] > 0, f"{label}/{cond} scored zero steps"
        assert label in manifest["gate"]
        assert "verdict" in manifest["gate"][label]

    # n=0 (step 0 of every doc) must never appear as a scored step
    assert all(r["n"] >= 1 for r in manifest["per_step"])
