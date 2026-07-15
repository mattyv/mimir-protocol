"""Tests for the front-loaded context test helpers (run_frontload.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from marker.run_frontload import (
    ARMS,
    _n_inject_for,
    _solve_arm,
    _with_read_adapter,
    answer_done,
    context_split,
)


def test_context_split_half_capped_and_bounded():
    assert context_split(3) == 2  # ceil(3/2)=2
    assert context_split(4) == 2
    assert context_split(6) == 3
    assert context_split(12) == 4  # capped
    assert context_split(12, cap=6) == 6


def test_context_split_leaves_a_step_for_the_model():
    # never consume the whole solution as context
    assert context_split(3) < 3
    with pytest.raises(ValueError, match="steps"):
        context_split(2)


def test_answer_done_requires_marker_digit_and_newline():
    assert not answer_done("still thinking about 5 things")
    assert not answer_done("#### ")  # marker but no digit
    assert not answer_done("#### 42")  # no newline yet — number may continue
    assert answer_done("blah\n#### 42\n")
    assert answer_done("#### 1,000\nnext")


# ── gist_read / none_read arms ────────────────────────────────────────────────
# gist_read must build the IDENTICAL injected KV cache as gist_true (same
# injection code path, same depth) and differ ONLY in which adapter is active
# during the final free-generate solve decode. none_read is the load-bearing
# control: same (empty) injection as `none`, same render-adapter solve as
# gist_read.


def test_gist_read_is_wired_into_arms():
    assert "gist_read" in ARMS


def test_none_read_is_wired_into_arms():
    assert "none_read" in ARMS


def test_gist_read_injects_same_depth_as_gist_true():
    # same n_inject, for every m -> same injection loop, same number of steps
    # pulled into the KV cache. This is the "same injection branch taken"
    # invariant the spec allows as a stand-in for intercepting the raw cache.
    for m in (2, 3, 4):
        assert _n_inject_for("gist_read", m) == _n_inject_for("gist_true", m) == m
    # the other arms are untouched by adding gist_read
    assert _n_inject_for("none", 4) == 0
    assert _n_inject_for("text", 4) == 0
    assert _n_inject_for("gist_render", 4) == 0
    assert _n_inject_for("gist_minus", 4) == 3
    assert _n_inject_for("gist_pred", 4) == 4


def test_none_read_injects_nothing_like_none():
    # none_read's prompt/injection must be identical to `none` -- zero gists --
    # so it isolates ONLY the render-adapter-active-during-solve variable.
    for m in (2, 3, 4):
        assert _n_inject_for("none_read", m) == _n_inject_for("none", m) == 0


class _FakeAdapterModel:
    """Stands in for the PEFT-wrapped model: records set_adapter calls without
    needing a real model, so the adapter-switch invariant is fast to check."""

    def __init__(self):
        self.calls = []
        self.active = "default"

    def set_adapter(self, name):
        self.calls.append(name)
        self.active = name


@pytest.mark.parametrize("read_arm", ["gist_read", "none_read"])
def test_solve_arm_reads_read_arms_through_render_adapter_and_restores_default(read_arm):
    pm = _FakeAdapterModel()
    seen_adapter_during_generate = {}

    def fake_free_generate(cache, pos, logits):
        seen_adapter_during_generate["adapter"] = pm.active
        return "text", 5

    text, ntok = _solve_arm(pm, read_arm, fake_free_generate, "CACHE", 3, "LOGITS")
    assert (text, ntok) == ("text", 5)
    assert seen_adapter_during_generate["adapter"] == "render"  # active DURING decode
    assert pm.calls == ["render", "default"]  # switch then restore, in order
    assert pm.active == "default"  # back to default after


def test_solve_arm_leaves_other_arms_on_default_untouched():
    for arm in ARMS:
        if arm.endswith("_read"):
            continue
        pm = _FakeAdapterModel()
        _solve_arm(pm, arm, lambda cache, pos, logits: ("x", 1), "CACHE", 3, "LOGITS")
        assert pm.calls == [], arm  # never touches the adapter for a non-`_read` arm


@pytest.mark.parametrize("read_arm", ["gist_read", "none_read"])
def test_solve_arm_restores_default_even_if_generate_raises(read_arm):
    pm = _FakeAdapterModel()

    def raising_free_generate(cache, pos, logits):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _solve_arm(pm, read_arm, raising_free_generate, "CACHE", 3, "LOGITS")
    assert pm.calls == ["render", "default"]  # finally-restore still fired
    assert pm.active == "default"  # not left stuck on "render"


# ── Fix 2: the render adapter must be active for the FIRST generated token ────
# _with_read_adapter is the shared helper _run uses to compute the logits that
# SELECT token 0 (the priming forward, either the post-injection newline
# read-through for gist_read, or the redone prefill for none_read). Pinning
# it here is the mechanical stand-in for "adapter is render at the moment
# token-0 logits are produced" without needing a real model.


@pytest.mark.parametrize("read_arm", ["gist_read", "none_read"])
def test_with_read_adapter_runs_fn_under_render_for_read_arms(read_arm):
    pm = _FakeAdapterModel()
    seen = {}

    def fn():
        seen["adapter"] = pm.active  # the adapter active WHILE fn (the logits
        # forward) runs -- this is the token-0 timing invariant
        return "logits-for-token-0"

    result = _with_read_adapter(pm, read_arm, fn)
    assert result == "logits-for-token-0"
    assert seen["adapter"] == "render"
    assert pm.calls == ["render", "default"]  # switch then restore
    assert pm.active == "default"  # restored after


def _record_active_adapter(pm, seen):  # noqa: ANN001
    """Test helper: returns a no-arg fn that records pm's active adapter into
    `seen` when called. Pulled out of the loop below so ruff (B023) doesn't
    flag a closure over a loop variable -- fn is called immediately, within
    the same iteration, so the closure is safe, but a named helper is clearer
    either way."""

    def fn():
        seen["adapter"] = pm.active
        return "logits"

    return fn


def test_with_read_adapter_runs_fn_under_default_for_non_read_arms():
    for arm in ARMS:
        if arm.endswith("_read"):
            continue
        pm = _FakeAdapterModel()
        seen = {}
        assert _with_read_adapter(pm, arm, _record_active_adapter(pm, seen)) == "logits"
        assert seen["adapter"] == "default"
        assert pm.calls == [], arm  # adapter never touched for non-`_read` arms


def test_with_read_adapter_restores_default_even_if_fn_raises():
    pm = _FakeAdapterModel()

    def raising():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _with_read_adapter(pm, "gist_read", raising)
    assert pm.calls == ["render", "default"]
    assert pm.active == "default"


@pytest.mark.slow
def test_smoke_manifest_covers_every_arm_and_logs_every_problem():
    # runs the real `--smoke` CLI path (tiny cached HF model, real GSM8K
    # few-shot prompt) end to end and checks the manifest for two things:
    # gist_read actually ran, and per_problem has one record for EVERY scored
    # (problem, arm) pair, not just the dumped samples.
    import json

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "marker.run_frontload", "--smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={
            **os.environ,
            "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + "\n" + proc.stderr[-4000:]
    (line,) = (
        line_ for line_ in proc.stdout.splitlines() if line_.startswith("[FRONTLOAD MANIFEST]")
    )
    manifest = json.loads(line[len("[FRONTLOAD MANIFEST] ") :])
    assert set(manifest["arms"]) == set(ARMS)
    n_scored = manifest["arms"]["gist_read"]["n"]
    assert n_scored > 0
    assert len(manifest["per_problem"]) == n_scored * len(ARMS)
    assert {r["arm"] for r in manifest["per_problem"]} == set(ARMS)
    # Fix 3: full gen text is logged ONLY for the `_read` arms
    for r in manifest["per_problem"]:
        if r["arm"].endswith("_read"):
            assert "gen" in r and isinstance(r["gen"], str) and r["gen"]
        else:
            assert "gen" not in r
