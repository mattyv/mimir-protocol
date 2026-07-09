"""Model-free tests for the HF checkpoint push/resume logic (hf_push.py).

The network calls (upload_folder / snapshot_download) are thin wrappers not
unit-tested here; the RESUME logic — checkpoint naming, picking the latest
step from a repo listing, and the manifest round-trip — is pure and is where
a bug would silently orphan a run or resume from the wrong step.
"""

from __future__ import annotations

import json

from marker.hf_push import (
    checkpoint_dir,
    latest_step_from_listing,
    read_manifest,
    write_manifest,
)


def test_checkpoint_dir_zero_padded_and_parseable():
    assert checkpoint_dir(500) == "checkpoints/step-0000500"
    assert checkpoint_dir(4200) == "checkpoints/step-0004200"


def test_latest_step_picks_highest():
    listing = [
        "checkpoints/step-0000500/adapter_model.safetensors",
        "checkpoints/step-0001000/adapter_model.safetensors",
        "checkpoints/step-0000500/gist.safetensors",
        "README.md",
    ]
    assert latest_step_from_listing(listing) == 1000


def test_latest_step_none_when_no_checkpoints():
    assert latest_step_from_listing(["README.md", "config.json"]) is None


def test_latest_step_ignores_malformed_names():
    listing = [
        "checkpoints/step-0000500/x",
        "checkpoints/step-notanumber/y",
        "checkpoints/final/z",
    ]
    assert latest_step_from_listing(listing) == 500


def test_manifest_round_trip(tmp_path):
    meta = {"step": 1000, "tokens_seen": 5_000_000, "eval": {"ppl_gist": 12.3}}
    write_manifest(tmp_path, meta)
    assert (tmp_path / "manifest.json").exists()
    assert read_manifest(tmp_path) == meta


def test_read_manifest_missing_returns_none(tmp_path):
    assert read_manifest(tmp_path) is None


def test_manifest_is_valid_json(tmp_path):
    write_manifest(tmp_path, {"step": 42})
    raw = (tmp_path / "manifest.json").read_text()
    assert json.loads(raw)["step"] == 42
