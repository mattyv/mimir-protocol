"""Checkpoint push / resume to a private HF repo (see GIST_PILOT_PLAN.md infra).

A Stage-1 training run outlives both the Vast node (which dies) and this
orchestrating session (reclaimed on idle), so weights must PUSH to a durable
store mid-run, not be pulled at the end. This module wraps that:

  - save_bundle()  writes adapter + gist embeddings + manifest to a step dir
  - push_bundle()  uploads a step dir to the repo (upload_folder)
  - resume_step()  finds the latest checkpoint already in the repo, if any
  - fetch_step()   downloads a checkpoint to resume from

The pure logic (naming, latest-step selection, manifest) is unit-tested; the
huggingface_hub calls are lazy-imported thin wrappers so the tests need no
token or network.

Token hygiene (Vast hosts can read the container): the write token is passed
via onstart env (HF_TOKEN), fine-grained + scoped to the one private repo +
write-only, and REVOKED after the campaign. Never committed, echoed, or logged
— this module reads it from the environment, never takes it as a printable arg.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_STEP_RE = re.compile(r"checkpoints/step-(\d+)/")


def checkpoint_dir(step: int) -> str:
    """Repo-relative dir for a step's checkpoint, zero-padded so lexical and
    numeric order agree."""
    return f"checkpoints/step-{step:07d}"


def latest_step_from_listing(paths: list[str]) -> int | None:
    """Highest step number among checkpoint paths in a repo listing, or None
    if there are no well-formed checkpoints (malformed names ignored)."""
    steps = [int(m.group(1)) for p in paths if (m := _STEP_RE.search(p))]
    return max(steps) if steps else None


def write_manifest(dir_path: Path, meta: dict) -> None:
    """Write manifest.json (step, tokens_seen, eval metrics) into a dir."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    (Path(dir_path) / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_manifest(dir_path: Path) -> dict | None:
    """Read manifest.json from a dir, or None if absent."""
    p = Path(dir_path) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── Artifact bundling (needs peft/torch — lazy, not import-time) ─────────────────


def save_bundle(dir_path: Path, lora_model, gist_embeddings, meta: dict) -> None:  # noqa: ANN001
    """Write the trainable artifacts for one checkpoint: the LoRA adapter
    (peft), the gist embedding tensor, and the manifest. Base weights are
    never saved — they're a public pointer."""
    from safetensors.torch import save_file  # noqa: PLC0415

    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(d))  # adapter_model.safetensors + config
    save_file({"gist": gist_embeddings.detach().cpu().contiguous()}, str(d / "gist.safetensors"))
    write_manifest(d, meta)


# ── HF wrappers (lazy import; token from env only) ───────────────────────────────


def _token() -> str | None:
    return os.environ.get("HF_TOKEN")


def push_bundle(dir_path: Path, repo_id: str, step: int) -> None:  # noqa: ANN001
    """Upload a checkpoint dir into the repo under its step path."""
    from huggingface_hub import upload_folder  # noqa: PLC0415

    upload_folder(
        repo_id=repo_id,
        folder_path=str(dir_path),
        path_in_repo=checkpoint_dir(step),
        token=_token(),
        commit_message=f"checkpoint step {step}",
    )


def resume_step(repo_id: str) -> int | None:
    """Latest checkpoint step already in the repo, or None (fresh run). Returns
    None on any repo error (treat as fresh — a missing repo is not fatal)."""
    from huggingface_hub import list_repo_files  # noqa: PLC0415
    from huggingface_hub.utils import HfHubHTTPError  # noqa: PLC0415

    try:
        files = list_repo_files(repo_id=repo_id, token=_token())
    except (HfHubHTTPError, OSError):
        return None
    return latest_step_from_listing(files)


def fetch_step(repo_id: str, step: int, dest: Path) -> Path:  # noqa: ANN001
    """Download a checkpoint dir to `dest` for resuming."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    local = snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{checkpoint_dir(step)}/*",
        local_dir=str(dest),
        token=_token(),
    )
    return Path(local) / checkpoint_dir(step)
