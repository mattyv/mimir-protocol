# Saved weights inventory

All trained weights for the latent thought-prediction thread live in **one
private HF repo: `mattyvee/mimir-artifacts`**. The container is ephemeral —
nothing here is reconstructable from the working tree alone, so every trained
artifact is pushed to HF and this file is the index. (Single copy, no backup, by
decision — see the note at the bottom.)

Last verified: 2026-07-14 (loaded, shapes checked, all tensors finite).

## What's saved

| artifact | HF path | size | what it is |
|---|---|---|---|
| **Reasoning predictor** | `stage2_cot_openr1/predictor.pt` | 84 MB | the next-thought predictor everything downstream uses (d_model 384, d 3584, k 8, 4 layers). Trained on OpenR1 CoT. |
| Prose predictor | `stage2_predictor/predictor.pt` | 84 MB | the earlier web/prose next-thought predictor |
| **Bridge** | `bridge/bridge.pt` | 67 MB | predicted-thought → injectable KV (width 512, noise-trained, val-gated — the validated `bridge_pred 0.62` run) |
| **Gist encoder (LoRA)** | `checkpoints/step-000200 … 016000/` | 0.16 MB/step ×21 | the Stage-1 gist adapter (`adapter_model.safetensors` + `gist.safetensors`). Step **16000** is the one loaded by `_load_stage1`. |
| Render decoder (LoRA) | `render_adapter/render/adapter_model.safetensors` | 162 MB | thought → text reconstruction adapter |
| Render + ledger (LoRA) | `render_adapter_ledger/render/adapter_model.safetensors` | 162 MB | render adapter trained with the literals ledger (the validated F1 0.99 / numbers 100% run) |
| whiteners | `stage2_cot_openr1/whiteners.pt` | ~0 MB | identity stub (whitening off) — intentionally tiny, not a failed write |

Result manifests (no weights) also live in the repo: `*/manifest.json`,
`confidence_probe/`, `rollout/`.

## The rule (so nothing is ever local-only)

Every training harness pushes its weights to HF when `--out-repo` is passed —
the Vast launchers always pass it:

- `run_stage2.py`  → `stage2_predictor/` (predictor.pt + whiteners + manifest)
- `run_render.py`  → `render_adapter{,_ledger}/`
- `run_bridge.py`  → `bridge/` (bridge.pt + manifest)
- `run_confidence.py`, `run_rollout.py` → manifests only (eval-only, no weights)

Eval runs never overwrite weights. `bridge/bridge.pt` is overwritten by each new
bridge *training* run (kept: the latest/best), so re-run to a new `--out-repo`
subdir if a bridge must be preserved.

## Notes

- **Single copy, no backup.** If a second copy is wanted, mirror the repo (a
  second HF repo, or `huggingface-cli download mattyvee/mimir-artifacts` to
  durable local storage).
- **Token hygiene:** the HF write token is disposable and repo-scoped — revoke /
  rotate it once the campaign is done.
