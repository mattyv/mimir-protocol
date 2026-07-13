"""Fast-lane confidence probe (FASTLANE_PLAN gate) — the cheap, decisive test.

Reuses the ALREADY-TRAINED predictor (no new training). On held-out reasoning
chains, for every (thought -> next-thought) prediction it records whether the
predictor's top-1 pick is the true next thought (within-document pool, the
topic-shortcut control) and three inference-time confidence signals, then asks:
does any signal SEPARATE the correct predictions from the wrong ones?

  PASS  — a high-confidence bin is reliably more correct (AUC comfortably > 0.5,
          high tercile >> low tercile) => adaptive skipping is viable => build
          the bridge + gated rollout.
  FAIL  — confidence doesn't track correctness => confidence-gating is dead;
          the render lane stays the deliverable.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_confidence \\
        --repo mattyvee/mimir-artifacts \\
        --artifacts-repo mattyvee/mimir-artifacts --subdir stage2_cot_openr1 \\
        --dataset openai/gsm8k --n-docs 300 --window 6
Smoke (local tiny model, random predictor — mechanics only):
    PYTHONPATH=src python -m marker.run_confidence --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.confidence import (
    dropout_agreement,
    mc_dropout_pool,
    prediction_norm,
    retrieval_margin,
    tercile_report,
    within_doc_correct,
)
from marker.predictor import NextThoughtPredictor


def _predictor_from_state(state: dict, heads: int) -> NextThoughtPredictor:
    """Rebuild the predictor with dims INFERRED from its checkpoint (so the
    probe can't silently load a mismatched architecture): slot_proj [d_model, d],
    pool_proj [d_model, k*d], trunk.layers.N => layer count."""
    d_model, d = state["slot_proj.weight"].shape
    k = state["pool_proj.weight"].shape[1] // d
    layers = 1 + max(int(key.split(".")[2]) for key in state if key.startswith("trunk.layers."))
    model = NextThoughtPredictor(d=d, k=k, d_model=d_model, layers=layers, heads=heads)
    model.load_state_dict(state)
    print(f"predictor: d={d} k={k} d_model={d_model} layers={layers} heads={heads}", flush=True)
    return model


@torch.no_grad()
def collect(model, seqs, length, device, n_dropout):  # noqa: ANN001
    """Run every length-L window through the predictor and gather, per
    (position i -> i+1) pair: pooled prediction, pooled target, document id, and
    the confidence signals. Correctness/margin/norm use the eval-mode prediction;
    dropout_agreement uses n_dropout stochastic passes. Returns a dict of aligned
    [N] / [N, d_model] tensors on CPU."""
    from marker.run_stage2 import _windows  # noqa: PLC0415

    wins, win_doc = [], []
    for di, s in enumerate(seqs):
        for w in _windows(s, length):
            wins.append(w)
            win_doc.append(di)
    if not wins:
        raise RuntimeError("no windows: need docs with >= window+1 steps")

    was_training = model.training
    model.eval()
    preds, tgts, agree = [], [], []
    dm = model.pool_proj.out_features
    for i in range(0, len(wins), 64):
        stack = torch.stack(wins[i : i + 64]).to(device)  # [B, L, k, d]
        pred = model(stack)  # [B, L-1, k, d]
        preds.append(model.pool(pred).reshape(-1, dm).cpu())
        tgts.append(model.pool(stack[:, 1:]).reshape(-1, dm).cpu())
        samples = mc_dropout_pool(model, stack, n_samples=n_dropout)  # [S, B*(L-1), dm]
        agree.append(dropout_agreement(samples).cpu())
    model.train(was_training)

    p, t = torch.cat(preds), torch.cat(tgts)
    doc = torch.tensor(win_doc).repeat_interleave(length - 1)
    return {
        "pred": p,
        "tgt": t,
        "doc": doc,
        "correct": within_doc_correct(p, t, doc),
        "norm": prediction_norm(p),
        "margin": retrieval_margin(p, t),  # bank = the eval target pool
        "agree": torch.cat(agree),
    }


def _report(data: dict) -> dict:
    """Tercile/AUC separation for each confidence signal against correctness."""
    correct = data["correct"]
    signals = {
        "dropout_agreement": data["agree"],
        "retrieval_margin": data["margin"],
        "prediction_norm": data["norm"],
    }
    out = {"n_pairs": int(len(correct)), "base_correct": round(float(correct.float().mean()), 3)}
    for name, sig in signals.items():
        out[name] = tercile_report(sig, correct)
    # verdict heuristic: any signal with AUC >= 0.60 AND high tercile clearly
    # above base is a viable gate (pre-registered threshold, tune on the result)
    aucs = {n: out[n]["auc"] for n in signals if out[n]["auc"] is not None}
    best = max(aucs, key=aucs.get) if aucs else None
    out["best_signal"] = best
    out["best_auc"] = aucs.get(best) if best else None
    out["verdict"] = "PASS" if best and aucs[best] >= 0.60 else "FAIL/weak"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None, help="Stage-1 gist adapter repo")
    ap.add_argument("--artifacts-repo", default=None, help="HF repo with predictor.pt")
    ap.add_argument("--subdir", default="stage2_cot_openr1")
    ap.add_argument("--out-repo", default=None, help="push the probe manifest here")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--corpus", default="cot")
    ap.add_argument("--unit", choices=["line", "sentence"], default=None)
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-dropout", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    unit = args.unit or ("line" if args.corpus == "cot" else "sentence")
    max_span, max_sents = (96, 16) if args.corpus == "cot" else (64, 24)
    min_sents = args.window + 1
    if args.smoke:
        args.model_name, args.repo, args.n_docs = "Qwen/Qwen2.5-0.5B", None, 40
        args.window, max_span, max_sents, min_sents = 3, 24, 12, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _preflight(device, args.window)  # crash device/shape bugs in ~$0.01, not post-encode

    from marker.run_stage2 import _doc_texts, _load_stage1, _smoke_cot_texts, encode_corpus

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )

    if args.smoke:
        # no checkpoint: a randomly-initialised predictor exercises the full
        # path (correctness ~= chance; the probe MECHANICS are what smoke tests)
        model = NextThoughtPredictor(d=gist.shape[-1], k=8, d_model=48, layers=2, heads=4)
        docs = _smoke_cot_texts(args.n_docs)
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(repo_id=args.artifacts_repo, filename=f"{args.subdir}/predictor.pt")
        model = _predictor_from_state(torch.load(path, map_location="cpu"), args.heads)
        docs = list(
            _doc_texts(args.n_docs, args.corpus, args.dataset, args.dataset_config, args.text_field)
        )
    model = model.to(device)

    seqs = encode_corpus(pm, gist, tok, docs, max_span, max_sents, min_sents, device, unit)
    print(f"encoded {len(seqs)} held-out chains", flush=True)
    data = collect(model, seqs, args.window, device, args.n_dropout)
    manifest = _report(data)
    print(
        f"\n[CONFIDENCE PROBE] verdict={manifest['verdict']} "
        f"best={manifest['best_signal']} auc={manifest['best_auc']}",
        flush=True,
    )

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/confidence_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[CONFIDENCE MANIFEST] {json.dumps(manifest)}", flush=True)  # single-line, survives tail

    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo="confidence_probe")
        print(f"pushed probe manifest to {args.out_repo}/confidence_probe", flush=True)


def _preflight(device, window):  # noqa: ANN001
    """GRAD_OK-style fail-loud: run the FULL collect+report path on a throwaway
    tiny predictor + random chains on `device` before the expensive encode, so a
    GPU-only device/shape bug crashes in minute 1 (twice-bitten: cpu/cuda gather,
    to_leaf_param) rather than after a 60-min encode."""
    k, d = 8, 16
    m = NextThoughtPredictor(d=d, k=k, d_model=16, layers=1, heads=2).to(device)
    seqs = [torch.randn(window + 2, k, d) for _ in range(4)]
    rep = _report(collect(m, seqs, window, device, n_dropout=3))
    assert "verdict" in rep and rep["n_pairs"] > 0, f"preflight incomplete: {rep}"
    print(f"PREFLIGHT_OK {rep['verdict']} (n={rep['n_pairs']})", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
