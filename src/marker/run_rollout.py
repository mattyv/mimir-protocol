"""Latent chain rollout harness — the real fast-lane test.

Loads the trained encoder, predictor, and bridge, then for each held-out doc
fixes a real PREFIX of thoughts and free-runs the predictor on its own outputs
(rollout.py). At each rollout DEPTH it injects the predicted thought through the
bridge and scores how well the frozen model then predicts the true next step
(gap_closed), against a teacher-forced control (predict from TRUE history) and
the none/full/gist_true/shuffled anchors. The drift curve = 'free' falling
toward 'shuffled' as depth grows; the headline is the depth where a free-running
thought stops beating the shuffled floor.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_rollout \\
        --repo mattyvee/mimir-artifacts \\
        --artifacts-repo mattyvee/mimir-artifacts --subdir stage2_cot_openr1 \\
        --dataset open-r1/OpenR1-Math-220k --n-docs 400 --skip-docs 2000 \\
        --prefix 2 --max-depth 12
Smoke (local tiny model, random predictor+bridge — mechanics only):
    PYTHONPATH=src python -m marker.run_rollout --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.bridge import GistBridge
from marker.predictor import NextThoughtPredictor
from marker.rollout import drift_by_depth, rollout, teacher_forced
from marker.run_bridge import _encode_doc, _token_cache, tail_nll


def _load_bridge(state: dict, d: int, k: int, probe_kv, device):  # noqa: ANN001
    """Rebuild GistBridge with width inferred from the checkpoint and the KV
    geometry from a real gist_kv sample (n_kv_heads/head_dim/n_layers)."""
    width = state["trunk.0.weight"].shape[0]
    b = GistBridge(
        d=d,
        k=k,
        n_layers=probe_kv.n_layers,
        n_kv_heads=probe_kv.keys[0].shape[1],
        head_dim=probe_kv.keys[0].shape[3],
        width=width,
    )
    b.load_state_dict(state)
    print(f"bridge: width={width} n_layers={probe_kv.n_layers}", flush=True)
    return b.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--artifacts-repo", default=None)
    ap.add_argument("--subdir", default="stage2_cot_openr1")
    ap.add_argument("--bridge-subdir", default="bridge")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--corpus", default="cot")
    ap.add_argument("--unit", choices=["line", "sentence"], default=None)
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--skip-docs", type=int, default=2000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument(
        "--prefix", type=int, default=2, help="real thoughts before free-running starts"
    )
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--max-span", type=int, default=96)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    unit = args.unit or ("line" if args.corpus == "cot" else "sentence")
    if args.smoke:
        args.model_name, args.repo, args.n_docs = "Qwen/Qwen2.5-0.5B", None, 40
        args.skip_docs, args.max_span, args.prefix, args.max_depth = 0, 24, 2, 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.gist_model import gist_kv  # noqa: PLC0415
    from marker.run_axiom_mlp_demo import _build_dynamic_cache  # noqa: PLC0415
    from marker.run_stage2 import _doc_texts, _load_stage1, _smoke_cot_texts

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )

    if args.smoke:
        predictor = NextThoughtPredictor(
            d=gist.shape[-1], k=gist.shape[0], d_model=48, layers=2, heads=4
        )
        docs = _smoke_cot_texts(args.n_docs)
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        from marker.run_confidence import _predictor_from_state  # noqa: PLC0415

        pp = hf_hub_download(repo_id=args.artifacts_repo, filename=f"{args.subdir}/predictor.pt")
        predictor = _predictor_from_state(torch.load(pp, map_location="cpu"), args.heads)
        gen = _doc_texts(
            args.n_docs + args.skip_docs,
            args.corpus,
            args.dataset,
            args.dataset_config,
            args.text_field,
        )
        docs = list(gen)[args.skip_docs :]
    predictor = predictor.to(device).eval()

    encoded = [e for e in (_encode_doc(pm, gist, tok, t, unit, args.max_span) for t in docs) if e]
    print(f"encoded {len(encoded)} held-out docs", flush=True)

    probe_kv, _, _ = gist_kv(pm, gist, encoded[0][0][0])
    kv_dtype = probe_kv.keys[0].dtype
    if args.smoke:
        bridge = (
            GistBridge(
                d=gist.shape[-1],
                k=gist.shape[0],
                n_layers=probe_kv.n_layers,
                n_kv_heads=probe_kv.keys[0].shape[1],
                head_dim=probe_kv.keys[0].shape[3],
                width=256,
            )
            .to(device)
            .eval()
        )
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        bp = hf_hub_download(
            repo_id=args.artifacts_repo, filename=f"{args.bridge_subdir}/bridge.pt"
        )
        bridge = _load_bridge(
            torch.load(bp, map_location="cpu"), gist.shape[-1], gist.shape[0], probe_kv, device
        )

    def _bridge_cache(vec):  # noqa: ANN001
        kv = bridge(vec.to(device))
        return _build_dynamic_cache(
            type(kv)(
                kv.n_layers, [k.to(kv_dtype) for k in kv.keys], [v.to(kv_dtype) for v in kv.values]
            ),
            device,
        )

    cos = torch.nn.functional.cosine_similarity

    # ── roll out every doc, bucket scores by rollout depth ───────────────────
    by_depth: dict[int, dict[str, list[float]]] = {}

    def _bucket(d, rung, val):  # noqa: ANN001
        by_depth.setdefault(
            d,
            {
                r: []
                for r in (
                    "none",
                    "full",
                    "gist_true",
                    "tf",
                    "free",
                    "shuffled",
                    "free_cos",
                    "tf_cos",
                )
            },
        )[rung].append(val)

    P = args.prefix
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for di, (ids, summ) in enumerate(encoded):
            L = len(ids)
            if L < P + 2:
                continue
            summ_dev = summ.to(device)
            depth = min(args.max_depth, L - 1 - P)  # predict steps P..P+depth-1 (need step t+1)
            if depth < 1:
                continue
            free = rollout(predictor, summ_dev[:P], depth, args.window)  # [depth,k,d]
            tf = teacher_forced(predictor, summ_dev, P, depth, args.window)  # [depth,k,d]
            for d in range(1, depth + 1):
                t = P + d - 1  # step being predicted
                cont = ids[t + 1]
                _bucket(d, "none", tail_nll(pm, None, 0, cont))
                cache, cs = _token_cache(pm, ids[t])
                _bucket(d, "full", tail_nll(pm, cache, cs, cont))
                kv, cs, _ = gist_kv(pm, gist, ids[t])
                _bucket(d, "gist_true", tail_nll(pm, _build_dynamic_cache(kv, device), cs, cont))
                _bucket(d, "tf", tail_nll(pm, _bridge_cache(tf[d - 1]), bridge.k, cont))
                _bucket(d, "free", tail_nll(pm, _bridge_cache(free[d - 1]), bridge.k, cont))
                # cross-doc mislead control
                dj = di
                while dj == di and len(encoded) > 1:
                    dj = int(torch.randint(0, len(encoded), (1,), generator=rng))
                o_ids, o_summ = encoded[dj]
                jj = int(torch.randint(0, len(o_ids), (1,), generator=rng))
                _bucket(d, "shuffled", tail_nll(pm, _bridge_cache(o_summ[jj]), bridge.k, cont))
                _bucket(d, "free_cos", float(cos(free[d - 1], summ_dev[t], dim=-1).mean()))
                _bucket(d, "tf_cos", float(cos(tf[d - 1], summ_dev[t], dim=-1).mean()))

    curve = drift_by_depth(by_depth)
    manifest = {
        "probe_dataset": args.dataset,
        "prefix": P,
        "window": args.window,
        "skip_docs": args.skip_docs,
        "n_docs": len(encoded),
        "drift_curve": curve,
    }
    print("\n[ROLLOUT DRIFT] depth: free / tf / shuffled gap_closed  (free_cos)", flush=True)
    for d in sorted(curve):
        c = curve[d]
        print(
            f"  d={d:2d} n={c['n']:3d}  free={c['free']}  tf={c['tf']}  "
            f"shuf={c['shuffled']}  gist={c['gist_true']}  free_cos={c['free_cos']}",
            flush=True,
        )

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    dd = Path("/tmp/rollout_out")  # noqa: S108
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ROLLOUT MANIFEST] {json.dumps(manifest)}", flush=True)

    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(dd), path_in_repo="rollout")
        print(f"pushed rollout manifest to {args.out_repo}/rollout", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
