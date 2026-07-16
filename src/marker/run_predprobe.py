"""PREDPROBE (Fable-specced): does a PREDICTED gist, rendered to words, keep
its arithmetic structure? Pure eval, no training, no solve-generation.

Render-of-TRUE-gist is 0.92 relations-exact (validated, gistprobe). We have
never measured render-of-PREDICTED-gist -- this probe measures it, and it
gates the whole fuzzy-thought direction: a good number means the
plan-then-refine loop is assembly work around parts we already own; a mushy
one means the predictor's imitation training is the bottleneck and the real
project is an outcome-based training signal.

Render reads per-layer K/V (gist_kv); the predictor only emits a final-layer
SUMMARY. A predicted thought reaches render via
predict_step -> bridge(summary) -> KV ("Path B"), which is NOT the
gist_kv(true text) distribution render was trained on ("Path A"). Five
rendered conditions per scorable step (n>=1 -- predict_step needs >=1 prior
step, so step 0 of every doc is unscorable for ALL conditions, including the
ceiling):

    true_gistkv     gist_kv(true step n)                    -- Path A ceiling
    true_bridged    encode_gist(true n) -> bridge            -- Path B on truth (=B)
    wrong_bridged   encode_gist(step from a DIFFERENT doc)   -- Path B floor  (=W)
    pred_bridged    predict_step(true ctx 1..n-1) -> bridge  -- Path B on prediction (=P)
    noised_bridged  bridge(noised(true summary n, ratio=1.0))-- magnitude-matched noise (=N)

plus a non-gating secondary: pred_bridged with the literals ledger WITHHELD
(the honest preview of the closed loop, where the predictor must supply
literals itself).

Every condition hands the TRUE step-n ledger + true first token (parity with
the 0.92 ceiling) EXCEPT the secondary. Read margins over wrong_bridged
(H = true_bridged - wrong_bridged), never absolute relations-exact -- the
ledger flatters every condition's absolute number. See predprobe.gated_verdict
for the full gate table; it is a CONVENIENCE field only, never a hard
assertion -- the human + Fable read the manifest.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_predprobe \\
        --repo mattyvee/mimir-artifacts --artifacts-repo mattyvee/mimir-artifacts \\
        --subdir stage2_cot_openr1 --bridge-subdir bridge_validated \\
        --render-subdir render_adapter_ledger
Smoke: PYTHONPATH=src python -m marker.run_predprobe --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.gistprobe import digit_token_mask, per_token_ce, relation_score
from marker.predprobe import (
    CEILING,
    bridged_condition,
    gated_verdict,
    pick_cross_doc_step,
    scorable_ns,
)
from marker.run_bridge import noised, predict_step
from marker.run_frontload import context_split


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None, help="Stage-1 gist adapter repo")
    ap.add_argument("--artifacts-repo", default=None, help="HF repo with predictor/bridge/render")
    ap.add_argument("--subdir", default="stage2_cot_openr1", help="predictor.pt subdir")
    ap.add_argument("--bridge-subdir", default="bridge_validated")
    ap.add_argument("--render-subdir", default="render_adapter_ledger")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--n-problems", type=int, default=63)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument(
        "--window",
        type=int,
        default=8,
        help="predictor input window: MUST match the predictor's training window "
        "(stage2 default 8) so sentence-position embeddings stay in-distribution",
    )
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.n_problems = "Qwen/Qwen2.5-0.5B", None, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.bridge import GistBridge  # noqa: PLC0415
    from marker.gist_model import encode_gist, gist_kv  # noqa: PLC0415
    from marker.predictor import NextThoughtPredictor  # noqa: PLC0415
    from marker.reason_check import split_solution_steps  # noqa: PLC0415
    from marker.render import extract_ledger  # noqa: PLC0415
    from marker.run_render import _f1_tok, _num_recall, _render_reconstruct  # noqa: PLC0415
    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    k = gist.shape[0]
    nl_id = next((t for t in tok("\n", add_special_tokens=False).input_ids if t), None)
    probe_kv, _, _ = gist_kv(pm, gist, tok("hi", add_special_tokens=False).input_ids)
    kv_dtype = probe_kv.keys[0].dtype

    # ── predictor + bridge (Path B) -- reference wiring: run_frontload.py ────
    from marker.run_confidence import _predictor_from_state  # noqa: PLC0415
    from marker.run_rollout import _load_bridge  # noqa: PLC0415

    if args.smoke:
        predictor = NextThoughtPredictor(d=gist.shape[-1], k=k, d_model=48, layers=2, heads=4)
        bridge = (
            GistBridge(
                d=gist.shape[-1],
                k=k,
                n_layers=probe_kv.n_layers,
                n_kv_heads=probe_kv.keys[0].shape[1],
                head_dim=probe_kv.keys[0].shape[3],
                width=64,
            )
            .to(device)
            .eval()
        )
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        predictor = _predictor_from_state(
            torch.load(
                hf_hub_download(args.artifacts_repo, f"{args.subdir}/predictor.pt"),
                map_location="cpu",
            ),
            args.heads,
        )
        bridge = _load_bridge(
            torch.load(
                hf_hub_download(args.artifacts_repo, f"{args.bridge_subdir}/bridge.pt"),
                map_location="cpu",
            ),
            gist.shape[-1],
            k,
            probe_kv,
            device,
        )
    predictor = predictor.to(device).eval()

    # ── render adapter (the trained gist-reader) ─────────────────────────────
    if args.smoke:
        from marker.render import attach_render  # noqa: PLC0415

        attach_render(pm, r=4)
        pm.set_adapter("default")
    else:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        loc = snapshot_download(
            args.artifacts_repo, allow_patterns=[f"{args.render_subdir}/render/*"]
        )
        pm.load_adapter(f"{loc}/{args.render_subdir}/render", adapter_name="render")
        pm.set_adapter("default")
        print(f"render adapter loaded from {args.render_subdir}", flush=True)

    # ── the two DOC-preserving step sets: hard (>=7-step problems, m-cap 6 --
    # the reconstitute config) and easy control (<7-step, m-cap 4). Docs (not a
    # flattened step list, gistprobe's shape) because pred_bridged needs each
    # doc's OWN true-summary history and wrong_bridged needs a DIFFERENT doc to
    # pair against. ──────────────────────────────────────────────────────────
    def _docs_for(min_steps, max_steps, m_cap):  # noqa: ANN001
        if args.smoke:
            from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

            texts = _smoke_cot_texts(args.n_problems)
        else:
            from datasets import load_dataset  # noqa: PLC0415

            ds = load_dataset(args.dataset, "main", split="test", streaming=True)
            texts = []
            for row in ds:
                if len(texts) >= args.n_problems:
                    break
                n = len(split_solution_steps(row["answer"]))
                if min_steps <= n < max_steps:
                    texts.append(row["answer"])
        out = []
        for t in texts:
            steps = split_solution_steps(t)
            if len(steps) < 3:
                continue
            m = context_split(len(steps), m_cap)
            doc = []
            for s in steps[:m]:
                ids = tok(s, add_special_tokens=False).input_ids[: args.max_span]
                if len(ids) >= 3:
                    doc.append((s, ids))
            if len(doc) >= 2:  # need >=1 scorable n (n>=1 requires m>=2)
                out.append(doc)
        return out

    sets = {
        "hard": _docs_for(7, 10**9, 6),
        "easy": _docs_for(3, 7, 4),
    }
    print({label: len(docs) for label, docs in sets.items()}, "docs per set", flush=True)

    def _ledger_ids(text):  # noqa: ANN001
        nums = extract_ledger(text)
        return tok(" ".join(nums) + "\n", add_special_tokens=False).input_ids if nums else []

    CONDS = ("true_gistkv", "true_bridged", "wrong_bridged", "pred_bridged", "noised_bridged")

    @torch.no_grad()
    def _probe_set(docs, label):  # noqa: ANN001, PLR0915
        pair_gen = torch.Generator().manual_seed(0)  # wrong_bridged cross-doc draw
        noise_gen = torch.Generator().manual_seed(1)  # noised_bridged jitter (CPU-only, see noised)
        doc_lengths = [len(d) for d in docs]
        summs = [encode_gist(pm, gist, [ids for _, ids in doc]).float() for doc in docs]

        agg = {
            c: {"f1": [], "numrec": [], "rel": [], "nll_full": [], "nll_struct": []} for c in CONDS
        }
        secondary = {"f1": [], "numrec": [], "rel": [], "nll_full": [], "nll_struct": []}
        per_step = []
        shown = 0

        for di, doc in enumerate(docs):
            summ = summs[di]
            for n in scorable_ns(len(doc)):
                text, ids = doc[n]
                led = _ledger_ids(text)

                kv_true, cs_true, _ = gist_kv(pm, gist, ids)  # Path A, gistprobe's own convention
                kv_tb, cs_tb = bridged_condition(bridge, summ[n], kv_dtype)
                dj, sj = pick_cross_doc_step(di, doc_lengths, pair_gen, step_idx=n)
                kv_wb, cs_wb = bridged_condition(bridge, summs[dj][sj], kv_dtype)
                ghat = predict_step(predictor, summ, n, args.window)  # windowed (bug A)
                kv_pb, cs_pb = bridged_condition(bridge, ghat, kv_dtype)
                ghat_noised = noised(summ[n], 1.0, noise_gen)
                kv_nb, cs_nb = bridged_condition(bridge, ghat_noised, kv_dtype)

                for cond, kv, cs in (
                    ("true_gistkv", kv_true, cs_true),
                    ("true_bridged", kv_tb, cs_tb),
                    ("wrong_bridged", kv_wb, cs_wb),
                    ("pred_bridged", kv_pb, cs_pb),
                    ("noised_bridged", kv_nb, cs_nb),
                ):
                    pm.set_adapter("render")
                    rec = _render_reconstruct(
                        pm, kv, cs, ids[0], args.max_span, {nl_id}, prefix_ids=led
                    )
                    ce, tgt = per_token_ce(pm, kv, cs, led, list(ids) + [nl_id])
                    pm.set_adapter("default")
                    rtext = tok.decode(rec)
                    f1 = _f1_tok(rec, ids)
                    agg[cond]["f1"].append(f1)
                    nr = _num_recall(rtext, text)
                    if nr is not None:
                        agg[cond]["numrec"].append(nr)
                    rs = relation_score(rtext, text)
                    if rs["exact"] is not None:
                        agg[cond]["rel"].append(rs["exact"])
                    mask = digit_token_mask(tok.convert_ids_to_tokens(tgt))
                    agg[cond]["nll_full"].append(float(ce.mean()))
                    nll_struct = None
                    if (~mask).any():
                        nll_struct = float(ce[~mask.to(ce.device)].mean())
                        agg[cond]["nll_struct"].append(nll_struct)
                    per_step.append(
                        {
                            "set": label,
                            "doc": di,
                            "n": n,
                            "cond": cond,
                            "rel": rs["exact"],
                            "f1": round(f1, 4),
                            "nll_struct": nll_struct,
                        }
                    )
                    if shown < 3 and cond == "true_gistkv":
                        print(f"[{label}] STEP : {text!r}\n      RECON: {rtext!r}", flush=True)
                        shown += 1

                # secondary (non-gating): pred_bridged with the ledger WITHHELD
                pm.set_adapter("render")
                rec2 = _render_reconstruct(
                    pm, kv_pb, cs_pb, ids[0], args.max_span, {nl_id}, prefix_ids=[]
                )
                ce2, tgt2 = per_token_ce(pm, kv_pb, cs_pb, [], list(ids) + [nl_id])
                pm.set_adapter("default")
                rtext2 = tok.decode(rec2)
                f1_2 = _f1_tok(rec2, ids)
                secondary["f1"].append(f1_2)
                nr2 = _num_recall(rtext2, text)
                if nr2 is not None:
                    secondary["numrec"].append(nr2)
                rs2 = relation_score(rtext2, text)
                if rs2["exact"] is not None:
                    secondary["rel"].append(rs2["exact"])
                mask2 = digit_token_mask(tok.convert_ids_to_tokens(tgt2))
                secondary["nll_full"].append(float(ce2.mean()))
                nll_struct2 = None
                if (~mask2).any():
                    nll_struct2 = float(ce2[~mask2.to(ce2.device)].mean())
                    secondary["nll_struct"].append(nll_struct2)
                per_step.append(
                    {
                        "set": label,
                        "doc": di,
                        "n": n,
                        "cond": "pred_bridged_no_ledger",
                        "rel": rs2["exact"],
                        "f1": round(f1_2, 4),
                        "nll_struct": nll_struct2,
                    }
                )

        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        results = {
            c: {k: mean(v) for k, v in d.items()} | {"n": len(d["f1"])} for c, d in agg.items()
        }
        results["pred_bridged_no_ledger"] = {k: mean(v) for k, v in secondary.items()} | {
            "n": len(secondary["f1"])
        }

        # raw means, None-safe: gated_verdict reads any None (a set with no
        # extractable relations / struct tokens) as INSUFFICIENT_DATA -- never
        # coerce to 0.0 (fakes a gate-0 INVALID, or a fake headroom for W)
        gate = gated_verdict(
            results["true_gistkv"]["rel"],
            results["true_bridged"]["rel"],
            results["wrong_bridged"]["rel"],
            results["pred_bridged"]["rel"],
            results["wrong_bridged"]["nll_struct"],
            results["true_bridged"]["nll_struct"],
            results["pred_bridged"]["nll_struct"],
        )

        return results, gate, per_step

    results, gates, per_step = {}, {}, []
    for label, docs in sets.items():
        if not docs:
            continue
        r, g, ps = _probe_set(docs, label)
        results[label], gates[label] = r, g
        per_step.extend(ps)

    manifest = {
        "n_problems": args.n_problems,
        "window": args.window,
        # reference only -- the run's actual (recomputed, n>=1) ceiling is
        # results[set]["true_gistkv"]["rel"]; gate 0 floors on that, not this
        "published_ceiling": CEILING,
        "results": results,
        "gate": gates,
        "per_step": per_step,
    }
    print("\n[PREDPROBE]", flush=True)
    for label, r in results.items():
        for cond in (*CONDS, "pred_bridged_no_ledger"):
            print(f"  {label:5s} {cond:24s} {r[cond]}", flush=True)
        print(f"  {label:5s} GATE {gates[label]}", flush=True)

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/predprobe_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[PREDPROBE MANIFEST] {json.dumps(manifest)}", flush=True)  # single-line, survives tail
    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo="predprobe")
        print(f"pushed predprobe manifest to {args.out_repo}/predprobe", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
