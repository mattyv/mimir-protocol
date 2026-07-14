"""Gist fidelity probe (the ~$1 discriminator, Fable-specced): is the RELATIONAL
structure of a reasoning step in the 8-slot gist at all — and does the render
reader actually read it?

Pure eval, no training, no solve-generation. Over the SAME context steps the
reconstitute experiment used (GSM8K test: hard >=7-step problems, plus an easy
control set), per step:

  true-gist recon    render-reconstruct from the step's own gist (ledger +
                     first-token sidecar, the validated config) -> token-F1,
                     number recall, RELATION score (a op b = c extraction)
  wrong-gist recon   same ledger + first token, but ANOTHER step's gist —
                     if scores stay close to true-gist, the reader is a
                     ledger-crutched prior and barely reads the gist
  NLL contrast       teacher-forced CE of the TRUE step under render, true vs
                     wrong gist, averaged over NON-digit (structure) tokens
                     only. Needs no generation: if true-gist structure-NLL is
                     clearly lower, the relational info IS in the gist and the
                     greedy reader is the limiter; if flat, the gist lacks it.

Verdict grid:
  rel_exact(true) >> rel_exact(wrong) and struct-NLL contrast big
      -> gist holds structure; READER is the weak link (retrain reader)
  contrast flat -> gist lacks structure -> encoder (web-trained) or k is the
      wall (retrain encoder on CoT / k=16)
  hard ~= easy everywhere -> no hard-specific drop; plain compounding story

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_gistprobe \\
        --repo mattyvee/mimir-artifacts --artifacts-repo mattyvee/mimir-artifacts
Smoke: PYTHONPATH=src python -m marker.run_gistprobe --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.burst import extract_answer  # noqa: F401  (dataset field sanity reuse)
from marker.gistprobe import digit_token_mask, per_token_ce, relation_score
from marker.run_frontload import context_split


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--artifacts-repo", default=None)
    ap.add_argument("--render-subdir", default="render_adapter_ledger")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--n-problems", type=int, default=63)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.n_problems = "Qwen/Qwen2.5-0.5B", None, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.gist_model import gist_kv  # noqa: PLC0415
    from marker.reason_check import split_solution_steps  # noqa: PLC0415
    from marker.render import extract_ledger  # noqa: PLC0415
    from marker.run_render import _f1_tok, _num_recall, _render_reconstruct  # noqa: PLC0415
    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    nl_id = next((t for t in tok("\n", add_special_tokens=False).input_ids if t), None)

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

    # ── the two step sets: hard (>=7-step problems, m-cap 6 — the reconstitute
    # config) and easy control (<7-step problems, m-cap 4) ────────────────────
    def _steps_for(min_steps, max_steps, m_cap):  # noqa: ANN001
        if args.smoke:
            from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

            docs = _smoke_cot_texts(args.n_problems)
        else:
            from datasets import load_dataset  # noqa: PLC0415

            ds = load_dataset(args.dataset, "main", split="test", streaming=True)
            docs = []
            for row in ds:
                if len(docs) >= args.n_problems:
                    break
                n = len(split_solution_steps(row["answer"]))
                if min_steps <= n < max_steps:
                    docs.append(row["answer"])
        out = []
        for d in docs:
            steps = split_solution_steps(d)
            if len(steps) < 3:
                continue
            m = context_split(len(steps), m_cap)
            for s in steps[:m]:
                ids = tok(s, add_special_tokens=False).input_ids[: args.max_span]
                if len(ids) >= 3:
                    out.append((s, ids))
        return out

    sets = {
        "hard": _steps_for(7, 10**9, 6),
        "easy": _steps_for(3, 7, 4),
    }
    print({k: len(v) for k, v in sets.items()}, "steps per set", flush=True)

    def _ledger_ids(text):  # noqa: ANN001
        nums = extract_ledger(text)
        return tok(" ".join(nums) + "\n", add_special_tokens=False).input_ids if nums else []

    @torch.no_grad()
    def _probe_set(items, label):  # noqa: ANN001, PLR0915
        # fixed wrong-gist pairing: shift-by-1 (never self)
        n = len(items)
        agg = {
            c: {"f1": [], "numrec": [], "rel": [], "opseq": [], "nll_full": [], "nll_struct": []}
            for c in ("true", "wrong")
        }
        shown = 0
        for i, (text, ids) in enumerate(items):
            led = _ledger_ids(text)
            kv_true, cs_true, _ = gist_kv(pm, gist, ids)
            w_text, w_ids = items[(i + 1) % n]
            kv_wrong, _, _ = gist_kv(pm, gist, w_ids)
            # wrong gist decodes from ITS OWN frame; ledger/first-tok are true's
            cs_wrong = len(w_ids) + gist.shape[0]
            for cond, kv, cs in (("true", kv_true, cs_true), ("wrong", kv_wrong, cs_wrong)):
                pm.set_adapter("render")
                rec = _render_reconstruct(
                    pm, kv, cs, ids[0], args.max_span, {nl_id}, prefix_ids=led
                )
                # teacher-forced CE of the TRUE step (+ newline, the training
                # target) under this gist — the generation-free structure probe
                ce, tgt = per_token_ce(pm, kv, cs, led, list(ids) + [nl_id])
                pm.set_adapter("default")
                rtext = tok.decode(rec)
                agg[cond]["f1"].append(_f1_tok(rec, ids))
                nr = _num_recall(rtext, text)
                if nr is not None:
                    agg[cond]["numrec"].append(nr)
                rs = relation_score(rtext, text)
                if rs["exact"] is not None:
                    agg[cond]["rel"].append(rs["exact"])
                    agg[cond]["opseq"].append(1.0 if rs["op_seq"] else 0.0)
                mask = digit_token_mask(tok.convert_ids_to_tokens(tgt))
                agg[cond]["nll_full"].append(float(ce.mean()))
                if (~mask).any():
                    agg[cond]["nll_struct"].append(float(ce[~mask.to(ce.device)].mean()))
                if shown < 3 and cond == "true":
                    print(f"[{label}] STEP : {text!r}\n      RECON: {rtext!r}", flush=True)
                    shown += 1
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        return {c: {k: mean(v) for k, v in d.items()} | {"n": len(d["f1"])} for c, d in agg.items()}

    results = {label: _probe_set(items, label) for label, items in sets.items() if items}

    manifest = {"n_problems": args.n_problems, "results": results}
    print("\n[GISTPROBE]", flush=True)
    for label, r in results.items():
        for cond in ("true", "wrong"):
            print(f"  {label:5s} {cond:5s} {r[cond]}", flush=True)

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/gistprobe_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[GISTPROBE MANIFEST] {json.dumps(manifest)}", flush=True)
    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo="gistprobe")
        print(f"pushed gistprobe manifest to {args.out_repo}/gistprobe", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
