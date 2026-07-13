"""Fast-lane confidence probe (FASTLANE_PLAN gate) — the cheap, decisive test.

Reuses the ALREADY-TRAINED predictor (no new training). On held-out reasoning
chains, for every (thought -> next-thought) prediction it records whether the
predictor's top-1 pick is the true next thought (within-document pool, the
topic-shortcut control) and inference-time confidence signals, then asks: does
any signal SEPARATE the correct predictions from the wrong ones — and at a
usable skip-rate?

Gate readout (Fable review hardened):
  - the deployable metric is PRECISION @ COVERAGE: of the steps we'd actually
    skip (top-x% by confidence), what fraction are correct? — not AUC alone,
    which can be "real but useless" at a 30% base rate.
  - signals are split select/confirm BY DOCUMENT: the best signal is chosen on
    the select half, the verdict is read on the confirm half (no cherry-pick).
  - only INFERENCE-CLEAN signals feed the verdict (prediction_norm,
    dropout_agreement); retrieval_margin is a diagnostic only — its bank would
    contain the not-yet-generated true thought, which no deployment has.
  - a ONE-STEP-DRIFT block re-measures with the last input thought replaced by
    the predictor's own (drifting) prediction — a first-order look at the
    rollout distribution the teacher-forced probe never sees.

  PASS => build the bridge + gated rollout (to TEST the rollout — a clean-input
          PASS is necessary, not sufficient, for the rollout to work).
  FAIL => confidence-gating is dead; the render lane stays the deliverable.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_confidence \\
        --repo mattyvee/mimir-artifacts \\
        --artifacts-repo mattyvee/mimir-artifacts --subdir stage2_cot_openr1 \\
        --dataset openai/gsm8k --n-docs 400 --window 6 --skip-docs 0
Smoke (local tiny model, random predictor — mechanics only):
    PYTHONPATH=src python -m marker.run_confidence --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.confidence import (
    coverage_curve,
    dropout_agreement,
    mc_dropout_pool,
    prediction_norm,
    rank_auc,
    retrieval_margin,
    slot_cosine,
    tercile_report,
    within_doc_correct,
)
from marker.predictor import NextThoughtPredictor

# pre-registered gate (Fable review: precision@coverage, not AUC): on the
# CONFIRM split, the best inference-clean signal must make its top-COVERAGE
# skip-bin reliably correct and clearly above the blind base rate.
GATE_COVERAGE = 0.2
GATE_MIN_ACC = 0.60
GATE_MIN_LIFT = 0.15
GATE_MIN_BIN = 25


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


def _load_whitener(artifacts_repo, subdir):  # noqa: ANN001
    """Load the predictor's whitener from HF (pushed beside predictor.pt) so the
    probe feeds the model gists in the SAME space it was trained in. A whitened
    checkpoint probed in raw space craters correctness and fakes a FAIL (Fable
    review). Absent / identity file -> IdentityWhitener (no-op, free)."""
    from marker.whiten import IdentityWhitener, PerSlotWhitener, Whitener  # noqa: PLC0415

    if not artifacts_repo:
        return IdentityWhitener(), "identity(none)"
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    try:
        wp = hf_hub_download(repo_id=artifacts_repo, filename=f"{subdir}/whiteners.pt")
    except Exception as e:  # noqa: BLE001
        print(f"no whiteners.pt ({type(e).__name__}); using identity", flush=True)
        return IdentityWhitener(), "identity(absent)"
    raw = torch.load(wp, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and raw.get("identity"):
        return IdentityWhitener(), "identity"
    w = PerSlotWhitener([Whitener(d["mean"], d["w"], d["w_inv"]) for d in raw])
    return w, "per-slot"


def _windowize(seqs, length):  # noqa: ANN001
    from marker.run_stage2 import _windows  # noqa: PLC0415

    wins, win_doc = [], []
    for di, s in enumerate(seqs):
        for w in _windows(s, length):
            wins.append(w)
            win_doc.append(di)
    if not wins:
        raise RuntimeError("no windows: need docs with >= window+1 steps")
    return wins, win_doc


@torch.no_grad()
def _predict_pooled(model, x, dm):  # noqa: ANN001
    """Eval-mode prediction over a [B, T, k, d] batch -> (pooled preds, pred
    slots) for positions 1..T-1. Whitening is applied by the caller."""
    pred = model(x)  # [B, T-1, k, d]
    return model.pool(pred).reshape(-1, dm), pred.reshape(-1, *pred.shape[2:])


@torch.no_grad()
def collect(model, seqs, length, device, n_dropout, whitener):  # noqa: ANN001
    """Teacher-forced pass: every (position i -> i+1) pair over all length-L
    windows. Gathers pooled pred/target, doc id, pred/target SLOTS (absolute
    label), and dropout agreement. Correctness/norm/margin/slot-cosine computed
    in _finalize AFTER dedup. Whitening mirrors run_stage2.evaluate()."""
    wins, win_doc = _windowize(seqs, length)
    was_training = model.training
    model.eval()
    dm = model.pool_proj.out_features
    preds, tgts, pslots, tslots, agree = [], [], [], [], []
    for i in range(0, len(wins), 64):
        stack = torch.stack(wins[i : i + 64])  # [B, L, k, d] cpu
        b, ln, k, d = stack.shape
        wz = whitener.transform(stack.reshape(b * ln, k, d)).reshape(b, ln, k, d).float().to(device)
        pp, ps = _predict_pooled(model, wz, dm)
        preds.append(pp.cpu())
        pslots.append(ps.cpu())
        tgts.append(model.pool(wz[:, 1:]).reshape(-1, dm).cpu())
        tslots.append(wz[:, 1:].reshape(-1, k, d).cpu())
        samples = mc_dropout_pool(model, wz, n_samples=n_dropout)  # [S, B*(L-1), dm]
        agree.append(dropout_agreement(samples).cpu())
    model.train(was_training)
    doc = torch.tensor(win_doc).repeat_interleave(length - 1)
    return {
        "pred": torch.cat(preds),
        "tgt": torch.cat(tgts),
        "pred_slots": torch.cat(pslots),
        "tgt_slots": torch.cat(tslots),
        "agree": torch.cat(agree),
        "doc": doc,
    }


@torch.no_grad()
def collect_drift(model, seqs, length, device, n_dropout, whitener):  # noqa: ANN001
    """One-step-drift pass (needs L>=3): predict the LAST step of each window
    with its immediately-preceding input thought replaced by the predictor's
    OWN prediction of it — a first-order look at the rollout distribution the
    teacher-forced probe never sees. One (pred, target) per window."""
    wins, win_doc = _windowize(seqs, length)
    if length < 3:
        return None
    was_training = model.training
    model.eval()
    preds, tgts, pslots, tslots, agree, docs = [], [], [], [], [], []
    for i in range(0, len(wins), 64):
        stack = torch.stack(wins[i : i + 64])  # [B, L, k, d]
        b, ln, k, d = stack.shape
        wz = whitener.transform(stack.reshape(b * ln, k, d)).reshape(b, ln, k, d).float().to(device)
        # ŵ_{L-2}: predict the second-to-last thought from w0..w_{L-3}
        ghat = model(wz[:, : ln - 1])[:, -1]  # [B, k, d]
        drifted = torch.cat([wz[:, : ln - 2], ghat.unsqueeze(1)], dim=1)  # [B, L-1, k, d]
        pred = model(drifted)[:, -1]  # predicts w_{L-1} from the drifted prefix
        preds.append(model.pool(pred).cpu())
        pslots.append(pred.cpu())
        tgts.append(model.pool(wz[:, -1]).cpu())
        tslots.append(wz[:, -1].cpu())
        # dropout agreement on the drifted prediction
        s = torch.stack(
            [model.pool(_drift_once(model, wz, ln)).cpu() for _ in _dropout_iter(model, n_dropout)]
        )
        agree.append(dropout_agreement(s))
        docs.extend(win_doc[i : i + b])
    model.train(was_training)
    return {
        "pred": torch.cat(preds),
        "tgt": torch.cat(tgts),
        "pred_slots": torch.cat(pslots),
        "tgt_slots": torch.cat(tslots),
        "agree": torch.cat(agree),
        "doc": torch.tensor(docs),
    }


def _dropout_iter(model, n):  # noqa: ANN001
    """Enable dropout for n stochastic passes, restore mode after."""
    model.train()
    try:
        yield from range(n)
    finally:
        model.eval()


def _drift_once(model, wz, ln):  # noqa: ANN001
    ghat = model(wz[:, : ln - 1])[:, -1]
    drifted = torch.cat([wz[:, : ln - 2], ghat.unsqueeze(1)], dim=1)
    return model(drifted)[:, -1]


def _finalize(raw: dict) -> dict:
    """Dedup duplicate targets (guaranteed false negatives; mirrors
    evaluate()), then compute correctness + all confidence signals + the
    absolute slot-cosine label on the deduped pool."""
    from marker.run_stage2 import _dedup_keep_idx  # noqa: PLC0415

    keep = _dedup_keep_idx(raw["tgt"])
    p, t = raw["pred"][keep], raw["tgt"][keep]
    doc = raw["doc"][keep]
    return {
        "n": int(len(keep)),
        "dropped": int(len(raw["tgt"]) - len(keep)),
        "correct": within_doc_correct(p, t, doc),
        "doc": doc,
        "signals": {
            "prediction_norm": prediction_norm(p),
            "dropout_agreement": raw["agree"][keep],
            "retrieval_margin": retrieval_margin(p, t),  # DIAGNOSTIC ONLY (bank leaks)
        },
        "abs_cos": slot_cosine(raw["pred_slots"][keep], raw["tgt_slots"][keep]),
    }


def _doc_split(doc: torch.Tensor):
    """Split examples into select/confirm halves BY DOCUMENT (no doc spans both),
    so the best signal is chosen on one half and the verdict read on the other —
    kills the 3-signal cherry-pick. Even-ranked docs -> select, odd -> confirm."""
    uniq = torch.unique(doc)
    sel_docs = set(uniq[::2].tolist())
    sel = torch.tensor([int(d) in sel_docs for d in doc])
    return sel, ~sel


CLEAN = ("prediction_norm", "dropout_agreement")  # inference-time-honest signals


def _analyze(fin: dict, label: str) -> dict:
    """Per-signal tercile+AUC+coverage-curve, then the pre-registered gate:
    pick the best CLEAN signal by AUC on the select split, read precision@
    coverage on the confirm split."""
    correct = fin["correct"]
    out = {
        "label": label,
        "n_pairs": fin["n"],
        "dup_dropped": fin["dropped"],
        "base_correct": round(float(correct.float().mean()), 3),
        "abs_cos_mean": round(float(fin["abs_cos"].mean()), 3),
        "abs_cos_p50": round(float(fin["abs_cos"].median()), 3),
    }
    for name, sig in fin["signals"].items():
        out[name] = {
            **tercile_report(sig, correct),
            "coverage_curve": coverage_curve(sig, correct),
            "clean": name in CLEAN,
        }
    sel, conf = _doc_split(fin["doc"])
    # choose the best clean signal on the SELECT split, judge on CONFIRM
    sel_aucs = {}
    for name in CLEAN:
        a = rank_auc(fin["signals"][name][sel], correct[sel])
        if a is not None:
            sel_aucs[name] = a
    if not sel_aucs or conf.sum() < GATE_MIN_BIN:
        out["gate"] = {"verdict": "INSUFFICIENT", "reason": "too few confirm-split examples"}
        return out
    best = max(sel_aucs, key=sel_aucs.get)
    pac = coverage_curve(fin["signals"][best][conf], correct[conf], fractions=(GATE_COVERAGE,))[0]
    passed = pac["acc"] >= GATE_MIN_ACC and pac["lift"] >= GATE_MIN_LIFT and pac["n"] >= GATE_MIN_BIN
    out["gate"] = {
        "verdict": "PASS" if passed else "FAIL",
        "chosen_signal": best,
        "selected_on_auc": round(sel_aucs[best], 3),
        "confirm_precision_at_coverage": pac,
        "thresholds": {
            "coverage": GATE_COVERAGE,
            "min_acc": GATE_MIN_ACC,
            "min_lift": GATE_MIN_LIFT,
            "min_bin": GATE_MIN_BIN,
        },
    }
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
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument(
        "--skip-docs",
        type=int,
        default=0,
        help="stream past the first N docs — use to avoid the predictor's TRAIN "
        "range when probing on the same dataset it was trained on (Fable review: "
        "'held-out' must be enforced, not asserted)",
    )
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-dropout", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    unit = args.unit or ("line" if args.corpus == "cot" else "sentence")
    max_span, max_sents = (96, 16) if args.corpus == "cot" else (64, 24)
    min_sents = args.window + 1
    if args.smoke:
        args.model_name, args.repo, args.n_docs = "Qwen/Qwen2.5-0.5B", None, 60
        args.window, max_span, max_sents, min_sents = 4, 24, 12, 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _preflight(device, args.window)  # crash device/shape bugs in ~$0.01, not post-encode

    from marker.run_stage2 import _doc_texts, _load_stage1, _smoke_cot_texts, encode_corpus

    whitener, wtag = _load_whitener(None if args.smoke else args.artifacts_repo, args.subdir)
    print(f"whitener: {wtag}", flush=True)
    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )

    if args.smoke:
        model = NextThoughtPredictor(d=gist.shape[-1], k=8, d_model=48, layers=2, heads=4)
        docs = _smoke_cot_texts(args.n_docs)
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(repo_id=args.artifacts_repo, filename=f"{args.subdir}/predictor.pt")
        model = _predictor_from_state(torch.load(path, map_location="cpu"), args.heads)
        # stream past the training range, then take n_docs
        gen = _doc_texts(
            args.n_docs + args.skip_docs,
            args.corpus,
            args.dataset,
            args.dataset_config,
            args.text_field,
        )
        docs = list(gen)[args.skip_docs :]
    model = model.to(device)

    seqs = encode_corpus(pm, gist, tok, docs, max_span, max_sents, min_sents, device, unit)
    print(f"encoded {len(seqs)} held-out chains (skip_docs={args.skip_docs})", flush=True)
    if len(seqs) < 60:
        print(
            f"WARNING: only {len(seqs)} chains kept — GSM8K solutions are often "
            f"< window+1={min_sents} lines; AUC/precision will be noisy. Raise --n-docs.",
            flush=True,
        )

    tf = _analyze(_finalize(collect(model, seqs, args.window, device, args.n_dropout, whitener)),
                  "teacher_forced")
    drift_raw = collect_drift(model, seqs, args.window, device, args.n_dropout, whitener)
    drift = _analyze(_finalize(drift_raw), "one_step_drift") if drift_raw else None

    manifest = {
        "probe_dataset": args.dataset,
        "subdir": args.subdir,
        "whitener": wtag,
        "window": args.window,
        "skip_docs": args.skip_docs,
        "teacher_forced": tf,
        "one_step_drift": drift,
    }
    tfg = tf["gate"]
    print(
        f"\n[CONFIDENCE PROBE] teacher_forced gate={tfg['verdict']} "
        f"signal={tfg.get('chosen_signal')} "
        f"p@{GATE_COVERAGE}={tfg.get('confirm_precision_at_coverage')}",
        flush=True,
    )
    if drift:
        print(f"[CONFIDENCE PROBE] one_step_drift gate={drift['gate']['verdict']}", flush=True)

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
    """GRAD_OK-style fail-loud: run the FULL collect+finalize+analyze path
    (teacher-forced AND drift) on a throwaway tiny predictor + random chains on
    `device` before the expensive encode, so a GPU-only device/shape bug crashes
    in minute 1 (twice-bitten: cpu/cuda gather, to_leaf_param) not after a
    60-min encode."""
    from marker.whiten import IdentityWhitener  # noqa: PLC0415

    k, d = 8, 16
    m = NextThoughtPredictor(d=d, k=k, d_model=16, layers=1, heads=2).to(device)
    seqs = [torch.randn(window + 2, k, d) for _ in range(10)]
    wh = IdentityWhitener()
    tf = _analyze(_finalize(collect(m, seqs, window, device, 3, wh)), "tf")
    dr = collect_drift(m, seqs, window, device, 3, wh)
    assert tf["n_pairs"] > 0, f"preflight incomplete: {tf}"
    assert window < 3 or dr is not None, "drift pass missing"
    print(f"PREFLIGHT_OK tf_gate={tf['gate']['verdict']} (n={tf['n_pairs']})", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
