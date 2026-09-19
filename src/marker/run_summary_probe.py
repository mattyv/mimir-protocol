"""SUMMARY-CONTENT PROBE (Fable-vetted spec): does a step's arithmetic
OPERATOR survive in the 8x3584 gist SUMMARY -- the final-layer readout the
Stage-2 predictor consumes and emits -- clean, under noise the bridge/render
converter would ignore (ratio 0.5), under predictor-sized noise (ratio 1.0),
and in the predictor's ACTUAL output? Paired, doc-disjoint, linear probe.

Every other probe in this repo reads structure back out through a RENDER
(words) or an injected KV cache. This one skips both: it asks the question
directly in vector space with a small in-torch linear classifier (multinomial
logistic regression, no sklearn) over the summary itself. Five conditions,
same (doc, n) pairing throughout:

    clean      summ[n]                              -- the encoder's own output
    noised_05  noised(summ[n], 0.5)                  -- converter-scale noise
    noised_10  noised(summ[n], 1.0)                  -- predictor-scale noise
    pred       predict_step(true history) -> ĝ[n]    -- the predictor's guess
    hist       summ[n-1]                             -- P_hist baseline feature

Fits (P_clean, P_pred, P_noised_10, P_hist) are doc-disjoint 80/20 splits,
scored with a `shallow` (position/length only) and `shuffled` (label-permuted)
control, then read through summary_verdict -- see summaryprobe.py for all of
the pure logic (label extraction, splitting, the normalize->standardize->PCA
feature pipeline, the probe itself, Wilson CI, and the gate read).

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_summary_probe \\
        --repo mattyvee/mimir-artifacts --artifacts-repo mattyvee/mimir-artifacts \\
        --subdir stage2_cot_openr1 --out-repo mattyvee/mimir-artifacts
Smoke: PYTHONPATH=src python -m marker.run_summary_probe --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from marker.summaryprobe import (
    OP_CLASSES,
    doc_disjoint_split,
    encode_labels,
    evaluate_probe,
    majority_rate,
    normalize_flatten,
    op_label,
    pca_apply,
    pca_fit,
    shuffle_labels_train,
    standardize_apply,
    standardize_fit,
    summary_verdict,
    train_probe,
)

# ── model-free plumbing (unit-tested without loading anything) ──────────────


def _smoke_varied_cot_texts(n: int) -> list[str]:
    """n synthetic multi-step traces for --smoke, with the OPERATOR ORDER
    permuted per doc (doc i uses permutation i % 24 of +,-,*,/). run_stage2's
    _smoke_cot_texts uses ONE fixed template, so there op = f(step position)
    exactly, the `shallow` (position/length) control scores 1.0, chance_p
    pins at 1.0 and the verdict is RED by construction -- the GREEN/YELLOW
    branch logic never runs. Permuting the order per doc breaks
    op-determined-by-position, so the smoke exercises the real verdict path
    (the smoke test asserts shallow < 1.0). Operands are picked so every
    relation is exact positive-integer arithmetic."""
    perms = list(itertools.permutations(("+", "-", "*", "/")))
    out = []
    for i in range(n):
        lines = [f"Start with {3 + i} items."]
        c = 3 + i
        for j, op in enumerate(perms[i % len(perms)]):
            a, b = 3 + ((i + 2 * j) % 7), 2 + ((3 * i + j) % 5)
            if op == "+":
                c = a + b
            elif op == "-":
                a, c = a + b, a
            elif op == "*":
                c = a * b
            else:
                a, c = a * b, a
            lines.append(f"Then compute {a} {op} {b} = {c}.")
        lines.append(f"#### {c}")
        out.append("\n".join(lines))
    return out


def _items_from_docs(docs: list[list[tuple[str, list[int]]]]) -> tuple[list[dict], int]:
    """docs: per-doc step lists [(text, ids), ...] -> (items, n_dropped).
    items: one {"doc", "n", "op", "n_tokens"} dict per step n>=1 with an
    extractable relation -- predict_step needs >=1 prior step, so n=0 is
    unscorable for EVERY condition and never appears here. Every condition
    the harness builds walks this SAME list in this SAME order (see
    _condition_features), so clean/noised/pred/hist stay paired on (doc, n)
    by construction, not by a later join. Steps with no relation are dropped
    and counted (n_dropped), never coerced to a fake label."""
    items: list[dict] = []
    n_dropped = 0
    for di, doc in enumerate(docs):
        for n in range(1, len(doc)):
            text, ids = doc[n]
            op = op_label(text)
            if op is None:
                n_dropped += 1
                continue
            items.append({"doc": di, "n": n, "op": op, "n_tokens": len(ids)})
    return items, n_dropped


def _shallow_features(items: list[dict]) -> torch.Tensor:
    """[one-hot min(n,7) (8 dims), n_tokens] per item -- a cheap position/
    length baseline with NO gist content at all. Beating it (chance_p in
    summary_verdict) means a probe is reading the SUMMARY, not just step
    position or how long the step's text was."""
    feats = []
    for it in items:
        oh = [0.0] * 8
        oh[min(it["n"], 7)] = 1.0
        feats.append([*oh, float(it["n_tokens"])])
    return torch.tensor(feats)


def _condition_features(items, summs, predictor, window, noise_gen05, noise_gen10):  # noqa: ANN001
    """The five paired conditions, each a [N, k, d] float32 tensor in items'
    own order -- built from ONE loop over `items`, so every condition shares
    the exact same (doc, n) pairing by construction. `summs` is stored fp16
    (see _encode_single_span); each doc's slice is upcast to float32 here
    before it touches the predictor or noised() -- a Linear layer's fp32
    weights against an fp16 input hard-crashes ("expected ... same dtype",
    the same trap test_vector_builder.py already hits elsewhere in this
    repo), so the upcast happens once per doc, not left to chance downstream.
    Devices: `summs` lives on CPU (_encode_single_span stores it there) while
    the predictor may sit on CUDA -- predict_step does NO device move of its
    own, so the doc's summaries are shuttled to the predictor's device for
    the forward and the result brought back, keeping every returned condition
    tensor on CPU (train_probe's nn.Linear and the labels are CPU; a CUDA
    feature tensor would crash the first probe fit on the real GPU run)."""
    from marker.run_bridge import noised, predict_step  # noqa: PLC0415

    pdev = next(predictor.parameters()).device
    clean, n05, n10, pred, hist = [], [], [], [], []
    for it in items:
        summ, n = summs[it["doc"]].float(), it["n"]
        clean.append(summ[n])
        n05.append(noised(summ[n], 0.5, noise_gen05))
        n10.append(noised(summ[n], 1.0, noise_gen10))
        pred.append(predict_step(predictor, summ.to(pdev), n, window).float().cpu())
        hist.append(summ[n - 1])
    return {
        "clean": torch.stack(clean),
        "noised_05": torch.stack(n05),
        "noised_10": torch.stack(n10),
        "pred": torch.stack(pred),
        "hist": torch.stack(hist),
    }


def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean whole-summary cosine similarity between two [N, k, d] batches (the
    k*d gist flattened per item, not per-slot)."""
    af, bf = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return round(float(F.cosine_similarity(af, bf, dim=-1).mean()), 4)


def _split_indices(items: list[dict], seed: int = 0):
    """Doc-disjoint 80/20 train/test, then a doc-disjoint 10% val slice carved
    out of the 80% train side (for early stopping) -- one split shared by
    every probe in this run, so P_clean/P_pred/P_noised_10/P_hist are all
    scored on the SAME held-out (doc, n) pairs."""
    doc_ids = [it["doc"] for it in items]
    train_docs, test_docs = doc_disjoint_split(doc_ids, 0.2, seed)
    fit_docs, val_docs = doc_disjoint_split(sorted(train_docs), 0.1, seed)
    idx_fit = [i for i, it in enumerate(items) if it["doc"] in fit_docs]
    idx_val = [i for i, it in enumerate(items) if it["doc"] in val_docs]
    idx_test = [i for i, it in enumerate(items) if it["doc"] in test_docs]
    assert idx_fit and idx_val and idx_test, (
        f"empty split (fit={len(idx_fit)} val={len(idx_val)} test={len(idx_test)}) "
        "-- too few docs for a doc-disjoint 80/10/10 split"
    )
    return idx_fit, idx_val, idx_test, sorted(train_docs), sorted(test_docs), sorted(val_docs)


def _prepare(feat_all, cond, idx_fit, idx_val, n_components):  # noqa: ANN001
    """normalize -> standardize(TRAIN) -> PCA(TRAIN) for one condition's train
    split. Returns the transformed train/val features plus the fitted
    (mean, std, pca_mean, components) -- _transform reuses these verbatim for
    every other condition's test split, never refitting."""
    xtr = normalize_flatten(feat_all[cond][idx_fit])
    xv = normalize_flatten(feat_all[cond][idx_val])
    mean, std = standardize_fit(xtr)
    xtr_s, xv_s = standardize_apply(xtr, mean, std), standardize_apply(xv, mean, std)
    pmean, comp = pca_fit(xtr_s, n_components)
    return pca_apply(xtr_s, pmean, comp), pca_apply(xv_s, pmean, comp), (mean, std, pmean, comp)


def _transform(feat_all, cond, idx, params):  # noqa: ANN001
    """Apply a train-fit (mean, std, pca_mean, components) to another
    condition's rows -- the TEST-uses-TRAIN-transform invariant."""
    mean, std, pmean, comp = params
    x = standardize_apply(normalize_flatten(feat_all[cond][idx]), mean, std)
    return pca_apply(x, pmean, comp)


def _fit_and_score_shallow(items, y, idx_fit, idx_val, idx_test, max_steps, patience, seed=0):  # noqa: ANN001
    """The `shallow` control: standardize(TRAIN)-only (no PCA -- the feature
    vector is already tiny), fit on train, scored on test."""
    feats = _shallow_features(items)
    mean, std = standardize_fit(feats[idx_fit])
    xtr = standardize_apply(feats[idx_fit], mean, std)
    xv = standardize_apply(feats[idx_val], mean, std)
    xt = standardize_apply(feats[idx_test], mean, std)
    model = train_probe(
        xtr,
        y[idx_fit],
        xv,
        y[idx_val],
        len(OP_CLASSES),
        max_steps=max_steps,
        patience=patience,
        seed=seed,
    )
    return evaluate_probe(model, xt, y[idx_test])


def _run_probes(items, y, feat_all, n_components=128, max_steps=2000, patience=200, seed=0):  # noqa: ANN001
    """Every fit + score cell the spec calls for, over one shared doc-disjoint
    split. Pure tensor-in, dict-out -- no model, no I/O -- so it's directly
    unit-testable with synthetic features."""
    idx_fit, idx_val, idx_test, train_docs, test_docs, val_docs = _split_indices(items, seed)

    xtr_p, xv_p, params_clean = _prepare(feat_all, "clean", idx_fit, idx_val, n_components)
    model_clean = train_probe(
        xtr_p,
        y[idx_fit],
        xv_p,
        y[idx_val],
        len(OP_CLASSES),
        max_steps=max_steps,
        patience=patience,
        seed=seed,
    )
    scores_clean = {
        c: evaluate_probe(model_clean, _transform(feat_all, c, idx_test, params_clean), y[idx_test])
        for c in ("clean", "noised_05", "noised_10", "pred")
    }

    # The control's VAL labels are permuted too: val is carved out of the
    # train side, and early-stopping the shuffled model on TRUE val labels
    # would let checkpoint selection chase real label signal -- a
    # true-label channel into the very control that exists to certify the
    # pipeline carries none (it inflates `shuffled` and can fake INVALID).
    y_shuf = shuffle_labels_train(y[idx_fit], seed=seed)
    y_shuf_val = shuffle_labels_train(y[idx_val], seed=seed + 100)
    model_shuf = train_probe(
        xtr_p,
        y_shuf,
        xv_p,
        y_shuf_val,
        len(OP_CLASSES),
        max_steps=max_steps,
        patience=patience,
        seed=seed + 1,
    )
    score_shuffled = evaluate_probe(
        model_shuf, _transform(feat_all, "clean", idx_test, params_clean), y[idx_test]
    )

    xtr_pp, xv_pp, params_pred = _prepare(feat_all, "pred", idx_fit, idx_val, n_components)
    model_pred = train_probe(
        xtr_pp,
        y[idx_fit],
        xv_pp,
        y[idx_val],
        len(OP_CLASSES),
        max_steps=max_steps,
        patience=patience,
        seed=seed,
    )
    score_pred = evaluate_probe(
        model_pred, _transform(feat_all, "pred", idx_test, params_pred), y[idx_test]
    )

    xtr_n, xv_n, params_n10 = _prepare(feat_all, "noised_10", idx_fit, idx_val, n_components)
    model_n10 = train_probe(
        xtr_n,
        y[idx_fit],
        xv_n,
        y[idx_val],
        len(OP_CLASSES),
        max_steps=max_steps,
        patience=patience,
        seed=seed,
    )
    score_n10 = evaluate_probe(
        model_n10, _transform(feat_all, "noised_10", idx_test, params_n10), y[idx_test]
    )

    xtr_h, xv_h, params_hist = _prepare(feat_all, "hist", idx_fit, idx_val, n_components)
    model_hist = train_probe(
        xtr_h,
        y[idx_fit],
        xv_h,
        y[idx_val],
        len(OP_CLASSES),
        max_steps=max_steps,
        patience=patience,
        seed=seed,
    )
    score_hist = evaluate_probe(
        model_hist, _transform(feat_all, "hist", idx_test, params_hist), y[idx_test]
    )

    score_shallow = _fit_and_score_shallow(
        items, y, idx_fit, idx_val, idx_test, max_steps, patience, seed
    )
    majority_acc = majority_rate(y[idx_test])

    cells = {
        "majority": majority_acc,
        "shallow": score_shallow["acc"],
        "shuffled": score_shuffled["acc"],
        "clean": scores_clean["clean"]["acc"],
        "pred_from_clean": scores_clean["pred"]["acc"],
        "pred_from_pred": score_pred["acc"],
        "hist": score_hist["acc"],
    }
    details = {
        "P_clean": scores_clean,
        "P_pred": score_pred,
        "P_noised_10": score_n10,
        "P_hist": score_hist,
        "shallow": score_shallow,
        "shuffled": score_shuffled,
        "majority": {"acc": majority_acc, "n": len(idx_test)},
        "split": {"train_docs": train_docs, "test_docs": test_docs, "val_docs": val_docs},
    }
    return cells, details


def _assert_window_matches(window: int, remote_manifest: dict) -> None:
    """Fail loud if --window doesn't match the predictor's OWN training
    window (stage2_cot_openr1/manifest.json) -- predict_step's windowing only
    stays in-distribution (sentence-position embeddings) when they agree."""
    remote_window = remote_manifest.get("window")
    assert remote_window == window, (
        f"--window {window} != predictor's trained window {remote_window!r} "
        "(read from the predictor's own manifest.json) -- sentence-position "
        "embeddings would run out-of-distribution"
    )


def _write_cache_shards(docs, summs, out_dir, shard_size=100):  # noqa: ANN001
    """docs (per-doc [(text, ids), ...]) + summs (matching [n_steps, k, d]
    tensors, fp16 as _encode_single_span stores them) -> fp16 safetensors
    shards + one index.json row per step: {doc, n, text, shard, row} --
    enough to look up any step's vector later without re-encoding. `.half()`
    here is idempotent, so this also accepts fp32 input safely."""
    from safetensors.torch import save_file  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for shard_id, start in enumerate(range(0, len(docs), shard_size)):
        chunk_docs = docs[start : start + shard_size]
        chunk_summs = summs[start : start + shard_size]
        shard_name = f"shard_{shard_id:04d}.safetensors"
        row = 0
        for local_i, doc in enumerate(chunk_docs):
            for n, (text, _ids) in enumerate(doc):
                index.append(
                    {"doc": start + local_i, "n": n, "text": text, "shard": shard_name, "row": row}
                )
                row += 1
        cat = torch.cat(chunk_summs, dim=0).half().contiguous()
        save_file({"summ": cat}, str(out / shard_name))
    (out / "index.json").write_text(json.dumps(index))
    return out


# ── model-touching helpers ───────────────────────────────────────────────────


@torch.no_grad()
def _encode_single_span(pm, gist, ids_list):  # noqa: ANN001
    """[step ids, ...] -> [n_steps, k, d] fp16 on CPU: ONE forward per step --
    the predictor's own training distribution (per-doc batching right-pads
    shorter spans and shifts their gist positions; see
    _sanity_cos_single_vs_batched). fp16 storage halves the host-RAM
    footprint of the ~1000-doc corpus; callers upcast to float32 on read
    (see _condition_features) before any arithmetic that needs it."""
    from marker.gist_model import encode_gist  # noqa: PLC0415

    return torch.stack([encode_gist(pm, gist, [ids]).float()[0] for ids in ids_list]).half().cpu()


@torch.no_grad()
def _sanity_cos_single_vs_batched(pm, gist, docs, summs, target_n=50):
    """On up to `target_n` steps, compare the single-span encode (stored) to
    the per-doc BATCHED encode (encode_gist on the whole doc's spans at once,
    the shape run_predprobe.py uses) -- batching right-pads to the doc's
    longest span, so a shorter step's gist lands at a shifted position. Mean
    cosine between the two -- the manifest's `cos_single_vs_batched` sanity
    cell."""
    from marker.gist_model import encode_gist  # noqa: PLC0415

    cos_vals = []
    for di, doc in enumerate(docs):
        if len(cos_vals) >= target_n:
            break
        ids_list = [ids for _, ids in doc]
        # .cpu(): the encode runs on the model's device (CUDA on the real
        # run) but `summs` is stored on CPU -- cosine across devices crashes
        batched = encode_gist(pm, gist, ids_list).float().cpu()  # [n_steps, k, d]
        single = summs[di].float()  # stored fp16 -- upcast before comparing
        for s in range(batched.shape[0]):
            if len(cos_vals) >= target_n:
                break
            a, b = single[s].reshape(1, -1), batched[s].reshape(1, -1)
            cos_vals.append(float(F.cosine_similarity(a, b)))
    return round(sum(cos_vals) / len(cos_vals), 4) if cos_vals else None


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None, help="Stage-1 gist adapter repo")
    ap.add_argument("--artifacts-repo", default=None, help="HF repo with predictor.pt")
    ap.add_argument("--subdir", default="stage2_cot_openr1", help="predictor.pt subdir")
    ap.add_argument("--out-repo", default=None, help="push the summary cache + probe manifest here")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--max-span", type=int, default=96)
    ap.add_argument(
        "--window",
        type=int,
        default=8,
        help="predictor input window: MUST match the predictor's training window "
        "(stage2 default 8) so sentence-position embeddings stay in-distribution",
    )
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pca-components", type=int, default=128)
    ap.add_argument("--cache-shard-size", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo = "Qwen/Qwen2.5-0.5B", None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.predictor import NextThoughtPredictor  # noqa: PLC0415
    from marker.reason_check import split_solution_steps  # noqa: PLC0415
    from marker.run_confidence import _predictor_from_state  # noqa: PLC0415
    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    k = gist.shape[0]

    # ── predictor (stage2_cot_openr1) -- no whitener: that subdir's is an
    # identity stub, so raw .float() summaries feed the probe directly ───────
    if args.smoke:
        predictor = NextThoughtPredictor(d=gist.shape[-1], k=k, d_model=48, layers=2, heads=4)
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        state = torch.load(
            hf_hub_download(args.artifacts_repo, f"{args.subdir}/predictor.pt"),
            map_location="cpu",
        )
        predictor = _predictor_from_state(state, args.heads)
        remote_manifest = json.loads(
            Path(hf_hub_download(args.artifacts_repo, f"{args.subdir}/manifest.json")).read_text()
        )
        _assert_window_matches(args.window, remote_manifest)
    predictor = predictor.to(device).eval()

    # ── data: GSM8K test, streaming; keep ALL docs with >=3 steps ────────────
    if args.smoke:
        texts = _smoke_varied_cot_texts(20)
    else:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(args.dataset, "main", split="test", streaming=True)
        texts = [row["answer"] for row in ds]

    docs = []
    for t in texts:
        steps = split_solution_steps(t)
        if len(steps) < 3:
            continue
        doc = []
        for s in steps:
            ids = tok(s, add_special_tokens=False).input_ids[: args.max_span]
            if len(ids) >= 2:
                doc.append((s, ids))
        if len(doc) >= 3:
            docs.append(doc)
    print(f"kept {len(docs)} docs (>=3 steps) of {len(texts)} texts", flush=True)

    # ── encode: single-span per step, fp16 on CPU per doc ────────────────────
    summs = [_encode_single_span(pm, gist, [ids for _, ids in doc]) for doc in docs]
    cos_single_vs_batched = _sanity_cos_single_vs_batched(pm, gist, docs, summs, target_n=50)
    if cos_single_vs_batched is not None and cos_single_vs_batched < 0.99:
        # Report loudly, do NOT abort: the probe and the pushed cache use
        # single-span encodes throughout, so the run's own numbers are
        # internally consistent -- but a low value means the encoder's output
        # for the SAME step shifts under per-doc batch padding, which is
        # itself a finding about encode stability and a caveat when comparing
        # against harnesses that batch per doc (e.g. run_predprobe).
        print(
            f"[SUMPROBE WARNING] cos_single_vs_batched={cos_single_vs_batched} < 0.99 -- "
            "single-span vs per-doc-batched encodes of the same step disagree; the probe "
            "stays valid (single-span throughout) but cross-harness comparisons are not "
            "apples-to-apples.",
            flush=True,
        )

    # ── labels + the shared (doc, n) item list ───────────────────────────────
    items, n_dropped_no_relation = _items_from_docs(docs)
    y = encode_labels([it["op"] for it in items])

    # ── the five paired conditions ────────────────────────────────────────
    noise_gen05 = torch.Generator().manual_seed(5)
    noise_gen10 = torch.Generator().manual_seed(10)
    feat_all = _condition_features(items, summs, predictor, args.window, noise_gen05, noise_gen10)
    cosines = {
        "pred_vs_clean": _mean_cos(feat_all["pred"], feat_all["clean"]),
        "pred_vs_hist": _mean_cos(feat_all["pred"], feat_all["hist"]),
        "noised_05_vs_clean": _mean_cos(feat_all["noised_05"], feat_all["clean"]),
        "noised_10_vs_clean": _mean_cos(feat_all["noised_10"], feat_all["clean"]),
    }

    max_steps = 300 if args.smoke else 2000
    patience = 60 if args.smoke else 200
    cells, details = _run_probes(
        items, y, feat_all, n_components=args.pca_components, max_steps=max_steps, patience=patience
    )
    verdict = summary_verdict(cells)

    manifest = {
        "n_docs": len(docs),
        "window": args.window,
        "n_items": len(items),
        "n_dropped_no_relation": n_dropped_no_relation,
        "cos_single_vs_batched": cos_single_vs_batched,
        "cosines": cosines,
        "cells": cells,
        **details,
        "verdict": verdict,
    }
    print(f"[SUMPROBE MANIFEST] {json.dumps(manifest)}", flush=True)  # single-line, survives tail

    if not args.smoke and args.out_repo:
        from marker.run_render import _push_with_retry  # noqa: PLC0415

        cache_dir = Path("/tmp/summary_cache_out")  # noqa: S108
        _write_cache_shards(docs, summs, cache_dir, shard_size=args.cache_shard_size)
        _push_with_retry(args.out_repo, str(cache_dir), "summary_cache_gsm8k_test")
        print(f"pushed summary cache to {args.out_repo}/summary_cache_gsm8k_test", flush=True)

        d = Path("/tmp/summary_probe_out")  # noqa: S108
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _push_with_retry(args.out_repo, str(d), "summary_probe")
        print(f"pushed summary probe manifest to {args.out_repo}/summary_probe", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
