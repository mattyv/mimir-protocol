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
    question_verdict,
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


def _smoke_questions(n: int) -> list[str]:
    """n synthetic question texts for `--smoke --with-question`, one per doc,
    paired 1:1 by index with `_smoke_varied_cot_texts`'s docs (the question
    condition needs SOME text per doc; its content doesn't matter mechanically
    -- only that it tokenizes to a real span)."""
    return [f"After all the steps, what is doc {i}'s final total?" for i in range(n)]


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


def _question_condition_features(items, summs, q_summs, pred_feat, window):  # noqa: ANN001
    """The four `--with-question` paired conditions, built from the SAME
    `items` loop/order as `_condition_features` -- so they stay paired to
    every other condition on (doc, n) by construction, same as the five
    original conditions:

        q         the doc's question gist                    -- label op_n
        q_hist    concat [question gist ; summ[n-1]]          -- label op_n
        q_pred    concat [question gist ; pred[n]]            -- label op_n
        fullhist  mean of summ[max(0,n-w+1) .. n-1]           -- label op_n

    `q_pred` reuses `pred_feat` (the harness's already-computed "pred"
    condition, item-for-item) rather than calling predict_step again --
    same guess, no duplicate predictor forward pass. `summs`/`q_summs` are
    upcast to float32 per doc, same reasoning as `_condition_features` (fp16
    storage vs. fp32 downstream arithmetic)."""
    q, q_hist, q_pred, fullhist = [], [], [], []
    for i, it in enumerate(items):
        summ, n = summs[it["doc"]].float(), it["n"]
        qg = q_summs[it["doc"]].float()
        q.append(qg)
        q_hist.append(torch.cat([qg, summ[n - 1]], dim=0))
        q_pred.append(torch.cat([qg, pred_feat[i]], dim=0))
        a = max(0, n - window + 1)
        fullhist.append(summ[a:n].mean(dim=0))
    return {
        "q": torch.stack(q),
        "q_hist": torch.stack(q_hist),
        "q_pred": torch.stack(q_pred),
        "fullhist": torch.stack(fullhist),
    }


def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean whole-summary cosine similarity between two [N, k, d] batches (the
    k*d gist flattened per item, not per-slot)."""
    af, bf = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return round(float(F.cosine_similarity(af, bf, dim=-1).mean()), 4)


def _mean_pairwise_cos(x: torch.Tensor, n_pairs: int = 2000, seed: int = 0) -> float:
    """Mean cosine similarity over `n_pairs` random (i, j) item pairs (i != j)
    within ONE [N, k, d] batch, sampled with a seeded generator -- the
    `pairwise_pred` / `pairwise_clean` manifest diagnostic. Settles whether a
    batch's vectors all collapse toward a similar direction regardless of
    which item they're for (e.g. the predictor copying/blending history
    rather than tracking the actual step) -- unlike `_mean_cos`, which compares
    two batches item-for-item, this compares a batch against ITSELF, item i vs
    a different item j."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    xf = x.reshape(n, -1)
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return 0.0
    return round(float(F.cosine_similarity(xf[i], xf[j], dim=-1).mean()), 4)


def _truncate_question_ids(ids: list[int], max_span: int) -> tuple[list[int], int]:
    """Cap a question's token ids at `max_span`, same as every step -- but
    keeping the LAST max_span tokens rather than the first: GSM8K questions
    put the actual ask ("how many X does he have now?") at the end, so
    front-truncating would cut it off. Returns (kept_ids, n_truncated) --
    n_truncated is how many leading tokens were dropped, 0 when the question
    already fit."""
    if len(ids) <= max_span:
        return ids, 0
    return ids[-max_span:], len(ids) - max_span


def _cos_by_n(
    a: torch.Tensor, b: torch.Tensor, items: list[dict], cap: int = 8
) -> dict[str, float]:
    """Mean whole-summary cosine between paired batches, binned by the item's
    step index n (n >= cap pooled as "cap+"). The window diagnostic: if the
    predictor's sentence-position rows past its trained window were never
    learned, pred-vs-clean cosine drops off a cliff at that n."""
    af, bf = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    cos = F.cosine_similarity(af, bf, dim=-1)
    bins: dict[str, list[float]] = {}
    for it, c in zip(items, cos.tolist(), strict=True):
        key = f"{cap}+" if it["n"] >= cap else str(it["n"])
        bins.setdefault(key, []).append(c)
    return {
        k: round(sum(v) / len(v), 4)
        for k, v in sorted(bins.items(), key=lambda kv: int(kv[0].rstrip("+")))
    }


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


def _run_question_probes(  # noqa: PLR0913
    items, y, feat_all, idx_fit, idx_val, idx_test, n_components=128, max_steps=2000, patience=200, seed=0
):  # noqa: ANN001
    """The four `--with-question` probes (P_q, P_q_hist, P_q_pred,
    P_fullhist) -- each trained + scored on its OWN condition (same
    normalize -> standardize(train) -> PCA(train) -> fit -> test-transform
    pipeline as P_pred/P_hist in `_run_probes`), over the SAME doc-disjoint
    split (idx_fit/idx_val/idx_test) every other probe in this run uses --
    passed in, never refit here, so these cells stay comparable to
    P_clean/P_pred/P_hist on the identical held-out (doc, n) pairs."""
    conds = {"q": "P_q", "q_hist": "P_q_hist", "q_pred": "P_q_pred", "fullhist": "P_fullhist"}
    cells, details = {}, {}
    for cond, detail_key in conds.items():
        xtr, xv, params = _prepare(feat_all, cond, idx_fit, idx_val, n_components)
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
        score = evaluate_probe(model, _transform(feat_all, cond, idx_test, params), y[idx_test])
        cells[cond] = score["acc"]
        details[detail_key] = score
    return cells, details


def _assert_window_matches(window: int, remote_manifest: dict) -> str:
    """Fail loud if --window doesn't match the predictor's OWN training
    window (stage2_cot_openr1/manifest.json) -- predict_step's windowing only
    stays in-distribution (sentence-position embeddings) when they agree.
    Older stage-2 manifests (the cot_openr1 one) never recorded the window at
    all: then we cannot check, so WARN loudly and run with --window exactly as
    every downstream harness (bridge, rollout, predprobe) did, so this probe's
    numbers stay comparable to theirs. Returns the window's provenance for the
    manifest; `pred_vs_clean_by_n` is the empirical check (a cliff at some n
    means positions past the trained window were never learned)."""
    if "window" not in remote_manifest:
        print(
            f"[SUMPROBE WARNING] predictor manifest has no 'window' key -- cannot verify; "
            f"using --window {window} as the downstream harnesses did (read "
            "cosines.pred_vs_clean_by_n for a cliff)",
            flush=True,
        )
        return "assumed (predictor manifest lacks 'window')"
    remote_window = remote_manifest["window"]
    assert remote_window == window, (
        f"--window {window} != predictor's trained window {remote_window!r} "
        "(read from the predictor's own manifest.json) -- sentence-position "
        "embeddings would run out-of-distribution"
    )
    return "manifest"


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


def _write_question_cache(questions: list[str], q_summs: torch.Tensor, out_dir) -> Path:  # noqa: ANN001
    """questions (one text per doc, in doc order) + q_summs [n_docs, k, d] (as
    `_encode_single_span` returns) -> `questions.safetensors` (fp16) +
    `questions_index.json` ({doc, text} rows), written into the SAME cache
    dir `_write_cache_shards` uses so one push carries both. `.half()` is
    idempotent, so this also accepts fp32 input safely."""
    from safetensors.torch import save_file  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"q": q_summs.half().contiguous()}, str(out / "questions.safetensors"))
    index = [{"doc": i, "text": t} for i, t in enumerate(questions)]
    (out / "questions_index.json").write_text(json.dumps(index))
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
    ap.add_argument(
        "--with-question",
        action="store_true",
        help="also encode each doc's GSM8K question and score the q/q_hist/q_pred/"
        "fullhist probes (Fable's pre-registered question-gist check). Default off "
        "leaves today's manifest bitwise unchanged.",
    )
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
    window_source = "smoke"
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
        window_source = _assert_window_matches(args.window, remote_manifest)
    predictor = predictor.to(device).eval()

    # ── data: GSM8K test, streaming; keep ALL docs with >=3 steps ────────────
    # `questions_src` rides alongside `texts` (same order, same length) so
    # each kept doc's question survives the filtering loop below paired to
    # it -- collected unconditionally (cheap, no model call) so --with-question
    # ON/OFF never changes which docs/steps are kept.
    if args.smoke:
        texts = _smoke_varied_cot_texts(20)
        questions_src = _smoke_questions(20)
    else:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(args.dataset, "main", split="test", streaming=True)
        texts, questions_src = [], []
        for row in ds:
            texts.append(row["answer"])
            questions_src.append(row.get("question", ""))

    docs, questions = [], []
    for t, q_text in zip(texts, questions_src, strict=True):
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
            questions.append(q_text)
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

    # ── --with-question: one question gist per doc, single-span like a step ──
    q_summs, n_questions_truncated = None, 0
    if args.with_question:
        q_ids_list = []
        for q_text in questions:
            raw_ids = tok(q_text, add_special_tokens=False).input_ids
            q_ids, n_trunc = _truncate_question_ids(raw_ids, args.max_span)
            assert len(q_ids) >= 1, f"empty question ids for {q_text!r}"
            q_ids_list.append(q_ids)
            n_questions_truncated += int(n_trunc > 0)
        q_summs = _encode_single_span(pm, gist, q_ids_list)  # [n_docs, k, d] fp16 CPU

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
        "pred_vs_clean_by_n": _cos_by_n(feat_all["pred"], feat_all["clean"], items),
        "noised_05_vs_clean": _mean_cos(feat_all["noised_05"], feat_all["clean"]),
        "noised_10_vs_clean": _mean_cos(feat_all["noised_10"], feat_all["clean"]),
    }

    # ── --with-question: the four extra paired conditions + diagnostics ─────
    if args.with_question:
        feat_all = {
            **feat_all,
            **_question_condition_features(items, summs, q_summs, feat_all["pred"], args.window),
        }
        cosines["clean_n_vs_clean_prev"] = _mean_cos(feat_all["clean"], feat_all["hist"])
        cosines["pairwise_pred"] = _mean_pairwise_cos(feat_all["pred"], n_pairs=2000, seed=0)
        cosines["pairwise_clean"] = _mean_pairwise_cos(feat_all["clean"], n_pairs=2000, seed=0)

    max_steps = 300 if args.smoke else 2000
    patience = 60 if args.smoke else 200
    cells, details = _run_probes(
        items, y, feat_all, n_components=args.pca_components, max_steps=max_steps, patience=patience
    )
    verdict = summary_verdict(cells)  # unaffected by --with-question: reads only the original cells

    if args.with_question:
        # SAME doc-disjoint split every other probe above used (same seed=0
        # default _run_probes used internally) -- P_q_hist stays comparable to
        # P_clean/P_pred/P_hist on the identical held-out (doc, n) pairs.
        idx_fit, idx_val, idx_test, *_ = _split_indices(items, seed=0)
        q_cells, q_details = _run_question_probes(
            items,
            y,
            feat_all,
            idx_fit,
            idx_val,
            idx_test,
            n_components=args.pca_components,
            max_steps=max_steps,
            patience=patience,
        )
        cells.update(q_cells)
        details.update(q_details)

    manifest = {
        "n_docs": len(docs),
        "window": args.window,
        "window_source": window_source,
        "n_items": len(items),
        "n_dropped_no_relation": n_dropped_no_relation,
        "cos_single_vs_batched": cos_single_vs_batched,
        "cosines": cosines,
        "cells": cells,
        **details,
        "verdict": verdict,
    }
    if args.with_question:
        manifest["question_verdict"] = question_verdict(cells)
        manifest["n_questions_truncated"] = n_questions_truncated
    print(f"[SUMPROBE MANIFEST] {json.dumps(manifest)}", flush=True)  # single-line, survives tail

    if not args.smoke and args.out_repo:
        from marker.run_render import _push_with_retry  # noqa: PLC0415

        cache_dir = Path("/tmp/summary_cache_out")  # noqa: S108
        _write_cache_shards(docs, summs, cache_dir, shard_size=args.cache_shard_size)
        if args.with_question:
            _write_question_cache(questions, q_summs, cache_dir)
        _push_with_retry(args.out_repo, str(cache_dir), "summary_cache_gsm8k_test")
        print(f"pushed summary cache to {args.out_repo}/summary_cache_gsm8k_test", flush=True)

        # --with-question pushes to a SEPARATE subdir -- summary_probe/ (the
        # no-question manifest) is never overwritten by a run that scored
        # different (extra) cells.
        manifest_subdir = "summary_probe_q" if args.with_question else "summary_probe"
        d = Path("/tmp/summary_probe_out")  # noqa: S108
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _push_with_retry(args.out_repo, str(d), manifest_subdir)
        print(f"pushed summary probe manifest to {args.out_repo}/{manifest_subdir}", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
