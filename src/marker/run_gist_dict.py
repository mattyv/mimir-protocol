"""Stage-1 gist dictionary fidelity harness (GIST_LM_PLAN.md stage 1; full
spec: scratchpad/gist_dict_stage1_spec.md). Snaps each step's canonical
gist-KV to the nearest entry of a small per-slot dictionary and asks: can the
frozen reader still write the step correctly, and is the operation still
readable from the 8 IDs?

Pipeline (one process, whichever of --eval/--diagnose are requested runs
against the SAME fit/eval encode -- --smoke always runs both):
  A. encode_canonical: one forward per step -> (per-layer gist-KV, readout),
     both at canonical positions [base, base+k) (see gist_model.gist_kv's
     `return_hidden` flag).
  B. fit/eval sets: GSM8K train + OpenR1 steps (fit, doc-disjoint from eval,
     stored to disk in per-slot shards); GSM8K test + a fresh synthetic
     template set (eval).
  C. dictionaries: k-means-based codebooks (gist_dict.py), one slot's shard
     loaded at a time.
  D. --eval: native / quantized / wrong_doc_quantized / random_ids conditions
     through the trained reader (render_adapter_oneform), reusing
     run_render's scoring.
  E. --diagnose: CPU-only reconstruction cosines, usage stats, naive-Bayes
     op-from-IDs, a probe on centroid readouts.
  F. manifest + (non-smoke) push.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_gist_dict \\
        --repo mattyvee/mimir-artifacts --out-repo mattyvee/mimir-artifacts \\
        --n-fit 40000 --n-eval 300 --eval --diagnose
Smoke (local tiny model):
    PYTHONPATH=src python -m marker.run_gist_dict --smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from marker.gist_dict import (
    adjusted_mutual_info,
    build_dict_kv,
    build_dict_kv_residual,
    build_dict_ro,
    build_dict_whole,
    detokenize,
    fit_categorical_nb,
    kv_slot_matrix,
    predict_categorical_nb,
    stage1_verdict,
    tokenize,
)
from marker.gist_model import gist_kv
from marker.gistprobe import per_token_ce
from marker.predprobe import pick_cross_doc_step
from marker.run_render import _score_record

BASE = 64  # canonical gist_start (GIST_LM_PLAN.md "Tokenizer")
MAX_SPAN = 64
MAX_NEW = 64  # render generation/scoring budget (spec section D)

# gate 0 (native placement) thresholds -- section D
GATE0_GSM8K_MIN = 0.90
GATE0_FRESH_MIN = 0.95


# ── A. canonical encode ──────────────────────────────────────────────────────


def encode_canonical(pm, gist, ids: list[int], base: int = BASE, max_span: int = MAX_SPAN):  # noqa: ANN001
    """One forward -> (kv, readout, cont_start) at canonical positions
    [base, base+k). Fails loud on an over-length step (never silently
    truncates it -- that would store a canonical KV for a step SHORTER than
    the one the caller thinks it labeled); callers filter+count drops
    themselves (see build_fit_stream/build_eval_set)."""
    assert len(ids) <= max_span, f"step has {len(ids)} tokens > max_span={max_span}"
    k_slots = gist.shape[0]
    kv, cont_start, _first_logits, readout = gist_kv(
        pm, gist, ids, gist_start=base, return_hidden=True
    )
    assert cont_start == base + k_slots
    return kv, readout.float(), cont_start


# ── B. fit / eval sets ───────────────────────────────────────────────────────


class EvalStep(NamedTuple):
    """One eval step: doc_i (unique per step by construction -- eval is built
    ONE scorable step per doc, see build_eval_set) + its token ids/text."""

    doc_i: int
    ids: list[int]
    text: str


def _gsm8k_docs(split: str, n: int, dataset="openai/gsm8k", smoke: bool = False):  # noqa: ANN001
    """Yields (doc_key, [step_text,...]) from GSM8K `split` -- or, under
    --smoke, from run_stage2's offline synthetic GSM8K-style fixture (never
    hits the network, matching every other run_*.py harness's --smoke path)."""
    from marker.reason_check import split_solution_steps  # noqa: PLC0415

    if smoke:
        from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

        for i, text in enumerate(_smoke_cot_texts(n)):
            yield (f"gsm8k_{split}_smoke", i), split_solution_steps(text)
        return

    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset, "main", split=split, streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield (f"gsm8k_{split}", i), split_solution_steps(row["answer"])


def _openr1_docs(n: int, dataset: str):
    """Yields (doc_key, [step_text,...]) from an OpenR1-style cot corpus,
    reusing run_stage2's generic corpus loader + line splitter (the same
    step-per-line convention run_bridge/run_render use for non-GSM8K cot
    corpora)."""
    from marker.run_stage2 import _doc_texts, _split_units  # noqa: PLC0415

    for i, text in enumerate(_doc_texts(n, "cot", dataset)):
        yield ("openr1", i), _split_units(text, "line")


def _fresh_docs(n: int):
    """Yields (doc_key, [step_text,...]) from the fresh synthetic template
    set (run_summary_probe's _smoke_varied_cot_texts: operator order permuted
    per doc, so op is never a pure function of step position)."""
    from marker.reason_check import split_solution_steps  # noqa: PLC0415
    from marker.run_summary_probe import _smoke_varied_cot_texts  # noqa: PLC0415

    for i, text in enumerate(_smoke_varied_cot_texts(n)):
        yield ("fresh", i), split_solution_steps(text)


def build_eval_set(doc_iter, tok, max_span: int, n_eval: int):  # noqa: ANN001
    """doc_iter yields (doc_key, [step_text,...]) -> (steps, doc_keys). ONE
    scorable step per doc (first step with an extractable op_label AND
    <= max_span tokens) -- eval steps are doc-disjoint from EACH OTHER by
    construction, so pick_wrong_doc_step's cross-doc pairing never needs a
    second step from the same doc."""
    from marker.summaryprobe import op_label  # noqa: PLC0415

    steps: list[EvalStep] = []
    doc_keys: set = set()
    for doc_key, step_texts in doc_iter:
        if len(steps) >= n_eval:
            break
        for text in step_texts:
            if op_label(text) is None:
                continue
            ids = tok(text, add_special_tokens=False).input_ids
            if len(ids) < 2 or len(ids) > max_span:
                continue
            steps.append(EvalStep(doc_i=len(steps), ids=ids, text=text))
            doc_keys.add(doc_key)
            break
    return steps, doc_keys


def build_fit_items(
    doc_iters, tok, max_span: int, n_fit: int, excluded_doc_keys: set, counters: dict
):  # noqa: ANN001
    """Consumes `doc_iters` (a list of doc-generators, e.g. [gsm8k_train,
    openr1]) in order -> list[(doc_key, ids, text)], doc-disjoint from
    `excluded_doc_keys` (the eval set's docs), up to n_fit steps. Steps over
    max_span are DROPPED and COUNTED in counters['too_long'] (fail loud: a
    caller reading counters sees exactly how many were dropped, never a
    silent truncation)."""
    counters.setdefault("too_long", 0)
    items: list[tuple] = []
    for doc_iter in doc_iters:
        if len(items) >= n_fit:
            break
        for doc_key, step_texts in doc_iter:
            if len(items) >= n_fit:
                break
            if doc_key in excluded_doc_keys:
                continue
            for text in step_texts:
                if len(items) >= n_fit:
                    break
                ids = tok(text, add_special_tokens=False).input_ids
                if len(ids) < 2:
                    continue
                if len(ids) > max_span:
                    counters["too_long"] += 1
                    continue
                items.append((doc_key, ids, text))
    return items


# ── per-slot shard I/O (never all 8 slots in RAM at once) ──────────────────


def write_fit_shards(step_iter, out_dir, d: int, dr: int, k_slots: int) -> int:  # noqa: ANN001
    """step_iter yields (kv, readout) ONE STEP AT A TIME -- never the whole
    fit set materialized. Streams each step's k_slots rows straight to
    k_slots raw scratch files (O(1) extra RAM per step), then converts each
    scratch file to `slot_{s}.safetensors` ONE SLOT AT A TIME, so at most one
    slot's full [N, D] tensor is ever resident (never all k_slots at once --
    the ~18GB-at-40k figure section F's push guard exists for). Returns N."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_paths = [out / f"_slot_{s}.raw" for s in range(k_slots)]
    ro_path = out / "_readouts.raw"
    handles = [open(p, "wb") for p in raw_paths]  # noqa: SIM115
    ro_handle = open(ro_path, "wb")  # noqa: SIM115
    n = 0
    try:
        for kv, readout in step_iter:
            mat = kv_slot_matrix(kv).half().contiguous()  # [k_slots, D]
            for s in range(k_slots):
                handles[s].write(mat[s].numpy().tobytes())
            ro_handle.write(readout.half().contiguous().numpy().tobytes())
            n += 1
    finally:
        for h in handles:
            h.close()
        ro_handle.close()

    from safetensors.torch import save_file  # noqa: PLC0415

    for s in range(k_slots):
        arr = np.fromfile(raw_paths[s], dtype=np.float16).reshape(n, d)
        save_file({"kv": torch.from_numpy(arr.copy())}, str(out / f"slot_{s}.safetensors"))
        raw_paths[s].unlink()
    ro_arr = np.fromfile(ro_path, dtype=np.float16).reshape(n, k_slots, dr)
    save_file({"readout": torch.from_numpy(ro_arr.copy())}, str(out / "readouts.safetensors"))
    ro_path.unlink()
    return n


def load_slot_shard(out_dir, s: int) -> torch.Tensor:  # noqa: ANN001
    """slot_{s}.safetensors -> [N, D] fp16. Reads ONLY this slot's file --
    the other slots' files need not even exist for this to work."""
    from safetensors.torch import load_file  # noqa: PLC0415

    return load_file(str(Path(out_dir) / f"slot_{s}.safetensors"))["kv"]


def load_readouts(out_dir) -> torch.Tensor:  # noqa: ANN001
    """readouts.safetensors -> [N, k_slots, Dr] fp16 (small; kept resident)."""
    from safetensors.torch import load_file  # noqa: PLC0415

    return load_file(str(Path(out_dir) / "readouts.safetensors"))["readout"]


# ── C. dictionaries ──────────────────────────────────────────────────────────


def build_all_dicts(
    shard_dir,  # noqa: ANN001
    k_slots: int,
    geometry: dict,
    ks: list[int],
    res_k1: int,
    res_k2: int,
    ro_k: int,
    whole_k: int,
    whole_proj_dim: int | None,
    seed: int = 0,
):
    """Builds every configured dictionary from the on-disk shards, loading
    at most one slot at a time (per gist_dict.build_dict_* + build_dict_whole
    contracts). Returns (dicts, fit_ids): dicts[cfg] is a tokenize/detokenize-
    ready dict; fit_ids[cfg] is [N, k_slots] long -- the fit set's OWN
    tokenized ids, a free byproduct of the k-means assignment (never
    recomputed by re-running tokenize over the whole fit set)."""
    readouts = load_readouts(shard_dir).float()  # [N, k_slots, Dr] -- small

    def _slot_ro(s):
        return readouts[:, s, :]

    dicts: dict[str, dict] = {}
    fit_ids: dict[str, torch.Tensor] = {}

    for K in ks:
        slots, assigns = [], []
        for s in range(k_slots):
            mat = load_slot_shard(shard_dir, s).float()
            entry, assign = build_dict_kv(mat, _slot_ro(s), K, seed=seed)
            slots.append(entry)
            assigns.append(assign)
            del mat
        name = f"kv_K{K}"
        dicts[name] = {"cfg": name, "kind": "kv", "geometry": geometry, "slots": slots}
        fit_ids[name] = torch.stack(assigns, dim=1)

    slots, assigns = [], []
    for s in range(k_slots):
        mat = load_slot_shard(shard_dir, s).float()
        entry, a1, a2 = build_dict_kv_residual(mat, _slot_ro(s), res_k1, res_k2, seed=seed)
        slots.append(entry)
        assigns.append(a1 * res_k2 + a2)
        del mat
    name = f"kv_res_{res_k1}x{res_k2}"
    dicts[name] = {"cfg": name, "kind": "kv_res", "geometry": geometry, "slots": slots}
    fit_ids[name] = torch.stack(assigns, dim=1)

    slots, assigns = [], []
    for s in range(k_slots):
        mat = load_slot_shard(shard_dir, s).float()
        entry, assign = build_dict_ro(_slot_ro(s), mat, ro_k, seed=seed)
        slots.append(entry)
        assigns.append(assign)
        del mat
    name = f"ro_K{ro_k}"
    dicts[name] = {"cfg": name, "kind": "ro", "geometry": geometry, "slots": slots}
    fit_ids[name] = torch.stack(assigns, dim=1)

    def _loader(s):
        return load_slot_shard(shard_dir, s).float()

    entry, assign = build_dict_whole(
        _loader, k_slots, whole_k, seed=seed, proj_dim=whole_proj_dim, proj_seed=seed
    )
    name = f"whole_K{whole_k}"
    dicts[name] = {"cfg": name, "kind": "whole", "geometry": geometry, "entry": entry}
    fit_ids[name] = assign.unsqueeze(1).expand(-1, k_slots).clone()

    return dicts, fit_ids


def save_dict(dict_: dict, out_dir) -> None:  # noqa: ANN001
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict_, out / f"dict_{dict_['cfg']}.pt")


# ── D. GPU eval: native / quantized / wrong_doc_quantized / random_ids ──────


def pick_wrong_doc_step(pool: list[EvalStep], gen: torch.Generator) -> list[EvalStep]:
    """For each step in `pool` (one per doc, see build_eval_set), the
    wrong-doc pairing: another step from a DIFFERENT doc
    (predprobe.pick_cross_doc_step) -- scored later against the CURRENT
    step's own text (the cheating floor)."""
    doc_lengths = [1] * len(pool)
    out = []
    for i in range(len(pool)):
        dj, _sj = pick_cross_doc_step(i, doc_lengths, gen, step_idx=0)
        out.append(pool[dj])
    return out


def random_ids(K: int, k_slots: int, gen: torch.Generator) -> list[int]:
    """k_slots uniform random ids in [0, K), seeded via `gen` -- the
    random_ids condition's floor."""
    return [int(torch.randint(0, K, (1,), generator=gen)) for _ in range(k_slots)]


def _condition_metrics(pm, tok, gold: EvalStep, kv, cs: int, nl_id, max_new: int = MAX_NEW) -> dict:  # noqa: ANN001
    """f1/num_recall/rel_exact (run_render._score_record, reused not copied)
    + teacher-forced render NLL of the TRUE step (gistprobe.per_token_ce) --
    the sensitive metric the spec calls out. `render` adapter must already
    be active."""
    stop_ids = {nl_id} if nl_id is not None else set()
    f1, nr, rel = _score_record(pm, tok, gold, kv, cs, False, stop_ids, max_new)
    tail = list(gold.ids) + ([nl_id] if nl_id is not None else [])
    ce, _tgt = per_token_ce(pm, kv, cs, [], tail)
    return {"f1": f1, "num_recall": nr, "rel": rel, "nll": float(ce.mean())}


def _summarize(metrics: list[dict]) -> dict:
    from marker.summaryprobe import wilson_ci  # noqa: PLC0415

    rels = [m["rel"] for m in metrics if m["rel"] is not None]
    nrs = [m["num_recall"] for m in metrics if m["num_recall"] is not None]
    k_exact = sum(1 for r in rels if r >= 0.999)
    lo, hi = wilson_ci(k_exact, len(rels)) if rels else (0.0, 1.0)
    return {
        "rel_exact": round(sum(rels) / len(rels), 4) if rels else 0.0,
        "rel_exact_ci": [round(lo, 4), round(hi, 4)],
        "f1_mean": round(sum(m["f1"] for m in metrics) / max(1, len(metrics)), 4),
        "num_recall": round(sum(nrs) / len(nrs), 4) if nrs else 0.0,
        "nll_mean": round(sum(m["nll"] for m in metrics) / max(1, len(metrics)), 4),
        "n": len(metrics),
    }


def _dict_k(dict_: dict) -> int:
    """The valid id range [0, K) for a config, however its entry is shaped:
    kv/ro store K directly; kv_res's id is the JOINT id1*K2+id2 (range
    K1*K2); whole stores K on its single shared entry."""
    if dict_["kind"] == "whole":
        return dict_["entry"]["K"]
    slot = dict_["slots"][0]
    if dict_["kind"] == "kv_res":
        return slot["K1"] * slot["K2"]
    return slot["K"]


@torch.no_grad()
def eval_config(
    pm,
    tok,
    gist,
    eval_steps: list[EvalStep],
    dict_: dict,
    geometry: dict,
    seed: int = 0,
    max_new: int = MAX_NEW,
):  # noqa: ANN001
    """The 4 conditions of section D on ONE eval set (gsm8k or fresh), for
    ONE dictionary config. Returns ({"native","quantized",
    "wrong_doc_quantized","random_ids"} -> summary, ids [N, k_slots])."""
    k_slots = geometry["k_slots"]
    K_for_random = _dict_k(dict_)
    wrong_pool = pick_wrong_doc_step(eval_steps, torch.Generator().manual_seed(seed))
    rand_gen = torch.Generator().manual_seed(seed + 1)
    nl_id = next((t for t in tok("\n", add_special_tokens=False).input_ids if t), None)

    pm.set_adapter("default")
    natives = [encode_canonical(pm, gist, s.ids, base=BASE) for s in eval_steps]
    wrong_natives = [encode_canonical(pm, gist, s.ids, base=BASE) for s in wrong_pool]

    metrics = {c: [] for c in ("native", "quantized", "wrong_doc_quantized", "random_ids")}
    ids_out = []
    for gold, (kv, readout, cs), (wkv, wreadout, _wcs) in zip(
        eval_steps, natives, wrong_natives, strict=True
    ):
        ids = tokenize(kv, dict_, readout=readout)
        ids_out.append(ids)
        wrong_ids = tokenize(wkv, dict_, readout=wreadout)
        rids = random_ids(K_for_random, k_slots, rand_gen)

        pm.set_adapter("render")
        metrics["native"].append(_condition_metrics(pm, tok, gold, kv, cs, nl_id, max_new))
        metrics["quantized"].append(
            _condition_metrics(pm, tok, gold, detokenize(ids, dict_, geometry), cs, nl_id, max_new)
        )
        metrics["wrong_doc_quantized"].append(
            _condition_metrics(
                pm, tok, gold, detokenize(wrong_ids, dict_, geometry), cs, nl_id, max_new
            )
        )
        metrics["random_ids"].append(
            _condition_metrics(pm, tok, gold, detokenize(rids, dict_, geometry), cs, nl_id, max_new)
        )
        pm.set_adapter("default")

    return {c: _summarize(v) for c, v in metrics.items()}, torch.tensor(ids_out, dtype=torch.long)


def gate0_pass(native_gsm8k: dict, native_fresh: dict) -> bool:
    """Gate 0 (section D): canonical placement must already be readable by
    the trained reader BEFORE any dictionary is involved -- else nothing
    downstream is trustworthy."""
    return (
        native_gsm8k["rel_exact"] >= GATE0_GSM8K_MIN
        and native_fresh["rel_exact"] >= GATE0_FRESH_MIN
    )


def compute_r(rel_quantized: float, rel_wrong: float, rel_native: float) -> float:
    """R = (quantized - wrong) / (native - wrong) -- the margin over the
    wrong-thought floor (never the raw absolute rel_exact, which the ledger-
    free scorer already flatters uniformly across every condition)."""
    denom = rel_native - rel_wrong
    if denom <= 0:
        return 0.0
    return (rel_quantized - rel_wrong) / denom


# ── E. CPU diagnostics ───────────────────────────────────────────────────────


def usage_diagnostics(usage: torch.Tensor) -> dict:
    """Usage entropy / log(K) in [0,1] (1.0 = uniform use) + dead-entry
    count (usage==0)."""
    k = usage.numel()
    total = int(usage.sum())
    if total == 0 or k <= 1:
        return {"entropy_norm": 0.0, "dead_entries": int((usage == 0).sum())}
    p = usage.float() / total
    p = p[p > 0]
    h = float(-(p * p.log()).sum())
    return {
        "entropy_norm": round(h / torch.log(torch.tensor(float(k))).item(), 4),
        "dead_entries": int((usage == 0).sum()),
    }


def op_from_ids_nb(
    fit_ids: torch.Tensor, fit_ops: list[str], eval_ids: torch.Tensor, eval_ops: list[str]
):
    """Naive-Bayes op-from-IDs (section E): FIT on quantized fit-set steps,
    TESTED on quantized eval-set steps. Returns
    {"acc", "majority", "per_slot": [acc_slot0, ...]}. `k_sizes` for the
    smoothing table is read from the ACTUAL max id across fit+eval (a
    dictionary's declared K when available is the caller's business; this
    function only needs a smoothing-safe upper bound)."""
    from marker.summaryprobe import encode_labels, majority_rate  # noqa: PLC0415

    y_fit = encode_labels(fit_ops)
    y_eval = encode_labels(eval_ops)
    k_sizes = [
        int(torch.cat([fit_ids[:, s], eval_ids[:, s]]).max()) + 1 for s in range(fit_ids.shape[1])
    ]
    model = fit_categorical_nb(fit_ids, y_fit, n_classes=4, k_sizes=k_sizes)
    preds = predict_categorical_nb(model, eval_ids)
    acc = float((preds == y_eval).float().mean())

    per_slot = []
    for s in range(fit_ids.shape[1]):
        m = fit_categorical_nb(fit_ids, y_fit, n_classes=4, k_sizes=k_sizes, slots=[s])
        p = predict_categorical_nb(m, eval_ids)
        per_slot.append(round(float((p == y_eval).float().mean()), 4))

    return {"acc": round(acc, 4), "majority": round(majority_rate(y_eval), 4), "per_slot": per_slot}


def _slot_table(dict_: dict, s: int) -> torch.Tensor:
    """The [K, D] table used to look up slot s's quantized KV, for kv/kv_res/
    ro/whole dicts (kv_res needs c1+c2 recombined per stored id -- but its
    ids are JOINT, so this returns a materialized [K1*K2, D] table once,
    matching kv_res's tokenize/detokenize id convention id = id1*K2+id2)."""
    slot = dict_["entry"]["slots"][s] if dict_["kind"] == "whole" else dict_["slots"][s]
    if dict_["kind"] == "kv_res":
        c1, c2, k2 = slot["c1"].float(), slot["c2"].float(), slot["K2"]
        return torch.stack([c1[i1] + c2[i2] for i1 in range(c1.shape[0]) for i2 in range(k2)])
    return slot["centroids"].float()


def reconstruction_cosines_from_shards(
    shard_dir, dict_: dict, geometry: dict, ids: torch.Tensor
) -> dict:  # noqa: ANN001
    """Per-slot AND per-layer mean cosine between the quantized (looked-up)
    KV and the ORIGINAL fit-set KV, straight off the on-disk shards -- pure
    CPU, no GPU or reader needed (section E). Loads one slot's shard at a
    time (never all k_slots), matching the shard-I/O RAM contract."""
    import torch.nn.functional as F  # noqa: N812, PLC0415

    k_slots = geometry["k_slots"]
    n_layers, n_kv_heads, head_dim = (
        geometry["n_layers"],
        geometry["n_kv_heads"],
        geometry["head_dim"],
    )
    per_layer = 2 * n_kv_heads * head_dim
    per_slot_cos, per_layer_cos = [], [[] for _ in range(n_layers)]
    for s in range(k_slots):
        table = _slot_table(dict_, s)
        orig = load_slot_shard(shard_dir, s).float()
        q = table[ids[:, s]]
        cos = F.cosine_similarity(q, orig, dim=1)
        per_slot_cos.append(round(float(cos.mean()), 4))
        for layer in range(n_layers):
            lo, hi = layer * per_layer, (layer + 1) * per_layer
            per_layer_cos[layer].append(
                float(F.cosine_similarity(q[:, lo:hi], orig[:, lo:hi], dim=1).mean())
            )
        del orig
    return {
        "per_slot": per_slot_cos,
        "per_layer": [round(sum(v) / len(v), 4) for v in per_layer_cos],
    }


# ── F. manifest + push ───────────────────────────────────────────────────────


def _push_with_retry(repo_id, folder, path_in_repo):  # noqa: ANN001
    """Delegates to run_render's retry helper (imported, not copied)."""
    from marker.run_render import _push_with_retry as _impl  # noqa: PLC0415

    _impl(repo_id, folder, path_in_repo)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None, help="Stage-1 gist adapter repo")
    ap.add_argument("--render-subdir", default="render_adapter_oneform")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--openr1-dataset", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--n-fit", type=int, default=40000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-fresh", type=int, default=150)
    ap.add_argument("--max-span", type=int, default=MAX_SPAN)
    ap.add_argument("--base", type=int, default=BASE)
    ap.add_argument(
        "--ks", default="256,1024,4096", help="comma-separated K for the plain kv_K* configs"
    )
    ap.add_argument("--res-k1", type=int, default=1024)
    ap.add_argument("--res-k2", type=int, default=1024)
    ap.add_argument("--ro-k", type=int, default=1024)
    ap.add_argument("--whole-k", type=int, default=4096)
    ap.add_argument("--whole-proj-dim", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--cache-dir", default="/tmp/gist_dict_cache")  # noqa: S108
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo = "Qwen/Qwen2.5-0.5B", None
        args.n_fit, args.n_eval, args.n_fresh = 40, 6, 6
        args.ks = "8"
        args.res_k1, args.res_k2, args.ro_k, args.whole_k, args.whole_proj_dim = 4, 2, 8, 8, 6
        args.max_new = 12
        args.eval, args.diagnose = True, True

    ks = [int(x) for x in args.ks.split(",") if x]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    k_slots = gist.shape[0]
    pm.set_adapter("default")

    if args.smoke:
        from marker.render import attach_render  # noqa: PLC0415

        attach_render(pm, r=4)
    else:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        loc = snapshot_download(args.repo, allow_patterns=[f"{args.render_subdir}/render/*"])
        pm.load_adapter(f"{loc}/{args.render_subdir}/render", adapter_name="render")
    pm.set_adapter("default")

    # ── B. eval sets (built first so the fit set can be doc-disjoint) ───────
    eval_gsm8k, eval_gsm8k_docs = build_eval_set(
        _gsm8k_docs("test", args.n_eval * 3, args.dataset, smoke=args.smoke),
        tok,
        args.max_span,
        args.n_eval,
    )
    eval_fresh, eval_fresh_docs = build_eval_set(
        _fresh_docs(args.n_fresh * 2), tok, args.max_span, args.n_fresh
    )
    print(f"eval sets: {len(eval_gsm8k)} gsm8k, {len(eval_fresh)} fresh", flush=True)

    counters = {"too_long": 0}
    fit_doc_iters = (
        [_gsm8k_docs("train", args.n_fit, args.dataset, smoke=args.smoke)]
        if args.smoke
        else [
            _gsm8k_docs("train", args.n_fit, args.dataset),
            _openr1_docs(args.n_fit, args.openr1_dataset),
        ]
    )
    fit_items = build_fit_items(
        fit_doc_iters, tok, args.max_span, args.n_fit, eval_gsm8k_docs | eval_fresh_docs, counters
    )
    print(f"fit set: {len(fit_items)} steps ({counters['too_long']} dropped, too long)", flush=True)

    # ── A. encode the fit set, streamed straight to per-slot shards ────────
    def _fit_kv_readout_stream():
        for _doc_key, ids, _text in fit_items:
            kv, readout, _cs = encode_canonical(
                pm, gist, ids, base=args.base, max_span=args.max_span
            )
            yield kv, readout

    # geometry read from a real probe encode (run_bridge.py's own convention)
    # -- never hardcoded, since a different base model changes every one of
    # these numbers.
    probe_ids = fit_items[0][1] if fit_items else tok("hi", add_special_tokens=False).input_ids
    probe_kv, probe_readout, _ = encode_canonical(
        pm, gist, probe_ids, base=args.base, max_span=args.max_span
    )
    n_kv_heads, head_dim = probe_kv.keys[0].shape[1], probe_kv.keys[0].shape[3]
    geometry = {
        "n_layers": probe_kv.n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "k_slots": k_slots,
    }
    d = geometry["n_layers"] * 2 * n_kv_heads * head_dim
    dr = probe_readout.shape[-1]

    shard_dir = Path(args.cache_dir) / "fit_shards"
    n_fit_actual = write_fit_shards(_fit_kv_readout_stream(), shard_dir, d, dr, k_slots)
    print(f"wrote {n_fit_actual} fit steps to {shard_dir}", flush=True)

    # ── C. dictionaries ──────────────────────────────────────────────────────
    dicts, fit_ids = build_all_dicts(
        shard_dir,
        k_slots,
        geometry,
        ks,
        args.res_k1,
        args.res_k2,
        args.ro_k,
        args.whole_k,
        args.whole_proj_dim,
        seed=args.seed,
    )
    dict_out = Path(args.cache_dir) / "dicts"
    for dict_ in dicts.values():
        save_dict(dict_, dict_out)
    # push the fit-set's readouts + tokenized ids (small) alongside the
    # dictionaries -- NEVER the 18GB-at-40k per-slot KV shards (section F)
    shutil.copy(shard_dir / "readouts.safetensors", dict_out / "fit_readouts.safetensors")
    torch.save(fit_ids, dict_out / "fit_ids.pt")

    # naive-Bayes fit pool: ONLY fit steps with an extractable relation (an
    # unlabeled step has no honest class to assign it to -- dropping it is
    # the fail-loud choice, never a placeholder label that would bias the
    # fit). fit_ids[name] keeps its row order, so the SAME boolean mask
    # selects the matching id rows for every config.
    from marker.summaryprobe import op_label as _op_label  # noqa: PLC0415

    fit_texts = [text for _doc_key, _ids, text in fit_items]
    fit_label_mask = torch.tensor([_op_label(t) is not None for t in fit_texts])
    fit_ops_valid = [_op_label(t) for t in fit_texts if _op_label(t) is not None]

    manifest: dict = {
        "n_fit": n_fit_actual,
        "n_fit_too_long": counters["too_long"],
        "n_fit_labeled": len(fit_ops_valid),
        "n_eval_gsm8k": len(eval_gsm8k),
        "n_eval_fresh": len(eval_fresh),
        "geometry": geometry,
        "configs": {},
    }
    eval_ids_cache: dict[str, dict[str, torch.Tensor]] = {}

    gate0_ok = True
    if args.eval:
        first_name = next(iter(dicts))
        for name, dict_ in dicts.items():
            cells_gsm8k, ids_gsm8k = eval_config(
                pm, tok, gist, eval_gsm8k, dict_, geometry, seed=args.seed, max_new=args.max_new
            )
            cells_fresh, ids_fresh = eval_config(
                pm, tok, gist, eval_fresh, dict_, geometry, seed=args.seed + 1, max_new=args.max_new
            )
            if name == first_name:  # gate 0 only needs checking once (native placement)
                gate0_ok = gate0_pass(cells_gsm8k["native"], cells_fresh["native"])
            r_gsm8k = compute_r(
                cells_gsm8k["quantized"]["rel_exact"],
                cells_gsm8k["wrong_doc_quantized"]["rel_exact"],
                cells_gsm8k["native"]["rel_exact"],
            )
            r_fresh = compute_r(
                cells_fresh["quantized"]["rel_exact"],
                cells_fresh["wrong_doc_quantized"]["rel_exact"],
                cells_fresh["native"]["rel_exact"],
            )
            manifest["configs"].setdefault(name, {})
            manifest["configs"][name].update(
                {
                    "kind": "whole" if dict_["kind"] == "whole" else "per_slot",
                    "conditions": cells_gsm8k,  # the smoke/reader's flat view (gsm8k)
                    "conditions_gsm8k": cells_gsm8k,
                    "conditions_fresh": cells_fresh,
                    "R_gsm8k": round(r_gsm8k, 4),
                    "R_fresh": round(r_fresh, 4),
                }
            )
            eval_ids_cache[name] = {"gsm8k": ids_gsm8k, "fresh": ids_fresh}
        manifest["gate0_pass"] = gate0_ok
    else:
        manifest["gate0_pass"] = None

    if args.diagnose:
        eval_gsm8k_ops = [
            _op_label(s.text) for s in eval_gsm8k
        ]  # build_eval_set guarantees non-None
        for name, dict_ in dicts.items():
            K = _dict_k(dict_)
            # usage from the FIT set's own assignments (a byproduct of
            # k-means, uniform across kv/kv_res/ro/whole -- never a
            # kind-specific stored field, so residual's JOINT usage is
            # correct rather than approximated from one stage)
            usage = torch.bincount(fit_ids[name][:, 0], minlength=K)
            u = usage_diagnostics(usage)
            manifest["configs"].setdefault(name, {})
            manifest["configs"][name]["usage_entropy"] = u["entropy_norm"]
            manifest["configs"][name]["dead_entries"] = u["dead_entries"]

            fit_ids_valid = fit_ids[name][fit_label_mask]
            eval_ids_gsm8k = eval_ids_cache.get(name, {}).get("gsm8k")
            if eval_ids_gsm8k is not None and fit_ids_valid.shape[0] > 0:
                nb = op_from_ids_nb(fit_ids_valid, fit_ops_valid, eval_ids_gsm8k, eval_gsm8k_ops)
                manifest["configs"][name]["op_from_ids"] = nb["acc"]
                manifest["configs"][name]["op_from_ids_majority"] = nb["majority"]
                manifest["configs"][name]["op_from_ids_per_slot"] = nb["per_slot"]
                manifest["configs"][name]["op"] = nb["acc"]
            else:
                manifest["configs"][name]["op_from_ids"] = None
                manifest["configs"][name]["op"] = 0.0

            # per-slot/per-layer reconstruction cosine (quantized vs original
            # KV), CPU-only, straight off the fit shards -- no GPU needed
            cos = reconstruction_cosines_from_shards(shard_dir, dict_, geometry, fit_ids[name])
            manifest["configs"][name]["reconstruction_cosine"] = cos

        ro_name, kv_name = f"ro_K{args.ro_k}", f"kv_K{ks[0]}"
        ro_ids = eval_ids_cache.get(ro_name, {}).get("gsm8k")
        kv_ids = eval_ids_cache.get(kv_name, {}).get("gsm8k")
        if ro_ids is not None and kv_ids is not None:
            manifest["ro_vs_kv_ami"] = round(
                adjusted_mutual_info(ro_ids[:, 0].tolist(), kv_ids[:, 0].tolist()), 4
            )

    if args.eval and args.diagnose:
        manifest["verdict"] = stage1_verdict(
            {"gate0_pass": manifest["gate0_pass"], "configs": manifest["configs"]}
        )
    else:
        manifest["verdict"] = None

    print(f"[GISTDICT MANIFEST] {json.dumps(manifest, default=str)}", flush=True)

    if not args.smoke and args.out_repo:
        _push_with_retry(args.out_repo, str(dict_out), "gist_dict")
        print(f"pushed dictionaries to {args.out_repo}/gist_dict", flush=True)
        manifest_dir = Path(args.cache_dir) / "manifest_out"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        _push_with_retry(args.out_repo, str(manifest_dir), "gist_dict")
        print(f"pushed manifest to {args.out_repo}/gist_dict", flush=True)


if __name__ == "__main__":
    main()
