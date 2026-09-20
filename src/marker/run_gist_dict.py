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
    reusing run_stage2's generic corpus loader + SENTENCE splitter -- the
    unit every OpenR1 consumer in this repo uses (vast_cot.sh encoded
    stage2_cot_openr1 with UNIT=sentence; vast_bridge/vast_rollout inherit
    it), so the fit distribution matches how OpenR1 text was always fed to
    the encoder. Line-splitting OpenR1's paragraph-y LaTeX solutions would
    yield mostly over-length units that get dropped."""
    from marker.run_stage2 import _doc_texts, _split_units  # noqa: PLC0415

    for i, text in enumerate(_doc_texts(n, "cot", dataset)):
        yield ("openr1", i), _split_units(text, "sentence")


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
    construction, so pick_wrong_doc_indices's cross-doc pairing never needs a
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
            # .cpu() before .numpy(): on the real run the encode lands on
            # CUDA, and .numpy() on a CUDA tensor raises
            mat = kv_slot_matrix(kv).half().cpu().contiguous()  # [k_slots, D]
            for s in range(k_slots):
                handles[s].write(mat[s].numpy().tobytes())
            ro_handle.write(readout.half().cpu().contiguous().numpy().tobytes())
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
    device: str = "cpu",
):
    """Builds every configured dictionary from the on-disk shards, loading
    at most one slot at a time (per gist_dict.build_dict_* + build_dict_whole
    contracts). k-means runs on `device` (spec section C: GPU when available
    -- Lloyd's on [40k, 28672] x K=4096 x 20 iters is ~1e14 FLOPs, hours on
    CPU, seconds-per-slot on the card the model already occupies); every
    stored entry comes back on CPU regardless (builder contract). Returns
    (dicts, fit_ids): dicts[cfg] is a tokenize/detokenize-ready dict;
    fit_ids[cfg] is [N, k_slots] long -- the fit set's OWN tokenized ids, a
    free byproduct of the k-means assignment (never recomputed by re-running
    tokenize over the whole fit set)."""
    readouts = load_readouts(shard_dir)  # [N, k_slots, Dr] fp16, kept resident

    def _slot_ro(s):
        return readouts[:, s, :].float().to(device)

    def _slot_mat(s):
        return load_slot_shard(shard_dir, s).float().to(device)

    dicts: dict[str, dict] = {}
    fit_ids: dict[str, torch.Tensor] = {}

    for K in ks:
        slots, assigns = [], []
        for s in range(k_slots):
            mat = _slot_mat(s)
            entry, assign = build_dict_kv(mat, _slot_ro(s), K, seed=seed)
            slots.append(entry)
            assigns.append(assign)
            del mat
        name = f"kv_K{K}"
        dicts[name] = {"cfg": name, "kind": "kv", "geometry": geometry, "slots": slots}
        fit_ids[name] = torch.stack(assigns, dim=1)
        print(f"built {name}", flush=True)

    slots, assigns = [], []
    for s in range(k_slots):
        mat = _slot_mat(s)
        entry, a1, a2 = build_dict_kv_residual(mat, _slot_ro(s), res_k1, res_k2, seed=seed)
        slots.append(entry)
        assigns.append(a1 * res_k2 + a2)
        del mat
    name = f"kv_res_{res_k1}x{res_k2}"
    dicts[name] = {"cfg": name, "kind": "kv_res", "geometry": geometry, "slots": slots}
    fit_ids[name] = torch.stack(assigns, dim=1)
    print(f"built {name}", flush=True)

    slots, assigns = [], []
    for s in range(k_slots):
        mat = _slot_mat(s)
        entry, assign = build_dict_ro(_slot_ro(s), mat, ro_k, seed=seed)
        slots.append(entry)
        assigns.append(assign)
        del mat
    name = f"ro_K{ro_k}"
    dicts[name] = {"cfg": name, "kind": "ro", "geometry": geometry, "slots": slots}
    fit_ids[name] = torch.stack(assigns, dim=1)
    print(f"built {name}", flush=True)

    entry, assign = build_dict_whole(
        _slot_mat, k_slots, whole_k, seed=seed, proj_dim=whole_proj_dim, proj_seed=seed
    )
    # per-slot mean readouts for the whole dict too (spec C stores mu for
    # every config; quantized_readouts / the centroid probe need it)
    from marker.gist_dict import _cluster_means  # noqa: PLC0415

    for s in range(k_slots):
        entry["slots"][s]["mu_readout"] = _cluster_means(
            readouts[:, s, :].float(), assign, whole_k
        ).half()
    name = f"whole_K{whole_k}"
    dicts[name] = {"cfg": name, "kind": "whole", "geometry": geometry, "entry": entry}
    fit_ids[name] = assign.unsqueeze(1).expand(-1, k_slots).clone()
    print(f"built {name}", flush=True)

    return dicts, fit_ids


def save_dict(dict_: dict, out_dir) -> None:  # noqa: ANN001
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict_, out / f"dict_{dict_['cfg']}.pt")


# ── D. GPU eval: native / quantized / wrong_doc_quantized / random_ids ──────


def pick_wrong_doc_indices(n: int, gen: torch.Generator) -> list[int]:
    """For each eval step i (one per doc, see build_eval_set), the index of a
    step from a DIFFERENT doc (predprobe.pick_cross_doc_step) -- its KV is
    scored later against step i's own text (the cheating floor). Indices, not
    steps: the wrong step's canonical KV is already in the natives list
    (every eval step is encoded exactly once), so no re-encode."""
    doc_lengths = [1] * n
    out = []
    for i in range(n):
        dj, _sj = pick_cross_doc_step(i, doc_lengths, gen, step_idx=0)
        out.append(dj)
    return out


def random_ids(K: int, k_slots: int, gen: torch.Generator) -> list[int]:
    """k_slots uniform random ids in [0, K), seeded via `gen` -- the
    random_ids condition's floor."""
    return [int(torch.randint(0, K, (1,), generator=gen)) for _ in range(k_slots)]


def _condition_metrics(
    pm, tok, gold: EvalStep, kv, cs: int, nl_id, max_new: int = MAX_NEW, nll_only: bool = False
) -> dict:  # noqa: ANN001
    """f1/num_recall/rel_exact (run_render._score_record, reused not copied)
    + teacher-forced render NLL of the TRUE step (gistprobe.per_token_ce) --
    the sensitive metric the spec calls out. `render` adapter must already
    be active. Ledger ON (spec section D): the reader's trained frame -- and
    the gate-0 thresholds -- assume the visible-literals ledger prefix, both
    for generation and for the teacher-forced NLL. nll_only skips the
    generation (the random_ids floor's wall-clock cut: its NLL is the
    sensitive floor metric, its rendered relations are not gate inputs)."""
    from marker.run_render import _ledger_ids  # noqa: PLC0415

    ledger = _ledger_ids(tok, gold.text)
    tail = list(gold.ids) + ([nl_id] if nl_id is not None else [])
    ce, _tgt = per_token_ce(pm, kv, cs, ledger, tail)
    if nll_only:
        return {"nll": float(ce.mean())}
    stop_ids = {nl_id} if nl_id is not None else set()
    f1, nr, rel = _score_record(pm, tok, gold, kv, cs, True, stop_ids, max_new)
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


def _summarize_nll(metrics: list[dict]) -> dict:
    """Summary for an NLL-only condition (random_ids): just the teacher-forced
    NLL -- no rendered-relation fields to mistake for real ones."""
    return {
        "nll_mean": round(sum(m["nll"] for m in metrics) / max(1, len(metrics)), 4),
        "n": len(metrics),
        "nll_only": True,
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


def _nl_id(tok):  # noqa: ANN001
    return next((t for t in tok("\n", add_special_tokens=False).input_ids if t), None)


@torch.no_grad()
def score_native(pm, tok, eval_steps: list[EvalStep], natives, max_new: int = MAX_NEW) -> dict:  # noqa: ANN001
    """The native condition, scored ONCE per eval set -- it is
    config-independent (the step's own canonical KV through the reader), so
    re-running it per dictionary config would multiply the most expensive
    GPU loop by the config count for identical numbers. `render` adapter
    must already be active."""
    nl = _nl_id(tok)
    metrics = [
        _condition_metrics(pm, tok, gold, kv, cs, nl, max_new)
        for gold, (kv, _ro, cs) in zip(eval_steps, natives, strict=True)
    ]
    return _summarize(metrics)


def tokenize_eval(natives, dict_) -> torch.Tensor:  # noqa: ANN001
    """Every eval step's ids under one config -- pure CPU, no reader. Runs
    for EVERY config (including ones cut from the GPU eval) so op-from-IDs
    and the AMI diagnostic never depend on which configs got GPU budget."""
    return torch.tensor(
        [tokenize(kv, dict_, readout=ro) for kv, ro, _cs in natives], dtype=torch.long
    )


@torch.no_grad()
def eval_quantized_conditions(
    pm,
    tok,
    eval_steps: list[EvalStep],
    natives,
    wrong_idx: list[int],
    ids: torch.Tensor,
    dict_: dict,
    geometry: dict,
    seed: int = 0,
    max_new: int = MAX_NEW,
):  # noqa: ANN001
    """The config-DEPENDENT conditions of section D on one eval set:
    quantized + wrong_doc_quantized (full generation + NLL), random_ids
    (teacher-forced NLL only -- the wall-clock cut; see _condition_metrics).
    `render` adapter must already be active. `ids` is tokenize_eval's output
    for this config; the wrong step's ids are looked up from the same tensor
    (its KV was already tokenized as its OWN row)."""
    k_slots = geometry["k_slots"]
    rand_gen = torch.Generator().manual_seed(seed + 1)
    nl = _nl_id(tok)
    metrics = {"quantized": [], "wrong_doc_quantized": [], "random_ids": []}
    for i, gold in enumerate(eval_steps):
        cs = natives[i][2]
        metrics["quantized"].append(
            _condition_metrics(
                pm, tok, gold, detokenize(ids[i].tolist(), dict_, geometry), cs, nl, max_new
            )
        )
        metrics["wrong_doc_quantized"].append(
            _condition_metrics(
                pm,
                tok,
                gold,
                detokenize(ids[wrong_idx[i]].tolist(), dict_, geometry),
                cs,
                nl,
                max_new,
            )
        )
        rids = random_ids(_dict_k(dict_), k_slots, rand_gen)
        metrics["random_ids"].append(
            _condition_metrics(
                pm, tok, gold, detokenize(rids, dict_, geometry), cs, nl, max_new, nll_only=True
            )
        )
    return {
        "quantized": _summarize(metrics["quantized"]),
        "wrong_doc_quantized": _summarize(metrics["wrong_doc_quantized"]),
        "random_ids": _summarize_nll(metrics["random_ids"]),
    }


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


def usage_by_slot(fit_ids_cfg: torch.Tensor, K: int) -> list[dict]:
    """usage_diagnostics per slot -- a per-slot dictionary has 8 independent
    codebooks with 8 independent usage patterns (slot 0's entropy says
    nothing about slot 5's); a whole dict just repeats one column 8x, which
    collapses to identical entries."""
    return [
        usage_diagnostics(torch.bincount(fit_ids_cfg[:, s], minlength=K))
        for s in range(fit_ids_cfg.shape[1])
    ]


def quantized_readouts(dict_: dict, ids: torch.Tensor) -> torch.Tensor:
    """ids [N, k_slots] -> the centroid mean-readouts [N, k_slots, Dr]
    float32 (what a guesser over IDs could reconstruct). kv/ro/whole store a
    dense per-slot mu_readout [K, Dr]; kv_res stores it SPARSELY over the
    occupied joint ids (mu_ids) -- an id never occupied in fit gets a zero
    row, honestly reflecting that the dictionary carries no readout for it."""
    kind = dict_["kind"]
    n, k_slots = ids.shape
    out = None
    for s in range(k_slots):
        slot = dict_["entry"]["slots"][s] if kind == "whole" else dict_["slots"][s]
        if kind == "kv_res":
            mu_ids, mu = slot["mu_ids"], slot["mu_readout"].float()
            pos = torch.searchsorted(mu_ids, ids[:, s].clamp(max=int(mu_ids[-1])))
            hit = mu_ids[pos] == ids[:, s]  # compared against the UNclamped id
            rows = torch.zeros(n, mu.shape[1])
            rows[hit] = mu[pos[hit]]
        else:
            rows = slot["mu_readout"].float()[ids[:, s]]
        if out is None:
            out = torch.zeros(n, k_slots, rows.shape[1])
        out[:, s] = rows
    return out


def centroid_readout_op_probe(
    q_fit: torch.Tensor,
    fit_ops: list[str],
    fit_doc_keys: list,
    q_eval: torch.Tensor,
    eval_ops: list[str],
    seed: int = 0,
    pca_dim: int = 128,
) -> dict | None:
    """Section E's 'op probe on centroid readouts, trained on quantized':
    summaryprobe's exact recipe (per-slot normalize -> standardize ->
    PCA -> linear probe, early-stopped on a doc-disjoint val carve-out),
    trained on the QUANTIZED fit readouts, tested on the QUANTIZED eval
    readouts. Reference: the continuous-readout probe scored 0.825. Returns
    evaluate_probe's dict + majority, or None when the fit split is too
    small to carve a val set from (tiny smoke sets)."""
    from marker.summaryprobe import (  # noqa: PLC0415
        OP_CLASSES,
        doc_disjoint_split,
        encode_labels,
        evaluate_probe,
        majority_rate,
        normalize_flatten,
        pca_apply,
        pca_fit,
        standardize_apply,
        standardize_fit,
        train_probe,
    )

    doc_to_int = {d: i for i, d in enumerate(dict.fromkeys(fit_doc_keys))}
    doc_ints = [doc_to_int[d] for d in fit_doc_keys]
    keep, hold = doc_disjoint_split(doc_ints, frac_holdout=0.15, seed=seed)
    idx_tr = [i for i, d in enumerate(doc_ints) if d in keep]
    idx_va = [i for i, d in enumerate(doc_ints) if d in hold]
    if len(idx_tr) < 8 or len(idx_va) < 2 or len(eval_ops) == 0:
        return None
    y_fit = encode_labels(fit_ops)
    y_eval = encode_labels(eval_ops)
    x_all = normalize_flatten(q_fit)
    mean, std = standardize_fit(x_all[idx_tr])
    xt = standardize_apply(x_all[idx_tr], mean, std)
    xv = standardize_apply(x_all[idx_va], mean, std)
    pmean, comps = pca_fit(xt, pca_dim)
    xt, xv = pca_apply(xt, pmean, comps), pca_apply(xv, pmean, comps)
    model = train_probe(xt, y_fit[idx_tr], xv, y_fit[idx_va], n_classes=len(OP_CLASSES), seed=seed)
    xe = pca_apply(standardize_apply(normalize_flatten(q_eval), mean, std), pmean, comps)
    out = evaluate_probe(model, xe, y_eval)
    out["majority"] = round(majority_rate(y_eval), 4)
    return out


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


def _quantized_rows(dict_: dict, s: int, ids_col: torch.Tensor) -> torch.Tensor:
    """The quantized [len(ids_col), D] KV rows for slot s, looked up straight
    from the stored ids -- for kv_res via c1[id1] + c2[id2], NEVER by
    materializing the joint [K1*K2, D] table (K1*K2 ~ 1e6 rows x 28,672
    floats would be ~120 GB; the smoke's K1=4,K2=2 would happily hide that)."""
    slot = dict_["entry"]["slots"][s] if dict_["kind"] == "whole" else dict_["slots"][s]
    if dict_["kind"] == "kv_res":
        k2 = slot["K2"]
        return slot["c1"][ids_col // k2].float() + slot["c2"][ids_col % k2].float()
    return slot["centroids"][ids_col].float()


def reconstruction_cosines_from_shards(
    shard_dir, dict_: dict, geometry: dict, ids: torch.Tensor, chunk: int = 4096
) -> dict:  # noqa: ANN001
    """Per-slot AND per-layer mean cosine between the quantized (looked-up)
    KV and the ORIGINAL fit-set KV, straight off the on-disk shards -- pure
    CPU, no GPU or reader needed (section E). Loads one slot's shard at a
    time (never all k_slots) and walks it in row chunks, so peak extra RAM is
    ~2 x chunk x D floats, not two full [N, D] fp32 copies."""
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
        orig16 = load_slot_shard(shard_dir, s)  # [N, D] fp16, kept half
        n = orig16.shape[0]
        slot_sum, layer_sums = 0.0, [0.0] * n_layers
        for lo_row in range(0, n, chunk):
            rows = slice(lo_row, min(lo_row + chunk, n))
            q = _quantized_rows(dict_, s, ids[rows, s])
            orig = orig16[rows].float()
            slot_sum += float(F.cosine_similarity(q, orig, dim=1).sum())
            for layer in range(n_layers):
                lo, hi = layer * per_layer, (layer + 1) * per_layer
                layer_sums[layer] += float(
                    F.cosine_similarity(q[:, lo:hi], orig[:, lo:hi], dim=1).sum()
                )
        per_slot_cos.append(round(slot_sum / n, 4))
        for layer in range(n_layers):
            per_layer_cos[layer].append(layer_sums[layer] / n)
        del orig16
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
    ap.add_argument(
        "--eval-configs",
        default="auto",
        help="comma-separated config names to run the GPU eval conditions on; "
        "'auto' = every config except the smallest kv_K* and ro_K* (those two "
        "still get every CPU diagnostic -- ids, op-from-IDs, cosines, AMI); "
        "'all' = every config. The cut keeps the generation loop inside the "
        "node's wall-clock budget.",
    )
    ap.add_argument("--probe-fit-cap", type=int, default=20000)
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
        args.eval_configs = "all"  # smoke exercises every config's GPU path

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

    # ── C. dictionaries (k-means on the GPU when there is one) ──────────────
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
        device=device,
    )
    dict_out = Path(args.cache_dir) / "dicts"
    for dict_ in dicts.values():
        save_dict(dict_, dict_out)
    # push the fit-set's readouts + tokenized ids (small) alongside the
    # dictionaries -- NEVER the 18GB-at-40k per-slot KV shards (section F)
    shutil.copy(shard_dir / "readouts.safetensors", dict_out / "fit_readouts.safetensors")
    torch.save(fit_ids, dict_out / "fit_ids.pt")

    # EARLY push, BEFORE the eval loop (the render-run lesson: the encode +
    # k-means above are the expensive irreplaceable part -- if the eval then
    # hits the wall-clock cap, the dictionaries must already be off the node)
    if not args.smoke and args.out_repo:
        _push_with_retry(args.out_repo, str(dict_out), "gist_dict")
        print(f"pushed dictionaries (early, pre-eval) to {args.out_repo}/gist_dict", flush=True)

    # which configs get the GPU generation loop (all of them always get the
    # CPU diagnostics -- ids, op-from-IDs, cosines, AMI, readout probe)
    all_names = list(dicts)
    if args.eval_configs == "all":
        eval_names = set(all_names)
    elif args.eval_configs == "auto":
        drop = {f"ro_K{args.ro_k}"}
        if len(ks) > 1:
            drop.add(f"kv_K{min(ks)}")
        eval_names = {n for n in all_names if n not in drop}
    else:
        eval_names = {x for x in args.eval_configs.split(",") if x}
        unknown = eval_names - set(all_names)
        assert not unknown, f"--eval-configs names not built: {sorted(unknown)} vs {all_names}"

    # op labels: ONLY fit steps with an extractable relation (an unlabeled
    # step has no honest class to assign it to -- dropping it is the
    # fail-loud choice, never a placeholder label that would bias the fit).
    # fit_ids[name] keeps its row order, so the SAME index list selects the
    # matching id rows for every config.
    from marker.summaryprobe import op_label as _op_label  # noqa: PLC0415

    fit_ops_all = [_op_label(text) for _doc_key, _ids, text in fit_items]
    lab_idx = torch.tensor(
        [i for i, o in enumerate(fit_ops_all) if o is not None], dtype=torch.long
    )
    fit_ops_valid = [fit_ops_all[i] for i in lab_idx.tolist()]

    manifest: dict = {
        "n_fit": n_fit_actual,
        "n_fit_too_long": counters["too_long"],
        "n_fit_labeled": len(fit_ops_valid),
        "n_eval_gsm8k": len(eval_gsm8k),
        "n_eval_fresh": len(eval_fresh),
        "geometry": geometry,
        "eval_configs": sorted(eval_names),
        "configs": {},
    }

    # ── eval-step canonical encodes + per-config ids (needed by BOTH the
    # GPU eval and the CPU diagnostics) ─────────────────────────────────────
    natives: dict[str, list] = {}
    eval_ids_cache: dict[str, dict[str, torch.Tensor]] = {}
    if args.eval or args.diagnose:
        pm.set_adapter("default")
        natives["gsm8k"] = [
            encode_canonical(pm, gist, s.ids, base=args.base, max_span=args.max_span)
            for s in eval_gsm8k
        ]
        natives["fresh"] = [
            encode_canonical(pm, gist, s.ids, base=args.base, max_span=args.max_span)
            for s in eval_fresh
        ]
        for name, dict_ in dicts.items():
            eval_ids_cache[name] = {
                "gsm8k": tokenize_eval(natives["gsm8k"], dict_),
                "fresh": tokenize_eval(natives["fresh"], dict_),
            }
        print("eval steps encoded + tokenized under every config", flush=True)

    gate0_ok = True
    if args.eval:
        # gate 0 / native: config-independent, computed ONCE
        pm.set_adapter("render")
        native_gsm8k = score_native(pm, tok, eval_gsm8k, natives["gsm8k"], max_new=args.max_new)
        native_fresh = score_native(pm, tok, eval_fresh, natives["fresh"], max_new=args.max_new)
        pm.set_adapter("default")
        manifest["native_gsm8k"] = native_gsm8k
        manifest["native_fresh"] = native_fresh
        gate0_ok = gate0_pass(native_gsm8k, native_fresh)
        manifest["gate0_pass"] = gate0_ok
        print(
            f"gate0 native: gsm8k={native_gsm8k['rel_exact']} fresh={native_fresh['rel_exact']} "
            f"pass={gate0_ok}",
            flush=True,
        )
        if not gate0_ok:
            # spec section D: nothing downstream is trustworthy -- don't
            # spend hours of generation on it (verdict: INVALID_HARNESS).
            # --smoke still walks the whole path (its untrained render
            # adapter can't pass gate 0; the point is exercising the code).
            print("[GISTDICT] GATE 0 FAILED -- skipping per-config GPU eval", flush=True)
        if gate0_ok or args.smoke:
            wrong_idx = {
                "gsm8k": pick_wrong_doc_indices(
                    len(eval_gsm8k), torch.Generator().manual_seed(args.seed)
                ),
                "fresh": pick_wrong_doc_indices(
                    len(eval_fresh), torch.Generator().manual_seed(args.seed + 1)
                ),
            }
            pm.set_adapter("render")
            for name in all_names:
                if name not in eval_names:
                    continue
                dict_ = dicts[name]
                cells_gsm8k = eval_quantized_conditions(
                    pm,
                    tok,
                    eval_gsm8k,
                    natives["gsm8k"],
                    wrong_idx["gsm8k"],
                    eval_ids_cache[name]["gsm8k"],
                    dict_,
                    geometry,
                    seed=args.seed,
                    max_new=args.max_new,
                )
                cells_fresh = eval_quantized_conditions(
                    pm,
                    tok,
                    eval_fresh,
                    natives["fresh"],
                    wrong_idx["fresh"],
                    eval_ids_cache[name]["fresh"],
                    dict_,
                    geometry,
                    seed=args.seed + 1,
                    max_new=args.max_new,
                )
                cells_gsm8k["native"] = native_gsm8k
                cells_fresh["native"] = native_fresh
                r_gsm8k = compute_r(
                    cells_gsm8k["quantized"]["rel_exact"],
                    cells_gsm8k["wrong_doc_quantized"]["rel_exact"],
                    native_gsm8k["rel_exact"],
                )
                r_fresh = compute_r(
                    cells_fresh["quantized"]["rel_exact"],
                    cells_fresh["wrong_doc_quantized"]["rel_exact"],
                    native_fresh["rel_exact"],
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
                print(f"eval {name}: R_gsm8k={r_gsm8k:.3f} R_fresh={r_fresh:.3f}", flush=True)
            pm.set_adapter("default")
        # partial manifest NOW: if the wall clock dies during the CPU
        # diagnostics below, the GPU eval's numbers must already be in the
        # log (same lesson as the early dictionary push)
        print(f"[GISTDICT PARTIAL] {json.dumps(manifest, default=str)}", flush=True)
    else:
        manifest["gate0_pass"] = None

    if args.diagnose:
        eval_gsm8k_ops = [
            _op_label(s.text) for s in eval_gsm8k
        ]  # build_eval_set guarantees non-None
        # readout-probe fit rows: labeled steps, capped (seeded subsample)
        probe_idx = lab_idx
        if probe_idx.numel() > args.probe_fit_cap:
            g = torch.Generator().manual_seed(args.seed)
            keep = torch.randperm(probe_idx.numel(), generator=g)[: args.probe_fit_cap]
            probe_idx = probe_idx[keep.sort().values]
        probe_ops = [fit_ops_all[i] for i in probe_idx.tolist()]
        probe_docs = [fit_items[i][0] for i in probe_idx.tolist()]

        for name, dict_ in dicts.items():
            K = _dict_k(dict_)
            cell = manifest["configs"].setdefault(name, {})
            cell.setdefault("kind", "whole" if dict_["kind"] == "whole" else "per_slot")
            # usage from the FIT set's own assignments (a byproduct of
            # k-means, uniform across kv/kv_res/ro/whole), PER SLOT -- each
            # slot is its own codebook with its own entropy/dead count
            u = usage_by_slot(fit_ids[name], K)
            cell["usage_entropy"] = round(sum(x["entropy_norm"] for x in u) / len(u), 4)
            cell["usage_entropy_per_slot"] = [x["entropy_norm"] for x in u]
            cell["dead_entries"] = [x["dead_entries"] for x in u]

            fit_ids_valid = fit_ids[name][lab_idx]
            eval_ids_gsm8k = eval_ids_cache.get(name, {}).get("gsm8k")
            if eval_ids_gsm8k is not None and fit_ids_valid.shape[0] > 0:
                nb = op_from_ids_nb(fit_ids_valid, fit_ops_valid, eval_ids_gsm8k, eval_gsm8k_ops)
                cell["op_from_ids"] = nb["acc"]
                cell["op_from_ids_majority"] = nb["majority"]
                cell["op_from_ids_per_slot"] = nb["per_slot"]
                cell["op"] = nb["acc"]
            else:
                cell["op_from_ids"] = None
                cell["op"] = 0.0

            # per-slot/per-layer reconstruction cosine (quantized vs original
            # KV), CPU-only, straight off the fit shards -- no GPU needed
            cos = reconstruction_cosines_from_shards(shard_dir, dict_, geometry, fit_ids[name])
            cell["reconstruction_cosine"] = cos

            # op probe on the CENTROID readouts, trained on quantized
            # (section E; continuous-readout reference 0.825)
            probe = None
            if eval_ids_gsm8k is not None and probe_idx.numel() > 0:
                q_fit = quantized_readouts(dict_, fit_ids[name][probe_idx])
                q_eval = quantized_readouts(dict_, eval_ids_gsm8k)
                probe = centroid_readout_op_probe(
                    q_fit, probe_ops, probe_docs, q_eval, eval_gsm8k_ops, seed=args.seed
                )
                del q_fit, q_eval
            cell["op_probe_readout"] = probe
            print(f"diagnose {name} done", flush=True)

        # ro-vs-kv ID agreement (spec C: against kv_K1024 when built)
        kv_ref = 1024 if 1024 in ks else ks[0]
        ro_ids = eval_ids_cache.get(f"ro_K{args.ro_k}", {}).get("gsm8k")
        kv_ids = eval_ids_cache.get(f"kv_K{kv_ref}", {}).get("gsm8k")
        if ro_ids is not None and kv_ids is not None:
            per_slot_ami = [
                round(adjusted_mutual_info(ro_ids[:, s].tolist(), kv_ids[:, s].tolist()), 4)
                for s in range(ro_ids.shape[1])
            ]
            manifest["ro_vs_kv_ami_per_slot"] = per_slot_ami
            manifest["ro_vs_kv_ami"] = round(sum(per_slot_ami) / len(per_slot_ami), 4)

    if args.eval and args.diagnose:
        # only configs that ran the GPU eval have R cells; gate0=False makes
        # the verdict INVALID_HARNESS before configs are even consulted
        verdict_cells = {
            n: c for n, c in manifest["configs"].items() if "R_gsm8k" in c and "op" in c
        }
        manifest["verdict"] = stage1_verdict(
            {"gate0_pass": manifest["gate0_pass"], "configs": verdict_cells}
        )
    else:
        manifest["verdict"] = None

    print(f"[GISTDICT MANIFEST] {json.dumps(manifest, default=str)}", flush=True)

    if not args.smoke and args.out_repo:
        manifest_dir = Path(args.cache_dir) / "manifest_out"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        if eval_ids_cache:
            torch.save(eval_ids_cache, manifest_dir / "eval_ids.pt")
        _push_with_retry(args.out_repo, str(manifest_dir), "gist_dict")
        print(f"pushed manifest to {args.out_repo}/gist_dict", flush=True)


if __name__ == "__main__":
    main()
