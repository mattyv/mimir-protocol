"""Stage-2 corpus builder (GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3", Corpus v0;
full build order: scratchpad/stage2_build_order.md): tokenizes GSM8K train +
OpenR1 solutions into the stage-2 gist corpus -- per solution, a question, its
reasoning steps as TEXT, and each step's [k_slots]-id group(s) (see
gist_tokenizer.GistTokenizer) -- doc-disjoint from EVERY GSM8K-test
question (all 1319, one streaming pass), a strict superset of the eval sets
stage 1/3 read: the 200 stage-1b eval docs (reproduced via build_eval_set,
not re-invented) and the summary-probe run's 993 >=3-step docs.

Pipeline: A. load the frozen model + stage-1 dictionary + a GistTokenizer.
B. build the exclusion question set (stage-1b eval docs + all GSM8K test
questions). C. stream GSM8K
train + OpenR1 (run_stage2._doc_texts_qa), filtering each doc through
exclude_doc, encoding the survivors, and writing+pushing JSONL shards as each
one fills (build_corpus) -- resumable: shards already on --out-repo are
skipped, never re-encoded, and numbering continues (append-only). D. also
tokenize the 200 eval docs' WHOLE solutions (same schema) to
eval_gsm8k_test.jsonl, for stage 3's own eval. E. manifest + push.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_tokenize_corpus \\
        --repo mattyvee/mimir-artifacts --out-repo mattyvee/mimir-artifacts
Smoke (local tiny model):
    PYTHONPATH=src python -m marker.run_tokenize_corpus --smoke
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from marker.burst import extract_answer as _extract_answer_gsm8k
from marker.gist_dict import kv_slot_matrix
from marker.gist_tokenizer import GistTokenizer, split_long_step
from marker.reason_check import split_solution_steps
from marker.run_gist_dict import _gsm8k_docs, build_eval_set, encode_canonical
from marker.run_render import _push_with_retry

BASE = 64
MAX_SPAN = 64


def _dataset_revision(name: str) -> str | None:
    """The dataset repo's current commit sha (manifest provenance -- the
    streaming loaders pin nothing, so record WHAT was streamed). Best
    effort: None when the hub call fails, never a run-killing error."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        return HfApi().dataset_info(name).sha
    except Exception:  # noqa: BLE001 -- provenance only, run must not die on it
        return None


# ── A. answer extraction (per source; build order item 2's schema) ─────────

_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer_openr1(solution: str) -> str | None:
    """OpenR1's final answer: the LAST `\\boxed{...}` span (the LaTeX
    final-answer convention OpenR1 solutions use) if there is one, else the
    last non-empty line -- OpenR1 solutions don't carry GSM8K's '#### x'
    marker. None only when the solution has no text at all."""
    boxed = _BOXED.findall(solution)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def extract_answer(src: str, solution: str) -> str | None:
    """GSM8K: the '#### x' marker (burst.extract_answer -- the SAME
    extractor run_burst/run_frontload already score generations against).
    OpenR1: extract_answer_openr1."""
    if src == "gsm8k":
        return _extract_answer_gsm8k(solution)
    return extract_answer_openr1(solution)


# ── contamination filter: normalized-question exact match ──────────────────

_PUNCT = re.compile(r"[^\w\s]")


def normalize_question(text: str) -> str:
    """lowercase, collapse whitespace, strip punctuation -- the exact-match
    key exclusion (b) uses (OpenR1 contains GSM8K-derived items, sometimes
    reformatted)."""
    return " ".join(_PUNCT.sub("", text.lower()).split())


# ── exclude_doc: the counted exclusion reasons (build order item 2) ────────


def exclude_doc(question: str, solution: str, excl: dict) -> str | None:
    """None = keep; else the exclusion reason, each COUNTED by the caller:
      'contaminated' -- (b) normalized question exact-matches an eval set
      'pathological' -- (c) some step needs more than --max-groups groups
      'seq_cap'      -- (d) the laid-out sequence exceeds --seq-cap
    (a) 'any GSM8K-TEST doc never enters' is enforced structurally by the
    loaders (train-only streams, see main()) -- not a runtime check here.
    `excl` bundles everything this needs (kept as one dict so callers can
    reuse the same bundle across sources, swapping only `splitter`):
    {tok, max_span, max_groups, seq_cap, norm_questions (excluded set,
    ALREADY normalized), k_slots, splitter (solution -> steps)}."""
    if normalize_question(question) in excl["norm_questions"]:
        return "contaminated"

    tok, max_span = excl["tok"], excl["max_span"]
    steps = excl["splitter"](solution)
    if not steps:
        return "pathological"  # no usable reasoning steps at all

    groups_per_step = [len(split_long_step(s, tok, max_span)) for s in steps]
    if max(groups_per_step) > excl["max_groups"]:
        return "pathological"

    # layout: [question] <think> g^(1)..g^(m) <commit> [step text] <eos>
    q_len = len(tok(question, add_special_tokens=False).input_ids)
    total_groups = sum(groups_per_step)
    longest_step = max(len(tok(s, add_special_tokens=False).input_ids) for s in steps)
    seq_len = q_len + excl["k_slots"] * total_groups + 1 + longest_step + 2
    if seq_len > excl["seq_cap"]:
        return "seq_cap"
    return None


# ── shard I/O: JSONL, and the pure read-back tally resume relies on ────────


def write_jsonl(records: list[dict], path) -> None:  # noqa: ANN001
    """Write-then-rename: a kill mid-write leaves only a `.tmp` (invisible
    to the `shard_*.jsonl` glob _tally_shards and resume read), never a
    truncated shard that looks complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(path)


def _tally_shards(shard_paths: list) -> dict:
    """docs/steps/groups/split_steps straight from shard FILE CONTENT --
    never a running counter, so a resumed run's totals include shards it
    skipped re-encoding (their content is on disk either freshly written or
    downloaded from --out-repo, see main()) exactly as accurately as shards
    it just wrote."""
    docs = steps = groups = split_steps = 0
    for p in shard_paths:
        text = Path(p).read_text()
        if not text:
            continue
        for line in text.splitlines():
            rec = json.loads(line)
            docs += 1
            steps += len(rec["steps"])
            groups += rec["n_groups"]
            split_steps += sum(1 for g in rec["ids"] if len(g) > 1)
    return {"docs": docs, "steps": steps, "groups": groups, "split_steps": split_steps}


def list_existing_shards(out_repo, out_subdir: str, lister=None) -> set:  # noqa: ANN001
    """Shard FILE NAMES already pushed to <out_repo>/<out_subdir> -- what
    build_corpus's resume skips re-encoding. `lister(repo) -> [file paths]`
    is list_repo_files's signature; tests inject a stand-in. Empty (never an
    error) when there is no --out-repo to check -- a fresh/local-only run."""
    if not out_repo:
        return set()
    if lister is None:
        from huggingface_hub import list_repo_files  # noqa: PLC0415

        lister = list_repo_files
    prefix = f"{out_subdir}/"
    names = set()
    for f in lister(out_repo):
        if f.startswith(prefix):
            tail = f[len(prefix) :]
            if tail.startswith("shard_") and tail.endswith(".jsonl"):
                names.add(tail)
    return names


def complete_shards(out_dir, names, shard_size: int) -> set:  # noqa: ANN001
    """Of the already-downloaded shard files `names`, the ones holding
    exactly `shard_size` records -- the ONLY shards a resume may skip.
    A shard with fewer records is the trailing partial shard of an earlier
    (shorter or interrupted) run: build_corpus skips every doc whose target
    shard name is in existing_shards, so treating a partial shard as
    complete would silently DROP the docs a larger resume numbers into its
    unfilled tail. It is re-encoded and overwritten instead."""
    out_dir = Path(out_dir)
    full = set()
    for name in names:
        p = out_dir / name
        if p.exists():
            n = sum(1 for line in p.read_text().splitlines() if line.strip())
            if n == shard_size:
                full.add(name)
    return full


def _download_existing_shards(out_repo: str, out_subdir: str, names, out_dir) -> None:  # noqa: ANN001
    """Pulls each already-pushed shard down to `out_dir` so _tally_shards
    (and a possible later resume) can read it without re-encoding -- fails
    loud (hf_hub_download raises) rather than silently under-counting a
    resumed run's totals."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        p = hf_hub_download(out_repo, f"{out_subdir}/{name}")
        (out_dir / name).write_bytes(Path(p).read_bytes())


# ── C. build_corpus: filter -> encode -> shard, resumable ──────────────────


def build_corpus(
    sources, excl: dict, gtok, shard_size: int, out_dir, existing_shards, on_shard=None
) -> dict:  # noqa: ANN001
    """`sources` = [(src_name, (question, solution) iterable, splitter), ...]
    -> counts {"docs_seen": {src: n}, "excluded": {reason: n}, "cache_hits",
    "cache_misses"} (docs_kept/steps/groups/split_steps are NOT returned here
    -- read them back from the shard files themselves via _tally_shards,
    since a resumed run's skipped shards never pass through this function's
    own counters; see main()).

    Every doc that clears exclude_doc + answer extraction is numbered
    kept_index // shard_size into its target shard, REGARDLESS of whether
    that shard is in `existing_shards` -- so resuming with a larger source
    (20k -> 57k solutions) advances numbering exactly as if the whole run had
    gone in one pass, and only the NEW docs (in a fresh or a partially-filled
    shard) are ever re-encoded. `on_shard(name, path)` fires once a shard's
    file is written (the caller pushes it there, matching
    run_gist_dict.build_all_dicts' on_built convention)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict = {"docs_seen": {}, "excluded": {}}
    buffer: list[dict] = []
    buffer_shard_idx = 0
    kept = 0

    def _flush(idx: int) -> None:
        nonlocal buffer
        if not buffer:
            return
        name = f"shard_{idx:04d}.jsonl"
        path = out_dir / name
        write_jsonl(buffer, path)
        buffer = []
        if on_shard is not None:
            on_shard(name, path)

    for src, doc_iter, splitter in sources:
        counts["docs_seen"].setdefault(src, 0)
        for i, (question, solution) in enumerate(doc_iter):
            counts["docs_seen"][src] += 1
            reason = exclude_doc(question, solution, {**excl, "splitter": splitter})
            answer = None
            if reason is None:
                answer = extract_answer(src, solution)
                if answer is None:
                    reason = "no_answer"
            if reason is not None:
                counts["excluded"][reason] = counts["excluded"].get(reason, 0) + 1
                continue

            target_idx = kept // shard_size
            if target_idx != buffer_shard_idx:
                _flush(buffer_shard_idx)
                buffer_shard_idx = target_idx
            target_name = f"shard_{target_idx:04d}.jsonl"
            kept += 1
            if target_name in existing_shards:
                continue  # already completed by an earlier run -- number past it, don't re-encode

            steps = splitter(solution)
            ids = [gtok.encode_step(s) for s in steps]
            buffer.append(
                {
                    "src": src,
                    "doc_id": f"{src}_{i}",
                    "question": question,
                    "steps": steps,
                    "ids": ids,
                    "answer": answer,
                    "n_groups": sum(len(g) for g in ids),
                }
            )
    _flush(buffer_shard_idx)

    counts["cache_hits"] = gtok.hits
    counts["cache_misses"] = gtok.misses
    return counts


# ── B. eval-adjacent question sets to exclude ───────────────────────────────


def _eval_gsm8k_full_docs(doc_keys, n_candidates: int, dataset: str, smoke: bool) -> list[dict]:  # noqa: ANN001
    """(question, whole step list, raw solution text) for exactly the docs
    build_eval_set's 200 GSM8K-test eval steps were drawn from -- re-streams
    the SAME candidate window build_eval_set saw (same order, same cutoff),
    costing one extra pass over that window, never a second download. Feeds
    both the contamination filter (build_eval_set never captures the
    question) and eval_gsm8k_test.jsonl's whole-solution context."""
    if smoke:
        from marker.run_stage2 import _smoke_cot_texts  # noqa: PLC0415

        out = []
        for i, text in enumerate(_smoke_cot_texts(n_candidates)):
            if ("gsm8k_test_smoke", i) in doc_keys:
                out.append(
                    {
                        "question": f"smoke eval question {i}",
                        "steps": split_solution_steps(text),
                        "answer_raw": text,
                    }
                )
        return out

    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset, "main", split="test", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i >= n_candidates:
            break
        if ("gsm8k_test", i) in doc_keys:
            answer_raw = row.get("answer") or ""
            out.append(
                {
                    "question": row.get("question") or "",
                    "steps": split_solution_steps(answer_raw),
                    "answer_raw": answer_raw,
                }
            )
    return out


def _gsm8k_test_questions(dataset: str, smoke: bool) -> list[str]:
    """EVERY GSM8K test question (all 1319) -- the contamination filter
    excludes any OpenR1 item whose normalized question matches ANY of them
    (OpenR1 contains GSM8K-derived items, sometimes reformatted). This is a
    strict superset of the two eval-adjacent sets (the 200 stage-1b eval
    docs and the summary-probe run's 993 >=3-step docs are both drawn from
    this same test split), costs one streaming pass, and removes any
    dependence on reproducing those runs' filters exactly. --smoke has no
    real test split to stream offline, so it returns none; reason (b)'s own
    mechanics are unit-tested directly against exclude_doc
    (test_run_tokenize_corpus.py)."""
    if smoke:
        return []
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset, "main", split="test", streaming=True)
    return [row.get("question") or "" for row in ds]


# ── dictionary loading (real run vs --smoke) ────────────────────────────────


def _load_dict(repo: str, dict_subdir: str, dict_name: str, cache_dir):  # noqa: ANN001
    assert repo, "--repo is required to fetch the stage-1 dictionary (dict_<name>.pt)"
    local = Path(cache_dir) / "dicts" / f"dict_{dict_name}.pt"
    if local.exists():
        return torch.load(local, map_location="cpu")
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    p = hf_hub_download(repo, f"{dict_subdir}/dict_{dict_name}.pt")
    return torch.load(p, map_location="cpu")


def _build_smoke_dict(pm, gist, tok, base: int, max_span: int, K: int = 8, seed: int = 0) -> dict:  # noqa: ANN001
    """A random 'kv' dictionary sized to THIS model's real geometry (probed
    via one canonical encode -- never hardcoded, matching run_gist_dict's own
    geometry-from-probe convention) -- --smoke has no real dict_kv_K4096.pt
    to download."""
    probe_ids = tok("hi", add_special_tokens=False).input_ids
    kv, readout, _cs = encode_canonical(pm, gist, probe_ids, base=base, max_span=max_span)
    d = kv_slot_matrix(kv).shape[1]
    k_slots = kv.keys[0].shape[2]
    g = torch.Generator().manual_seed(seed)
    slots = [
        {
            "centroids": torch.randn(K, d, generator=g),
            "mu_readout": torch.randn(K, readout.shape[-1], generator=g),
            "usage": torch.zeros(K, dtype=torch.long),
            "K": K,
            "seed": seed,
        }
        for _ in range(k_slots)
    ]
    geometry = {
        "n_layers": kv.n_layers,
        "n_kv_heads": kv.keys[0].shape[1],
        "head_dim": kv.keys[0].shape[3],
        "k_slots": k_slots,
    }
    return {"cfg": f"kv_K{K}", "kind": "kv", "geometry": geometry, "slots": slots}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None, help="stage-1 gist adapter + dictionary repo")
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dict-subdir", default="gist_dict")
    ap.add_argument("--dict-name", default="kv_K4096")
    ap.add_argument("--gsm8k-dataset", default="openai/gsm8k")
    ap.add_argument("--openr1-dataset", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--n-gsm8k", type=int, default=7473, help="GSM8K train candidates (7473 = all)")
    ap.add_argument("--n-openr1", type=int, default=12500)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--seq-cap", type=int, default=512)
    ap.add_argument("--max-groups-per-step", type=int, default=4)
    ap.add_argument("--max-span", type=int, default=MAX_SPAN)
    ap.add_argument("--base", type=int, default=BASE)
    ap.add_argument(
        "--n-eval-gsm8k",
        type=int,
        default=200,
        help="must match the stage-1b eval run's --n-eval so build_eval_set "
        "reproduces the exact same 200 GSM8K-test docs",
    )
    ap.add_argument("--cache-dir", default="/tmp/gist_corpus_cache")  # noqa: S108
    ap.add_argument("--out-subdir", default="gist_corpus_K4096")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo = "Qwen/Qwen2.5-0.5B", None
        args.n_gsm8k, args.n_openr1 = 15, 15
        args.shard_size = 16
        args.n_eval_gsm8k = 4
        args.cache_dir = "/tmp/gist_corpus_cache_smoke"  # noqa: S108

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.run_stage2 import _doc_texts_qa, _load_stage1, _split_units  # noqa: PLC0415

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    pm.set_adapter("default")

    dict_ = (
        _build_smoke_dict(pm, gist, tok, args.base, args.max_span)
        if args.smoke
        else _load_dict(args.repo, args.dict_subdir, args.dict_name, args.cache_dir)
    )
    gtok = GistTokenizer(
        pm,
        gist,
        dict_,
        tok,
        base=args.base,
        max_span=args.max_span,
        cache_dir=str(Path(args.cache_dir) / "tok_cache"),
    )

    # ── B. eval sets: reproduce the stage-1b 200-doc GSM8K-test eval set
    # (never re-derived by hand -- build_eval_set IS the stage-1 definition)
    # and the summary-probe's 993-doc test set, for the contamination filter
    eval_steps, eval_doc_keys = build_eval_set(
        _gsm8k_docs("test", args.n_eval_gsm8k * 3, args.gsm8k_dataset, smoke=args.smoke),
        tok,
        args.max_span,
        args.n_eval_gsm8k,
    )
    eval_full_docs = _eval_gsm8k_full_docs(
        eval_doc_keys, args.n_eval_gsm8k * 3, args.gsm8k_dataset, args.smoke
    )
    test_questions = _gsm8k_test_questions(args.gsm8k_dataset, args.smoke)
    norm_questions = {normalize_question(d["question"]) for d in eval_full_docs} | {
        normalize_question(q) for q in test_questions
    }
    print(
        f"exclusion set: {len(eval_full_docs)} stage-1b eval docs + ALL "
        f"{len(test_questions)} GSM8K test questions -> {len(norm_questions)} "
        f"distinct normalized questions",
        flush=True,
    )

    excl = {
        "tok": tok,
        "max_span": args.max_span,
        "max_groups": args.max_groups_per_step,
        "seq_cap": args.seq_cap,
        "norm_questions": norm_questions,
        "k_slots": dict_["geometry"]["k_slots"],
    }

    if args.smoke:
        from marker.run_summary_probe import (  # noqa: PLC0415
            _smoke_questions,
            _smoke_varied_cot_texts,
        )

        qs, sols = _smoke_questions(30), _smoke_varied_cot_texts(30)
        half = 15
        sources = [
            ("gsm8k", list(zip(qs[:half], sols[:half], strict=True)), split_solution_steps),
            (
                "openr1",
                list(zip(qs[half:], sols[half:], strict=True)),
                lambda s: _split_units(s, "sentence"),
            ),
        ]
    else:
        # train-only streams -- GSM8K TEST never enters (exclusion reason a)
        sources = [
            (
                "gsm8k",
                _doc_texts_qa(args.n_gsm8k, dataset=args.gsm8k_dataset),
                split_solution_steps,
            ),
            (
                "openr1",
                _doc_texts_qa(args.n_openr1, dataset=args.openr1_dataset),
                lambda s: _split_units(s, "sentence"),
            ),
        ]

    out_dir = Path(args.cache_dir) / "shards"
    existing_shards = (
        set()
        if (args.smoke or not args.out_repo)
        else list_existing_shards(args.out_repo, args.out_subdir)
    )
    if existing_shards:
        _download_existing_shards(args.out_repo, args.out_subdir, existing_shards, out_dir)
        full = complete_shards(out_dir, existing_shards, args.shard_size)
        partial = sorted(existing_shards - full)
        existing_shards = full
        print(
            f"[TOKENIZE_CORPUS] resume: {len(full)} complete shards already on "
            f"{args.out_repo}/{args.out_subdir}"
            + (f"; re-encoding partial {', '.join(partial)}" if partial else ""),
            flush=True,
        )

    manifest: dict = {
        "dict_sha1": gtok._dict_sha1,  # noqa: SLF001 -- this module owns gtok, not a layering violation
        "dict_name": args.dict_name,
        "model": args.model_name,
        "base": args.base,
        "max_span": args.max_span,
        "batching": "single",
        "slot_order_note": "ids stored in natural order",
        "seq_cap": args.seq_cap,
        "max_groups_per_step": args.max_groups_per_step,
        "shard_size": args.shard_size,
        "n_gsm8k": args.n_gsm8k,
        "n_openr1": args.n_openr1,
        "datasets": {"gsm8k": args.gsm8k_dataset, "openr1": args.openr1_dataset},
        "dataset_revisions": None
        if args.smoke
        else {
            "gsm8k": _dataset_revision(args.gsm8k_dataset),
            "openr1": _dataset_revision(args.openr1_dataset),
        },
        "n_excluded_questions": len(norm_questions),
    }

    def _push_snapshot() -> None:
        if not args.smoke and args.out_repo:
            _push_with_retry(args.out_repo, str(out_dir), args.out_subdir)
        manifest_dir = Path(args.cache_dir) / "manifest_out"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        if not args.smoke and args.out_repo:
            _push_with_retry(args.out_repo, str(manifest_dir), args.out_subdir)

    def _on_shard(name, path) -> None:  # noqa: ANN001, ARG001
        manifest["counts"] = {
            **manifest.get("counts", {}),
            **_tally_shards(sorted(out_dir.glob("shard_*.jsonl"))),
        }
        _push_snapshot()
        print(f"[TOKENIZE_CORPUS PARTIAL] shard {name} done", flush=True)

    run_counts = build_corpus(
        sources, excl, gtok, args.shard_size, out_dir, existing_shards, on_shard=_on_shard
    )
    manifest["counts"] = {
        **run_counts,
        **_tally_shards(sorted(out_dir.glob("shard_*.jsonl"))),
    }

    # ── D. eval_gsm8k_test.jsonl: the 200 eval docs' WHOLE solutions ────────
    # build_eval_set selects docs in STREAM order and _eval_gsm8k_full_docs
    # re-walks the same window in the same order, so entry i of each list is
    # the same doc -- eval_step_index marks WHICH step of the whole solution
    # is the stage-1b scored one. `.index` finds the first occurrence, which
    # IS the selected one: build_eval_set takes the first passing step, and
    # an identical earlier text would have passed identically.
    eval_records = []
    for i, (doc, step) in enumerate(zip(eval_full_docs, eval_steps, strict=True)):
        eval_step_index = doc["steps"].index(step.text)
        ids = [gtok.encode_step(s) for s in doc["steps"]]
        eval_records.append(
            {
                "src": "gsm8k",
                "doc_id": f"gsm8k_eval_{i}",
                "question": doc["question"],
                "steps": doc["steps"],
                "ids": ids,
                "answer": extract_answer("gsm8k", doc["answer_raw"]),
                "n_groups": sum(len(g) for g in ids),
                "eval_step_index": eval_step_index,
            }
        )
    write_jsonl(eval_records, out_dir / "eval_gsm8k_test.jsonl")
    manifest["n_eval_records"] = len(eval_records)

    manifest["counts"]["cache_hits"] = gtok.hits
    manifest["counts"]["cache_misses"] = gtok.misses
    _push_snapshot()
    print(f"[TOKENIZE_CORPUS MANIFEST] {json.dumps(manifest, default=str)}", flush=True)


if __name__ == "__main__":
    main()
