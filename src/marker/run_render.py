"""Stage-3 render training: teach a small decoder to reconstruct a step's text
from its thought (see STAGE2_PLAN "Direction reset after 3b").

The frozen Stage-1 encoder ('default' adapter) makes thoughts; a trainable
'render' LoRA learns to decode the SOURCE step back out (reconstruction, not
continuation). Reports beyond averages (user requirement): reconstruction F1
quantiles AND number-token recall (the exact-literals slice the ledger will
later make deterministic).

BILINGUAL READER (--kv-source mixed): the render LoRA above is trained on one
dialect only -- encoder gist_kv. The converter bridge_validated speaks a
SECOND dialect (its own KV, not gist_kv's distribution) that the same reader
reads badly today (0.16 vs 0.92 relations-exact). bridge_validated is lossless
for next-step likelihood; reconstruction fidelity is what this run tests. This
run teaches ONE reader BOTH dialects -- a bilingual reader, not a swap -- by
warm-starting from the existing (encoder-only) reader and mixing bridge KV
into training alongside gist KV, so it keeps 0.92 and gains the converter.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_render \
        --repo mattyvee/mimir-artifacts --n-docs 800 --steps 2000
    # bilingual reader (real invocation -- see scripts/vast_render.sh):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_render \
        --repo mattyvee/mimir-artifacts --out-repo mattyvee/mimir-artifacts \
        --ledger --kv-source mixed --warm-start-subdir render_adapter_ledger \
        --out-subdir render_adapter_oneform --steps 3000 --lr 1e-4
Smoke (local tiny model):
    PYTHONPATH=src python -m marker.run_render --smoke
    PYTHONPATH=src python -m marker.run_render --smoke --kv-source mixed
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import NamedTuple

import torch

from marker.gist_model import encode_gist, gist_kv
from marker.gistprobe import relation_score
from marker.predprobe import bridged_condition, pick_cross_doc_step
from marker.render import attach_render, extract_ledger, ledger_render_nll, render_nll

_NUM = re.compile(r"\d+")

# baseline-mirage guard thresholds (Fable spec): a warm-started reader must
# already be good at gist and CANNOT already read bridge KV well, or an
# "improved bridge dialect" result would just mean it was bilingual already.
BASELINE_GIST_MIN = 0.80
BASELINE_BRIDGE_MAX = 0.40
BASELINE_WRONG_MAX = 0.30

# subdirs a bilingual-reader run (warm-started and/or mixed) must never
# overwrite, regardless of --out-subdir's computed default.
_ALWAYS_PROTECTED = {"render_adapter", "render_adapter_ledger"}


class Record(NamedTuple):
    """One (thought, step) pair, encoded under both dialects when a bridge is
    given. ids/text are shared by construction -- both dialects reconstruct
    the SAME source span, only the injected KV differs."""

    doc_i: int
    kv_gist: object
    cs_gist: int
    kv_bridge: object | None
    cs_bridge: int | None
    ids: list[int]
    text: str


def _cpu_kv(kv):  # noqa: ANN001
    """Move an AxiomKV's tensors to CPU for storage (many records precomputed
    up front; the frozen encoder/bridge never need to keep them on GPU)."""
    return type(kv)(kv.n_layers, [k.cpu() for k in kv.keys], [v.cpu() for v in kv.values])


def build_records(pm, gist, tok, docs, unit, max_span, bridge=None, kv_dtype=None):  # noqa: ANN001
    """docs -> list[Record] + doc_starts (record index where each doc begins,
    for the doc-boundary eval split -- same convention as today's _encode_docs
    even when a doc yields zero spans, so split indices don't shift).

    gist path: exactly today's gist_kv(pm, gist, ids) per span. bridge path
    (when bridge given): per DOC, batch the spans through encode_gist (NOT
    span-by-span -- padding differs numerically), then bridged_condition per
    span slot. Both dialects of a record share ids/text by construction.
    Requires the 'default' (frozen encoder) adapter; sets it active itself so
    callers don't need to remember to."""
    from marker.run_stage2 import _split_units  # noqa: PLC0415

    pm.set_adapter("default")
    records: list[Record] = []
    doc_starts: list[int] = []
    for doc_i, text in enumerate(docs):
        doc_starts.append(len(records))
        spans = []
        for s in _split_units(text, unit):
            ids = tok(s, add_special_tokens=False).input_ids[:max_span]
            if len(ids) < 2:
                continue
            spans.append((ids, s))
        if not spans:
            continue
        summ = (
            encode_gist(pm, gist, [ids for ids, _ in spans]).float() if bridge is not None else None
        )
        for j, (ids, s) in enumerate(spans):
            kv, cont_start, _ = gist_kv(pm, gist, ids)
            kv_gist = _cpu_kv(kv)
            if bridge is not None:
                kv_b, cs_b = bridged_condition(bridge, summ[j], kv_dtype)
                kv_bridge, cs_bridge = _cpu_kv(kv_b), cs_b
            else:
                kv_bridge, cs_bridge = None, None
            records.append(Record(doc_i, kv_gist, cont_start, kv_bridge, cs_bridge, ids, s))
    return records, doc_starts


def training_items(records, kv_source):  # noqa: ANN001
    """(record, dialect) training pool. 'gist': one 'gist' item per record --
    today's only path, so torch.randperm over this list reproduces today's
    draw order bitwise. 'mixed': each record ALSO contributes a 'bridge' item
    -- both dialects present once each (50/50 weight in the pool), never a
    per-item coin flip."""
    items = [(r, "gist") for r in records]
    if kv_source == "mixed":
        items += [(r, "bridge") for r in records]
    return items


def _pick_wrong_bridged(pool, gen):  # noqa: ANN001
    """For each record in pool, the wrong_bridged pairing: another record's
    bridge KV from a DIFFERENT doc_i (marker.predprobe.pick_cross_doc_step),
    scored later against the CURRENT record's own text -- the cheating floor.
    Pure index logic (no model calls), so it's testable without a bridge/KV."""
    by_doc: dict[int, list[int]] = {}
    for i, r in enumerate(pool):
        by_doc.setdefault(r.doc_i, []).append(i)
    doc_ids = sorted(by_doc)
    doc_lengths = [len(by_doc[d]) for d in doc_ids]
    doc_pos = {d: i for i, d in enumerate(doc_ids)}
    out = []
    for idx, r in enumerate(pool):
        di = doc_pos[r.doc_i]
        local_step = by_doc[r.doc_i].index(idx)
        dj, sj = pick_cross_doc_step(di, doc_lengths, gen, step_idx=local_step)
        out.append(pool[by_doc[doc_ids[dj]][sj]])
    return out


def check_out_subdir(out_subdir, warm_start_subdir, bridge_subdir, kv_source):  # noqa: ANN001
    """Clobber guard: a bilingual-reader run (warm-started and/or mixed
    kv-source) must never write over render_adapter / render_adapter_ledger,
    its own warm-start source, or the bridge checkpoint -- the whole point of
    --out-subdir is a NEW name (e.g. render_adapter_oneform). The plain
    gist-only, non-warm-start default is exempt: it legitimately still writes
    render_adapter[_ledger], today's behaviour, reproduced bitwise."""
    if not (warm_start_subdir or kv_source == "mixed"):
        return
    forbidden = _ALWAYS_PROTECTED | {warm_start_subdir, bridge_subdir}
    assert out_subdir not in forbidden, (
        f"--out-subdir={out_subdir!r} would clobber one of {forbidden - {None}} "
        "-- a bilingual-reader run must write to a NEW subdir"
    )


def baseline_ok(gist_rel: float, bridge_rel: float, wrong_rel: float) -> bool:
    """Warm-start baseline-mirage guard: the LOADED reader must already be
    good at gist and CANNOT already read bridge KV well -- otherwise any
    'bridge dialect improved' result at the end would be a mirage (an
    already-bilingual base, not something this run taught it)."""
    return (
        gist_rel >= BASELINE_GIST_MIN
        and bridge_rel <= BASELINE_BRIDGE_MAX
        and wrong_rel <= BASELINE_WRONG_MAX
    )


def warm_start_render(pm, loc, sub):  # noqa: ANN001
    """Load a PREVIOUSLY-TRAINED render LoRA as the starting point for a new
    run, instead of attach_render's fresh init -- the whole point of
    warm-starting is to KEEP the existing weights. loc is a local directory
    (already snapshot_download'ed) holding f"{sub}/render/*"."""
    pm.load_adapter(f"{loc}/{sub}/render", adapter_name="render", is_trainable=True)
    pm.set_adapter("render")
    return [(n, p) for n, p in pm.named_parameters() if "render" in n and p.requires_grad]


def _ledger_ids(tok, text):  # noqa: ANN001
    """The step's literals as visible tokens, newline-terminated (the \\n
    marks end-of-ledger; generation stops at its OWN later newline)."""
    nums = extract_ledger(text)
    return tok(" ".join(nums) + "\n", add_special_tokens=False).input_ids if nums else []


def _score_record(pm, tok, gold_rec, kv, cs, use_ledger, stop_ids, max_new):  # noqa: ANN001
    """Render `kv`/`cs` and score against gold_rec's OWN text/ids/ledger --
    the injected KV may come from a different record (wrong_bridged); the
    thing being reconstructed towards never does. max_new mirrors
    run_predprobe (args.max_span, newline-stop) -- the baseline-mirage
    thresholds were calibrated against probe numbers, so the generation
    budget must match or a tighter cap truncates relations and fires
    BASELINE_MIRAGE spuriously."""
    prefix = _ledger_ids(tok, gold_rec.text) if use_ledger else None
    rec_ids = _render_reconstruct(pm, kv, cs, gold_rec.ids[0], max_new, stop_ids, prefix_ids=prefix)
    rtext = tok.decode(rec_ids)
    f1 = _f1_tok(rec_ids, gold_rec.ids)
    nr = _num_recall(rtext, gold_rec.text)
    rel = relation_score(rtext, gold_rec.text)["exact"]
    return f1, nr, rel


def _summarize_scores(triples):
    f1s = [f for f, _, _ in triples]
    nrs = [n for _, n, _ in triples if n is not None]
    rels = [r for _, _, r in triples if r is not None]
    return {
        "rel_exact": round(sum(rels) / len(rels), 3) if rels else 0.0,
        "f1_mean": round(sum(f1s) / max(1, len(f1s)), 3),
        "num_recall": round(sum(nrs) / len(nrs), 3) if nrs else 0.0,
        "n": len(triples),
    }


@torch.no_grad()
def eval_dialects(pm, tok, records, cap, bridge_present, ledger, gen, max_new=64):  # noqa: ANN001
    """Per-dialect eval (replaces the old F1-quantile-only eval). Returns:
    - gist: reader on kv_gist.
    - bridge: reader on kv_bridge (only when bridge_present).
    - wrong_bridged: reader on another doc's kv_bridge, scored against THIS
      record's text -- the cheating floor (only when bridge_present).
    - bridge_no_ledger: bridge dialect with the ledger prefix withheld -- the
      non-gating honest preview (only when bridge_present and ledger).
    - margin: bridge.rel_exact - wrong_bridged.rel_exact (only when
      bridge_present). Read the margin, never bridge's raw rel_exact alone --
      the ledger flatters every condition's absolute number the same way.
    Report each dialect separately -- never pool across dialects."""
    pool = records[:cap]
    stop_ids = {t for t in (tok("\n", add_special_tokens=False).input_ids or []) if t}

    out = {
        "gist": _summarize_scores(
            [
                _score_record(pm, tok, r, r.kv_gist, r.cs_gist, ledger, stop_ids, max_new)
                for r in pool
            ]
        )
    }
    if bridge_present:
        out["bridge"] = _summarize_scores(
            [
                _score_record(pm, tok, r, r.kv_bridge, r.cs_bridge, ledger, stop_ids, max_new)
                for r in pool
            ]
        )
        wrong = _pick_wrong_bridged(pool, gen)
        out["wrong_bridged"] = _summarize_scores(
            [
                _score_record(pm, tok, r, w.kv_bridge, w.cs_bridge, ledger, stop_ids, max_new)
                for r, w in zip(pool, wrong, strict=True)
            ]
        )
        out["margin"] = round(out["bridge"]["rel_exact"] - out["wrong_bridged"]["rel_exact"], 3)
        if ledger:
            out["bridge_no_ledger"] = _summarize_scores(
                [
                    _score_record(pm, tok, r, r.kv_bridge, r.cs_bridge, False, stop_ids, max_new)
                    for r in pool
                ]
            )
    return out


def _f1_tok(pred, gold):  # noqa: ANN001
    from collections import Counter  # noqa: PLC0415

    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    o = sum((pc & gc).values())
    if o == 0:
        return 0.0
    p, r = o / len(pred), o / len(gold)
    return 2 * p * r / (p + r)


def _num_recall(pred_text: str, gold_text: str) -> float | None:
    """Fraction of the gold step's numbers that appear in the reconstruction.
    None when the gold step has no numbers (excluded from the mean)."""
    gold_nums = _NUM.findall(gold_text)
    if not gold_nums:
        return None
    pred_nums = set(_NUM.findall(pred_text))
    return sum(1 for g in gold_nums if g in pred_nums) / len(gold_nums)


@torch.no_grad()
def _render_reconstruct(pm, thought_kv, cont_start, first_tok, max_new, stop_ids, prefix_ids=None):  # noqa: ANN001
    """Greedy reconstruct a span from its thought under the active (render)
    adapter, primed with the true first token (the thought + a seed -> the
    step; a ledger/prefix would supply the seed at runtime). prefix_ids (the
    visible literals ledger) are fed before the first token and NOT counted as
    output."""
    from marker.run_axiom_mlp_demo import _build_dynamic_cache  # noqa: PLC0415

    device = next(pm.parameters()).device
    cache = _build_dynamic_cache(thought_kv, device)
    pos0 = cont_start
    if prefix_ids:
        out = pm(
            torch.tensor([prefix_ids], device=device),
            past_key_values=cache,
            position_ids=torch.arange(pos0, pos0 + len(prefix_ids), device=device).unsqueeze(0),
            use_cache=True,
        )
        cache = out.past_key_values
        pos0 += len(prefix_ids)
    gen = [first_tok]
    nxt = first_tok
    for j in range(max_new - 1):
        out = pm(
            torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            position_ids=torch.tensor([[pos0 + j]], device=device),
            use_cache=True,
        )
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        if nxt in stop_ids:
            break
        gen.append(nxt)
    return gen


def _push_with_retry(repo_id, folder, path_in_repo):  # noqa: ANN001
    """upload_folder with a 3-try retry, 20s apart -- a flaky node's network
    blip must not silently lose a manifest push. Raises after the 3rd failure
    (fail loud: a retry-then-raise, never a swallow-and-continue)."""
    from huggingface_hub import upload_folder  # noqa: PLC0415

    for attempt in range(1, 4):
        try:
            upload_folder(repo_id=repo_id, folder_path=folder, path_in_repo=path_in_repo)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"[PUSH RETRY {attempt}/3] {e}", flush=True)
            time.sleep(20)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out-repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--unit", choices=["line", "sentence"], default="line")
    ap.add_argument("--n-docs", type=int, default=800)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument(
        "--ledger",
        action="store_true",
        help="literals ledger: feed the step's exact numbers as a visible prefix "
        "(train + eval) — targets number-recall ~1.0 vs the no-ledger baseline. "
        "REQUIRED for the real bilingual-reader run (see module docstring).",
    )
    ap.add_argument(
        "--kv-source",
        choices=["gist", "mixed"],
        default="gist",
        help="'gist' (default, today's behaviour): encoder-only reader. 'mixed': "
        "also train on bridge_validated KV -- the bilingual reader. 'bridge'-only "
        "is not offered: the probe's gate 0 re-scores encoder KV through this same "
        "adapter, and the memory lane reads encoder KV.",
    )
    ap.add_argument("--bridge-subdir", default="bridge_validated")
    ap.add_argument(
        "--warm-start-subdir",
        default=None,
        help="load an existing render LoRA (e.g. render_adapter_ledger) as the "
        "starting point instead of attach_render's fresh init; --r is ignored",
    )
    ap.add_argument("--out-subdir", default=None)
    ap.add_argument(
        "--eval-cap", type=int, default=150, help="max records evaluated per dialect per eval"
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model_name, args.repo, args.n_docs, args.steps = "Qwen/Qwen2.5-0.5B", None, 20, 60

    args.out_subdir = args.out_subdir or (
        "render_adapter_ledger" if args.ledger else "render_adapter"
    )
    if args.out_repo:  # the guard protects a REMOTE artifact; nothing to protect locally
        check_out_subdir(
            args.out_subdir, args.warm_start_subdir, args.bridge_subdir, args.kv_source
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.run_stage2 import (  # noqa: PLC0415
        _doc_texts,
        _load_stage1,
        _smoke_cot_texts,
    )

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )

    # ── the render decoder: fresh LoRA, or warm-started from an existing one ──
    if args.warm_start_subdir:
        print(
            f"warm-start: --r={args.r} ignored (loading the existing render LoRA's own rank)",
            flush=True,
        )
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        loc = snapshot_download(
            repo_id=args.repo, allow_patterns=[f"{args.warm_start_subdir}/render/*"]
        )
        render_params = warm_start_render(pm, loc, args.warm_start_subdir)
    else:
        render_params = attach_render(pm, r=args.r)

    # ── the converter dialect (mixed only): frozen bridge_validated, loaded
    # exactly as run_predprobe.py's Path-B wiring (same throwaway gist_kv for
    # kv_dtype, same _load_bridge) ────────────────────────────────────────────
    bridge, kv_dtype = None, None
    if args.kv_source == "mixed":
        probe_kv, _, _ = gist_kv(pm, gist, tok("hi", add_special_tokens=False).input_ids)
        kv_dtype = probe_kv.keys[0].dtype
        if args.smoke:
            from marker.bridge import GistBridge  # noqa: PLC0415

            bridge = (
                GistBridge(
                    d=gist.shape[-1],
                    k=gist.shape[0],
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

            from marker.run_rollout import _load_bridge  # noqa: PLC0415

            bridge = _load_bridge(
                torch.load(
                    hf_hub_download(args.repo, f"{args.bridge_subdir}/bridge.pt"),
                    map_location="cpu",
                ),
                gist.shape[-1],
                gist.shape[0],
                probe_kv,
                device,
            )
    bridge_present = bridge is not None

    # ── encode steps under both dialects with the FROZEN encoder/bridge ─────
    docs = (
        _smoke_cot_texts(args.n_docs)
        if args.smoke
        else list(_doc_texts(args.n_docs, "cot", args.dataset, None, None))
    )
    records, doc_starts = build_records(
        pm, gist, tok, docs, args.unit, args.max_span, bridge=bridge, kv_dtype=kv_dtype
    )
    print(f"encoded {len(records)} records", flush=True)
    # eval split at a DOCUMENT boundary (Fable render review: a step-index split
    # straddles one doc, letting sibling steps leak train->eval style) --
    # BEFORE any dialect doubling (training_items), so no eval record's target
    # leaks into training under either dialect.
    want = max(2, len(records) // 10)
    n_eval = next((ds for ds in doc_starts if ds >= want), want)
    eval_records, train_records = records[:n_eval], records[n_eval:]
    # FRESH eval set (Fable memorization confound): our own synthetic step
    # templates — never on the internet, cannot be memorized. Reported
    # separately; the gsm8k eval may be flattered by pretraining recall.
    fresh_records, _ = (
        build_records(
            pm,
            gist,
            tok,
            _smoke_cot_texts(30),
            args.unit,
            args.max_span,
            bridge=bridge,
            kv_dtype=kv_dtype,
        )
        if not args.smoke
        else ([], None)
    )

    pm.set_adapter("render")

    def _run_evals():  # noqa: ANN202
        out = {
            "gsm8k": eval_dialects(
                pm,
                tok,
                eval_records,
                args.eval_cap,
                bridge_present,
                args.ledger,
                torch.Generator().manual_seed(0),
                max_new=args.max_span,
            )
        }
        if fresh_records:
            out["fresh_synth"] = eval_dialects(
                pm,
                tok,
                fresh_records,
                args.eval_cap,
                bridge_present,
                args.ledger,
                torch.Generator().manual_seed(1),
                max_new=args.max_span,
            )
        return out

    # ── step-0 eval: BEFORE any training (the warm-start baseline-mirage check
    # reads this; also reported for a before/after comparison) ──────────────
    step0 = _run_evals()
    print(f"\n[RENDER step0] {step0}", flush=True)

    if args.warm_start_subdir and not args.smoke and bridge_present:
        g = step0["gsm8k"]
        if not baseline_ok(
            g["gist"]["rel_exact"], g["bridge"]["rel_exact"], g["wrong_bridged"]["rel_exact"]
        ):
            print("BASELINE_MIRAGE", flush=True)
            sys.exit(2)

    manifest = {
        "ledger": bool(args.ledger),
        "kv_source": args.kv_source,
        "warm_start_subdir": args.warm_start_subdir,
        "bridge_subdir": args.bridge_subdir if bridge_present else None,
        "out_subdir": args.out_subdir,
        "eval_cap": args.eval_cap,
        "thresholds": {
            "baseline_gist_min": BASELINE_GIST_MIN,
            "baseline_bridge_max": BASELINE_BRIDGE_MAX,
            "baseline_wrong_max": BASELINE_WRONG_MAX,
        },
        "step0": step0,
    }

    from pathlib import Path  # noqa: PLC0415

    d = Path("/tmp/render_out")  # noqa: S108
    d.mkdir(parents=True, exist_ok=True)
    if args.out_repo:
        import json  # noqa: PLC0415

        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _push_with_retry(args.out_repo, str(d), args.out_subdir)
        print(f"[MANIFEST PUSHED step0] {args.out_repo}/{args.out_subdir}", flush=True)

    # ── train the render LoRA (render adapter already active) ────────────────
    nl = tok("\n", add_special_tokens=False).input_ids
    opt = torch.optim.AdamW([p for _, p in render_params], lr=args.lr, weight_decay=0.01)
    torch.manual_seed(0)
    step = 0
    train_items = training_items(train_records, args.kv_source)

    while step < args.steps:
        for idx in torch.randperm(len(train_items)):
            rec, dialect = train_items[idx]
            kv, cs = (
                (rec.kv_gist, rec.cs_gist) if dialect == "gist" else (rec.kv_bridge, rec.cs_bridge)
            )
            # target = step + newline: teach the decoder to STOP (the first
            # run repeated the step 3-4x to max_new — no end-of-step marker)
            tgt = list(rec.ids) + list(nl)
            if args.ledger:
                loss = ledger_render_nll(pm, kv, cs, _ledger_ids(tok, rec.text), tgt)
            else:
                loss = render_nll(pm, kv, cs, tgt)
            opt.zero_grad()
            loss.backward()
            if step == 0:
                # the quantized path is unexercised by CPU tests (pilot's GRAD
                # FAIL guard): a silent no-grad here would "train" nothing and
                # fake a render result — fail LOUDLY instead.
                ok = any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in render_params)
                assert ok, "GRAD FAIL: no gradient reached the render LoRA (quantized path)"
                print("GRAD_OK (render gradients flowing)", flush=True)
            opt.step()
            step += 1
            if step % (20 if args.smoke else 200) == 0:
                print(f"[step {step}] render nll {loss.item():.4f} ({dialect})", flush=True)
            if step >= args.steps:
                break

    # ── push the WEIGHTS immediately after training, BEFORE the (long) eval ──
    # A 4h run once died at the timeout during eval and the trained adapter died
    # with the node (nothing pushes until the end). Weights first, durably;
    # manifest follows after eval. (Unwrapped, as before -- only the manifest
    # pushes get the retry loop.)
    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        pm.save_pretrained(str(d), selected_adapters=["render"])
        upload_folder(repo_id=args.out_repo, folder_path=str(d), path_in_repo=args.out_subdir)
        print(f"[WEIGHTS PUSHED EARLY] {args.out_repo}/{args.out_subdir}", flush=True)

    # ── final eval: reconstruction fidelity, per dialect ─────────────────────
    final = _run_evals()
    print(f"\n[RENDER final] {final}", flush=True)
    print(
        "READ: gsm8k numbers may be flattered by pretraining memorization; "
        "FRESH-synth (our templates, unmemorizable) is the honest fidelity. "
        "Read the bridge/wrong_bridged MARGIN, never bridge's raw rel_exact alone "
        "-- the ledger flatters every condition's absolute number the same way."
    )
    manifest["final"] = final

    # persist results to a manifest — a flaky node's stderr spam once buried the
    # eval output past the log tail, and the node was gone before it was read.
    import json  # noqa: PLC0415

    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[RENDER MANIFEST] {json.dumps(manifest)}", flush=True)  # single-line, survives tail

    if args.out_repo:
        pm.save_pretrained(str(d), selected_adapters=["render"])
        _push_with_retry(args.out_repo, str(d), args.out_subdir)
        print(f"pushed render adapter + manifest to {args.out_repo}/{args.out_subdir}", flush=True)


if __name__ == "__main__":
    import os

    main()
    sys.stdout.flush()
    os._exit(0)
