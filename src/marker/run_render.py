"""Stage-3 render training: teach a small decoder to reconstruct a step's text
from its thought (see STAGE2_PLAN "Direction reset after 3b").

The frozen Stage-1 encoder ('default' adapter) makes thoughts; a trainable
'render' LoRA learns to decode the SOURCE step back out (reconstruction, not
continuation). Reports beyond averages (user requirement): reconstruction F1
quantiles AND number-token recall (the exact-literals slice the ledger will
later make deterministic).

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_render \
        --repo mattyvee/mimir-artifacts --n-docs 800 --steps 2000
Smoke (local tiny model):
    PYTHONPATH=src python -m marker.run_render --smoke
"""

from __future__ import annotations

import argparse
import re

import torch

from marker.gist_model import gist_kv
from marker.render import attach_render, render_nll

_NUM = re.compile(r"\d+")


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
def _render_reconstruct(pm, thought_kv, cont_start, first_tok, max_new, stop_ids):  # noqa: ANN001
    """Greedy reconstruct a span from its thought under the active (render)
    adapter, primed with the true first token (the thought + a seed -> the
    step; a ledger/prefix would supply the seed at runtime)."""
    from marker.run_axiom_mlp_demo import _build_dynamic_cache  # noqa: PLC0415

    device = next(pm.parameters()).device
    cache = _build_dynamic_cache(thought_kv, device)
    gen = [first_tok]
    nxt = first_tok
    for j in range(max_new - 1):
        out = pm(
            torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            position_ids=torch.tensor([[cont_start + j]], device=device),
            use_cache=True,
        )
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        if nxt in stop_ids:
            break
        gen.append(nxt)
    return gen


def main() -> None:
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model_name, args.repo, args.n_docs, args.steps = "Qwen/Qwen2.5-0.5B", None, 20, 60

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.run_stage2 import (  # noqa: PLC0415
        _doc_texts,
        _load_stage1,
        _smoke_cot_texts,
        _split_units,
    )

    pm, gist, tok = _load_stage1(
        args.model_name, args.repo, device, device == "cuda" and not args.smoke
    )
    stop_ids = {t for t in (tok("\n", add_special_tokens=False).input_ids or []) if t}
    render_params = attach_render(pm, r=args.r)

    # ── encode steps -> (thought KV on CPU, span) with the FROZEN encoder ─────
    pm.set_adapter("default")
    docs = (
        _smoke_cot_texts(args.n_docs)
        if args.smoke
        else list(_doc_texts(args.n_docs, "cot", args.dataset, None, None))
    )
    pairs = []
    doc_starts = []  # first pair-index of each doc, for a DOC-boundary eval split
    for text in docs:
        doc_starts.append(len(pairs))
        for s in _split_units(text, args.unit):
            ids = tok(s, add_special_tokens=False).input_ids[: args.max_span]
            if len(ids) < 2:
                continue
            kv, cont_start, _ = gist_kv(pm, gist, ids)
            cpu_kv = type(kv)(kv.n_layers, [k.cpu() for k in kv.keys], [v.cpu() for v in kv.values])
            pairs.append((cpu_kv, cont_start, ids, s))
    print(f"encoded {len(pairs)} (thought, step) pairs", flush=True)
    # eval split at a DOCUMENT boundary (Fable render review: a step-index split
    # straddles one doc, letting sibling steps leak train->eval style)
    want = max(2, len(pairs) // 10)
    n_eval = next((ds for ds in doc_starts if ds >= want), want)
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]

    def _gpu(kv):  # noqa: ANN001
        return type(kv)(
            kv.n_layers, [k.to(device) for k in kv.keys], [v.to(device) for v in kv.values]
        )

    # ── train the render LoRA (render adapter active) ────────────────────────
    pm.set_adapter("render")
    opt = torch.optim.AdamW([p for _, p in render_params], lr=args.lr, weight_decay=0.01)
    torch.manual_seed(0)
    step = 0
    while step < args.steps:
        for idx in torch.randperm(len(train_pairs)):
            kv, cs, ids, _ = train_pairs[idx]
            loss = render_nll(pm, _gpu(kv), cs, ids)
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
                print(f"[step {step}] render nll {loss.item():.4f}", flush=True)
            if step >= args.steps:
                break

    # ── eval: reconstruction quality, beyond averages ────────────────────────
    f1s, numrec = [], []
    shown = 0
    for kv, cs, ids, text in eval_pairs:
        rec = _render_reconstruct(pm, _gpu(kv), cs, ids[0], args.max_span, stop_ids)
        f1s.append(_f1_tok(rec, ids))
        nr = _num_recall(tok.decode(rec), text)
        if nr is not None:
            numrec.append(nr)
        if shown < 8:
            shown += 1
            print(f"\n[rec] STEP : {text!r}\n      RENDER: {tok.decode(rec)!r}", flush=True)

    f1s.sort()
    q = lambda p: round(f1s[min(len(f1s) - 1, int(p * len(f1s)))], 3) if f1s else 0.0  # noqa: E731
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 3)  # noqa: E731
    print(
        f"\n[RENDER] eval={len(f1s)} (doc-disjoint; PRIMED with true first token)  "
        f"reconstruct F1: mean={mean(f1s)} p10={q(0.1)} p50={q(0.5)} p90={q(0.9)}\n"
        f"  number-recall (steps with numbers, n={len(numrec)}): mean={mean(numrec)}",
        flush=True,
    )
    print(
        "READ: F1 p50 is typical fidelity; p10 the worst steps; number-recall < 1 "
        "quantifies exactly what the literals ledger must fix (exact digits)."
    )

    if args.out_repo:
        from huggingface_hub import upload_folder  # noqa: PLC0415

        d = "/tmp/render_out"  # noqa: S108
        pm.save_pretrained(d, selected_adapters=["render"])
        upload_folder(repo_id=args.out_repo, folder_path=d, path_in_repo="render_adapter")
        print(f"pushed render adapter to {args.out_repo}/render_adapter", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
