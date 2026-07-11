"""Encoder-on-reasoning check: does the FineWeb-trained Stage-1 gist encode
REASONING steps, or is it out-of-distribution on them? (Pre-registered gate in
STAGE2_PLAN before any CoT Stage-2 spend.)

Reasoning traces (equations, 'therefore', symbolic tokens) are a different
distribution from the web prose the gist adapter was trained on. Before the
CoT next-thought run, measure gap_closed on consecutive reasoning-step pairs
(span = step n, continuation = step n+1) with the EXISTING adapter:
- gap_closed >= ~0.4  -> encoder handles reasoning; proceed to the CoT run.
- collapse            -> Stage-1 re-fit on CoT data first (the bigger fork).
Prose reference: 0.887 (Stage-1 pilot).

Eval-only, ~10 min on a 3090 (a few hundred forwards):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.reason_check \
        --repo mattyvee/mimir-artifacts --n-problems 100
"""

from __future__ import annotations

import argparse
import re

import torch

_CALC = re.compile(r"<<[^>]*>>")


def split_solution_steps(solution: str) -> list[str]:
    """GSM8K-style solution -> reasoning steps: one per line, calculator
    annotations (<<48/2=24>>) stripped, the final '#### answer' line and
    blank lines dropped."""
    steps = []
    for line in solution.split("\n"):
        line = _CALC.sub("", line).strip()
        if not line or line.startswith("####"):
            continue
        steps.append(line)
    return steps


def step_pairs(steps: list[str]) -> list[tuple[str, str]]:
    """(step n, step n+1) pairs — the reasoning succession the gist must carry."""
    return list(zip(steps[:-1], steps[1:], strict=False))


def main() -> None:
    from marker.gist_model import gap_closed, gist_forward  # noqa: PLC0415
    from marker.run_stage2 import _load_stage1  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pm, gist, tok = _load_stage1(args.model_name, args.repo, device, device == "cuda")

    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    spans, conts = [], []
    for i, row in enumerate(ds):
        if i >= args.n_problems:
            break
        for a, b in step_pairs(split_solution_steps(row["answer"])):
            sa = tok(a, add_special_tokens=False).input_ids[: args.max_span]
            sb = tok(b, add_special_tokens=False).input_ids[: args.max_span]
            if sa and sb:
                spans.append(sa)
                conts.append(sb)
    print(f"{len(spans)} reasoning-step pairs from {args.n_problems} problems", flush=True)

    # mean CE per condition over all pairs -> one PPL each -> gap_closed
    sums = {"gist": 0.0, "full": 0.0, "none": 0.0}
    n_batches = 0
    sees = {
        "gist": frozenset({"gist"}),
        "full": frozenset({"gist", "span"}),
        "none": frozenset(),
    }
    with torch.no_grad():
        for i in range(0, len(spans) - args.batch + 1, args.batch):
            sp, co = spans[i : i + args.batch], conts[i : i + args.batch]
            for name, s in sees.items():
                sums[name] += float(gist_forward(pm, gist, sp, co, cont_sees=s))
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  ...{n_batches} batches", flush=True)
    ppls = {k: float(torch.exp(torch.tensor(v / n_batches))) for k, v in sums.items()}
    gc = gap_closed(ppls)
    print(f"[REASON CHECK] ppls={ {k: round(v, 3) for k, v in ppls.items()} }", flush=True)
    print(f"[REASON CHECK] gap_closed={gc:.3f}  (prose reference 0.887)", flush=True)
    print("GATE: >= 0.4 -> CoT Stage-2 run; below -> Stage-1 re-fit on CoT first.")


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
