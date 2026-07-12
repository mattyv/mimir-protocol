"""Stage-3b-i: draft-and-verify with REAL thoughts (no predictor, no bridge).

Isolates the loop before adding predicted thoughts (3b-ii). Per reasoning step:
  encode step n -> real thought (gist_kv) -> sample K candidate next-steps from
  the injected thought -> VERIFY each by its NLL under the REAL reasoning-so-far
  (question + steps 1..n) -> keep the best.

Measures, vs a greedy (K=1, no verify) baseline:
- did verify+sampling pick a step closer to the TRUE next step? (F1)
- ADVANCE RATE (Fable gate): is the pick closer to step n+1 than to step n?
  chaining needs forward motion, and 3a-i showed greedy restates ~ as much as
  it advances. Draft-and-verify must lift this.

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_draft_verify \
        --repo mattyvee/mimir-artifacts --n-docs 60 --k-drafts 8
Smoke (local tiny model):
    PYTHONPATH=src python -m marker.run_draft_verify --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.draft_verify import advance_rate, guard_trivial, pick_by_score
from marker.gist_model import decode_from_gist_kv, gist_kv


def _f1(pred, gold):  # noqa: ANN001
    from collections import Counter  # noqa: PLC0415

    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    o = sum((pc & gc).values())
    if o == 0:
        return 0.0
    p, r = o / len(pred), o / len(gold)
    return 2 * p * r / (p + r)


@torch.no_grad()
def _verify_nll(pm, prior_ids, cand_ids):  # noqa: ANN001
    """Mean NLL of cand_ids given the REAL prior context (one full forward) —
    the verify signal: how likely the big model thinks this step is, in situ."""
    import torch.nn.functional as F  # noqa: N812, PLC0415

    if not cand_ids:
        return float("inf")
    device = next(pm.parameters()).device
    ids = torch.tensor([prior_ids + cand_ids], device=device)
    logits = pm(ids).logits[0]
    s = len(prior_ids)
    pred = logits[s - 1 : s - 1 + len(cand_ids)]
    return float(F.cross_entropy(pred, torch.tensor(cand_ids, device=device)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--unit", choices=["line", "sentence"], default="line")
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--k-drafts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.n_docs, args.k_drafts = "Qwen/Qwen2.5-0.5B", None, 3, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.run_stage2 import (  # noqa: PLC0415
        _doc_texts,
        _load_stage1,
        _smoke_cot_texts,
        _split_units,
    )

    pm, gist, tok = _load_stage1(args.model_name, args.repo, device, device == "cuda")
    stop_ids = {t for t in (tok("\n", add_special_tokens=False).input_ids or []) if t}

    # GPU pre-flight (GRAD_OK/eval-smoke philosophy): exercise the FULL draft
    # path on-device — sample with a generator + verify — before the run, so
    # device bugs (a CPU generator vs GPU probs crashed the first attempt) fail
    # in seconds, not after setup+encode.
    _kv, _cs, _fl = gist_kv(pm, gist, tok("preflight check", add_special_tokens=False).input_ids)
    _g = torch.Generator(device="cpu").manual_seed(0)
    _d = decode_from_gist_kv(pm, _kv, _cs, _fl, max_new=4, temperature=0.9, generator=_g)
    _verify_nll(pm, tok("a b c", add_special_tokens=False).input_ids, _d)
    print("PREFLIGHT_OK (sample + verify on device)", flush=True)
    docs = (
        _smoke_cot_texts(args.n_docs)
        if args.smoke
        else list(_doc_texts(args.n_docs, "cot", args.dataset, None, None))
    )

    f_pick, f_greedy = [], []
    picks, greedys, curs, nexts = [], [], [], []
    shown = 0
    for text in docs:
        steps = _split_units(text, args.unit)
        step_ids = [tok(s, add_special_tokens=False).input_ids[: args.max_span] for s in steps]
        step_ids = [s for s in step_ids if s]
        for n in range(len(step_ids) - 1):
            span_a, cont_b = step_ids[n], step_ids[n + 1]
            kv, cont_start, first_logits = gist_kv(pm, gist, span_a)
            greedy = decode_from_gist_kv(
                pm, kv, cont_start, first_logits, max_new=args.max_new, stop_ids=stop_ids
            )
            drafts = [greedy]
            for j in range(args.k_drafts):
                g = torch.Generator(device="cpu").manual_seed(1000 * n + j)
                drafts.append(
                    decode_from_gist_kv(
                        pm,
                        kv,
                        cont_start,
                        first_logits,
                        max_new=args.max_new,
                        stop_ids=stop_ids,
                        temperature=args.temperature,
                        generator=g,
                    )
                )
            # steps 1..n as the verify context, newline-joined — without the
            # separators the prior reads as run-on text, off-distribution for
            # the verifier (Fable 3b review)
            nl = tok("\n", add_special_tokens=False).input_ids
            prior = []
            for s in step_ids[: n + 1]:
                prior.extend(s)
                prior.extend(nl)
            scores = guard_trivial(drafts, [_verify_nll(pm, prior, d) for d in drafts])
            picked, _ = pick_by_score(drafts, scores)
            f_pick.append(_f1(picked, cont_b))
            f_greedy.append(_f1(greedy, cont_b))
            picks.append(picked)
            greedys.append(greedy)
            curs.append(span_a)
            nexts.append(cont_b)
            if shown < args.show:
                shown += 1
                print(
                    f"\n[n={n}] TRUE-NEXT: {tok.decode(cont_b)!r}\n"
                    f"      GREEDY  : {tok.decode(greedy)!r}\n"
                    f"      PICKED  : {tok.decode(picked)!r}",
                    flush=True,
                )

    sim = lambda a, b: _f1(a, b)  # noqa: E731
    m = lambda xs: round(sum(xs) / max(1, len(xs)), 3)  # noqa: E731
    print(
        f"\n[DRAFT-VERIFY] pairs={len(f_pick)}  k={args.k_drafts}  temp={args.temperature}\n"
        f"  F1(next):     picked={m(f_pick)}  greedy={m(f_greedy)}\n"
        f"  ADVANCE rate: picked={round(advance_rate(picks, curs, nexts, sim), 3)}  "
        f"greedy={round(advance_rate(greedys, curs, nexts, sim), 3)}",
        flush=True,
    )
    print(
        "GATE: picked F1 > greedy F1 (verify helps) AND picked advance-rate > 0.5 "
        "and > greedy (drafts move to the NEXT step, don't restate)."
    )


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
