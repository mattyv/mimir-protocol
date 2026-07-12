"""Stage-3b: CHAIN-conditioned draft-and-verify with real thoughts.

3b-i (single-thought drafts) failed both gates: verify added nothing (picked
F1 0.391 ~= greedy 0.390) and drafts didn't advance (~0.29). Diagnosis: one
thought is directionless. This conditions drafts on the ACCUMULATED thoughts
of steps 1..n (chain_gist_kv), against Fable's second-pass requirements:
  - CHAIN-conditioned drafting (accumulated thought-memory, canonical positions)
  - ORACLE logged (best-of-K vs truth): localizes the failure — verify
    selection vs draft distribution
  - QUESTION in the verify context (3b-i judged steps blind to the problem)
  - advance metric CALIBRATED against a true-next ceiling + prev-step floor,
    not an arbitrary 0.5

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
from marker.gist_model import chain_gist_kv, decode_from_gist_kv, gist_kv


def _qa_docs(n, dataset):  # noqa: ANN001
    """Stream n (question, answer) pairs — the question is the verify context
    that 3b-i omitted (judging math steps blind to the problem)."""
    from datasets import load_dataset  # noqa: PLC0415

    is_gsm = dataset == "openai/gsm8k"
    ds = load_dataset(
        dataset, "main" if is_gsm else None, split="test" if is_gsm else "train", streaming=True
    )
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append((row.get("question") or "", row.get("answer") or row.get("solution") or ""))
    return out


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
    if args.smoke:
        qas = [("Q?", t) for t in _smoke_cot_texts(args.n_docs)]
    else:
        qas = _qa_docs(args.n_docs, args.dataset)
    nl = tok("\n", add_special_tokens=False).input_ids

    f_pick, f_greedy, f_oracle = [], [], []
    picks, greedys, curs, nexts, prevs = [], [], [], [], []
    shown = 0
    for question, answer in qas:
        steps = _split_units(answer, args.unit)
        step_ids = [tok(s, add_special_tokens=False).input_ids[: args.max_span] for s in steps]
        step_ids = [s for s in step_ids if s]
        q_ids = tok(question, add_special_tokens=False).input_ids if question else []
        for n in range(len(step_ids) - 1):
            cont_b = step_ids[n + 1]
            # CHAIN-CONDITIONED: draft from the accumulated thoughts of steps 1..n
            kv, cont_start, first_logits = chain_gist_kv(pm, gist, step_ids[: n + 1])
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
            # verify context = QUESTION + steps 1..n, newline-joined (3b-i judged
            # steps blind to the problem; a real runtime has the question)
            prior = list(q_ids) + list(nl)
            for s in step_ids[: n + 1]:
                prior.extend(s)
                prior.extend(nl)
            scores = guard_trivial(drafts, [_verify_nll(pm, prior, d) for d in drafts])
            picked, _ = pick_by_score(drafts, scores)
            f_pick.append(_f1(picked, cont_b))
            f_greedy.append(_f1(greedy, cont_b))
            f_oracle.append(
                max(_f1(d, cont_b) for d in drafts)
            )  # best-of-K: is verify or drafts the bottleneck?
            picks.append(picked)
            greedys.append(greedy)
            curs.append(step_ids[n])
            nexts.append(cont_b)
            prevs.append(step_ids[n - 1] if n > 0 else step_ids[n])
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
    ar = lambda ds: round(advance_rate(ds, curs, nexts, sim), 3)  # noqa: E731
    print(
        f"\n[DRAFT-VERIFY chain] pairs={len(f_pick)}  k={args.k_drafts}  temp={args.temperature}\n"
        f"  F1(next):     picked={m(f_pick)}  greedy={m(f_greedy)}  ORACLE(best-of-K)={m(f_oracle)}\n"
        f"  ADVANCE rate: picked={ar(picks)}  greedy={ar(greedys)}  "
        f"|| CEILING(true-next)={ar(nexts)}  FLOOR(prev-step)={ar(prevs)}",
        flush=True,
    )
    print(
        "READ: ORACLE >> greedy => verify is the bottleneck (good drafts exist, "
        "selection misses them); ORACLE ~= greedy => drafts are. Judge picked "
        "advance vs the CEILING/FLOOR bracket, not a fixed 0.5."
    )


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
