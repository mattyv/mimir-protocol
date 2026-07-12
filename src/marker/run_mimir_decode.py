"""Stage-3 3a-i ceiling test: does a THOUGHT round-trip back to text?

The Stage-2 predictor predicts a next-thought vector. Before building any
draft-and-verify loop we must know the decode path works AT ALL — so this
tests the CEILING with a REAL thought (not a predicted one), independent of
predictor quality:

  encode step n -> gist_kv (full per-layer thought) -> decode_from_gist_kv
  -> text, compared to the step's true CONTINUATION (step n+1 — what the gist
  was trained to make predictable).

Gates (vs baselines, on held-out GSM8K reasoning chains):
- reconstruct F1(decoded, next_step) must beat F1(no-injection, next_step)
  (the thought carries content) AND beat F1(decoded, random_step) (it's the
  RIGHT content, not generic). If it doesn't clear both, the decode path is
  the bottleneck — fix that before predicted thoughts.
- Qualitative: prints decoded vs true-next side by side (round-trip is
  judged by eye too, not just F1).

Run (GPU):
    HF_TOKEN=... PYTHONPATH=src python -u -m marker.run_mimir_decode \
        --repo mattyvee/mimir-artifacts --n-docs 60
Smoke (local tiny model):
    PYTHONPATH=src python -m marker.run_mimir_decode --smoke
"""

from __future__ import annotations

import argparse

import torch

from marker.gist_model import decode_from_gist_kv, gist_kv


def _f1(pred_ids: list[int], gold_ids: list[int]) -> float:
    """Token-overlap F1 (multiset) — cheap content-overlap proxy, order-free."""
    if not pred_ids or not gold_ids:
        return 0.0
    from collections import Counter  # noqa: PLC0415

    pc, gc = Counter(pred_ids), Counter(gold_ids)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pred_ids), overlap / len(gold_ids)
    return 2 * prec * rec / (prec + rec)


def _step_pairs_from_docs(tok, docs, max_span, max_steps, unit):  # noqa: ANN001
    """(step_n span, step_{n+1} ids) consecutive pairs across docs."""
    from marker.run_stage2 import _split_units  # noqa: PLC0415

    pairs = []
    for text in docs:
        steps = _split_units(text, unit)[:max_steps]
        ids = [tok(s, add_special_tokens=False).input_ids[:max_span] for s in steps]
        ids = [x for x in ids if x]
        for a, b in zip(ids[:-1], ids[1:], strict=False):
            pairs.append((a, b))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--unit", choices=["line", "sentence"], default="line")
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--max-span", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument(
        "--prime", type=int, default=None, help="seed token id (default: a leading space)"
    )
    ap.add_argument("--show", type=int, default=8, help="qualitative examples to print")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.model_name, args.repo, args.n_docs = "Qwen/Qwen2.5-0.5B", None, 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from marker.run_stage2 import _doc_texts, _load_stage1, _smoke_cot_texts  # noqa: PLC0415

    pm, gist, tok = _load_stage1(args.model_name, args.repo, device, device == "cuda")
    prime = (
        [args.prime] if args.prime is not None else tok(" ", add_special_tokens=False).input_ids[:1]
    )
    if not prime:
        prime = [tok.eos_token_id or 0]

    docs = (
        _smoke_cot_texts(args.n_docs)
        if args.smoke
        else list(_doc_texts(args.n_docs, "cot", args.dataset, None, None))
    )
    pairs = _step_pairs_from_docs(tok, docs, args.max_span, 16, args.unit)
    print(f"{len(pairs)} consecutive step pairs", flush=True)

    import random  # noqa: PLC0415

    rng = random.Random(0)
    f_next, f_span, f_rand, f_none = [], [], [], []
    shown = 0
    for i, (span_a, cont_b) in enumerate(pairs):
        kv = gist_kv(pm, gist, span_a)
        dec = decode_from_gist_kv(pm, kv, prime, max_new=args.max_new, eos_id=tok.eos_token_id)
        rand_b = pairs[rng.randrange(len(pairs))][1]
        # no-injection baseline: decode from the prime with an EMPTY cache
        none = _decode_no_kv(pm, prime, args.max_new, tok.eos_token_id)
        f_next.append(_f1(dec, cont_b))
        f_span.append(_f1(dec, span_a))
        f_rand.append(_f1(dec, rand_b))
        f_none.append(_f1(none, cont_b))
        if shown < args.show:
            shown += 1
            print(
                f"\n[ex {i}] TRUE-NEXT: {tok.decode(cont_b)!r}\n"
                f"        DECODED : {tok.decode(dec)!r}",
                flush=True,
            )
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(pairs)}", flush=True)

    m = lambda xs: round(sum(xs) / max(1, len(xs)), 3)  # noqa: E731
    print(
        f"\n[MIMIR DECODE] F1(decoded, next)={m(f_next)}  "
        f"vs no-inject={m(f_none)}  vs random-step={m(f_rand)}  vs own-span={m(f_span)}",
        flush=True,
    )
    print(
        "GATE: F1(decoded,next) > F1(no-inject,next) [thought carries content] "
        "AND > F1(decoded,random) [it's the RIGHT content]. Else decode path is the bottleneck."
    )


@torch.no_grad()
def _decode_no_kv(peft_model, prime, max_new, eos_id):  # noqa: ANN001
    """No-injection baseline: greedy decode from the prime alone (empty cache)."""
    device = next(peft_model.parameters()).device
    ids = torch.tensor([prime], device=device)
    out = peft_model(ids, use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax().item())
    gen = [nxt]
    for _ in range(max_new - 1):
        if nxt == eos_id:
            break
        out = peft_model(torch.tensor([[nxt]], device=device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        gen.append(nxt)
    return gen


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
