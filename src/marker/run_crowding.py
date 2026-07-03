"""Crowding experiment: how many facts fit in an N-token trained prefix?

See CROWDING_PLAN.md for the full design and pre-registered readings.

Grid: F facts-per-axiom in --f-list x N tokens-per-prefix in --n-list. One
synthetic axiom per F (crowding.make_axiom, deterministic), one trained
prefix per (F, N) cell. ZERO/FACTS baselines are computed once per F (they
don't depend on N) and reused across the N columns.

Per cell, two scored buckets per fact: TRAIN (one verbatim training question
per fact — the undertrained-vs-crowded control) and TEST (unseen phrasings,
the headline metric). Confusion is scored on wrong TEST answers: does the
generated text contain a SIBLING fact's value (cross-fact confusion) rather
than nothing recognizable (plain miss)?

Run (GPU):
    PYTHONPATH=src python -m marker.run_crowding --model-name Qwen/Qwen2.5-7B
Smoke (must pass locally before any Vast launch):
    PYTHONPATH=src python -m marker.run_crowding --smoke
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.crowding import make_axiom
from marker.prefix_poc import (
    build_prefix_cache,
    generate_with_cache,
    init_stat_matched,
    train_prefix,
)
from marker.run_axiom_mlp_demo import TEMPLATE, _build_dynamic_cache, compute_axiom_kv

TRAIN_TEMPLATES = ["Q: {q}\nA:", "{q}\n", "Question: {q}\nAnswer:"]

# Steps scale with F (the workload the prefix must fit), not N.
STEPS_BY_F = {2: 2000, 4: 2000, 8: 2500, 16: 3500, 32: 5000}


def _contains(answer: str, gold: str) -> bool:
    return gold.lower() in answer.lower()


def _score_test_bucket(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom: dict,
    kv_cache,  # noqa: ANN001
    max_new: int,
    print_detail: bool,
    label: str,
) -> tuple[int, int, list[tuple[str, str, bool]]]:
    """Score the TEST bucket (one probe per fact, in fact order) for one
    condition. Returns (correct, total, records) where records[i] corresponds
    to axiom["facts"][i] — the confusion metric relies on this ordering.
    """
    correct = 0
    total = 0
    records: list[tuple[str, str, bool]] = []
    for fact in axiom["facts"]:
        (q, gold) = fact["test"][0]
        prompt = TEMPLATE.format(q=q)
        out = generate_with_cache(model, tokenizer, prompt, kv_cache, max_new)
        ok = _contains(out, gold)
        correct += int(ok)
        total += 1
        records.append((q, out, ok))
        if print_detail:
            print(f"    [{label:11}] {'v' if ok else 'x'} {out[:80].replace(chr(10), ' ')}")
    return correct, total, records


def _score_train_control(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom: dict,
    kv_cache,  # noqa: ANN001
    max_new: int,
) -> tuple[int, int]:
    """One verbatim training question per fact — checks undertraining, not
    the headline metric, so results aren't printed per-question (log-size
    control at F=32).
    """
    correct = 0
    total = 0
    for fact in axiom["facts"]:
        q, _answer = fact["train"][0]
        gold = fact["value"]
        prompt = TEMPLATE.format(q=q)
        out = generate_with_cache(model, tokenizer, prompt, kv_cache, max_new)
        correct += int(_contains(out, gold))
        total += 1
    return correct, total


def _count_confusions(axiom: dict, records: list[tuple[str, str, bool]]) -> int:
    """records[i] must correspond to axiom["facts"][i] (one probe per fact).
    Among wrong answers, count those containing a DIFFERENT fact's value.
    """
    facts = axiom["facts"]
    assert len(records) == len(facts), "confusion metric assumes one probe per fact"
    confused = 0
    for i, (_q, out, ok) in enumerate(records):
        if ok:
            continue
        sibling_values = [f["value"].lower() for j, f in enumerate(facts) if j != i]
        if any(v in out.lower() for v in sibling_values):
            confused += 1
    return confused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--f-list", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--n-list", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-new", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    f_list, n_list = args.f_list, args.n_list
    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        f_list, n_list = [2], [4]
        args.max_new = min(args.max_new, 20)
        print("=== SMOKE MODE ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}")
    print(f"F list: {f_list}  N list: {n_list}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # results[F][N] = {"test": (c,t), "train": (c,t), "confusion": (n, wrong_total)}
    # baseline[F] = {"facts_test": (c,t), "zero_test": (c,t), "facts_positions": int}
    results: dict[int, dict[int, dict]] = {}
    baseline: dict[int, dict] = {}

    for f in f_list:
        n_steps = STEPS_BY_F.get(f, 2000) if not args.smoke else 10
        axiom = make_axiom(f"CrowdAxiom{f}", f, seed=args.seed)
        print(f"\n{'=' * 70}\n### F={f}  ({n_steps} steps/cell)")
        print(f"  fact_text: {axiom['fact_text'][:120]}")

        real_kv = compute_axiom_kv(model, tokenizer, axiom["fact_text"], term=axiom["name"])
        facts_positions = real_kv.keys[0].shape[2]
        print(f"  FACTS cache positions: {facts_positions}")

        zero_c, zero_t, _ = _score_test_bucket(
            model, tokenizer, axiom, None, args.max_new, False, "ZERO"
        )
        facts_cache = _build_dynamic_cache(real_kv, model_device)
        facts_c, facts_t, _ = _score_test_bucket(
            model, tokenizer, axiom, facts_cache, args.max_new, False, "FACTS"
        )
        print(f"  ZERO  TEST: {zero_c}/{zero_t}")
        print(f"  FACTS TEST: {facts_c}/{facts_t}")
        baseline[f] = {
            "facts_test": (facts_c, facts_t),
            "zero_test": (zero_c, zero_t),
            "facts_positions": facts_positions,
        }
        results[f] = {}

        for n in n_list:
            prefix = init_stat_matched(real_kv, n_tokens=n, term=axiom["name"])
            t0 = time.time()
            qa_groups = [fact["train"] for fact in axiom["facts"]]
            losses = train_prefix(
                model,
                tokenizer,
                prefix,
                n_steps=n_steps,
                lr=args.lr,
                lr_end=args.lr_end,
                weight_decay=args.weight_decay,
                qa_groups=qa_groups,
                templates=TRAIN_TEMPLATES,
            )
            elapsed = time.time() - t0
            print(f"  N={n}: loss {losses[0]:.3f} -> {losses[-1]:.4f}  ({elapsed:.0f}s)")

            with torch.no_grad():
                train_cache = build_prefix_cache(prefix, dtype)
                train_c, train_t = _score_train_control(
                    model, tokenizer, axiom, train_cache, args.max_new
                )
                test_cache = build_prefix_cache(prefix, dtype)
                test_c, test_t, test_records = _score_test_bucket(
                    model, tokenizer, axiom, test_cache, args.max_new, True, f"N={n} TEST"
                )
                confused = _count_confusions(axiom, test_records)
                wrong_total = sum(1 for *_r, ok in test_records if not ok)

            print(
                f"    TRAIN(control): {train_c}/{train_t}   TEST: {test_c}/{test_t}   "
                f"confused(of {wrong_total} wrong): {confused}"
            )
            results[f][n] = {
                "test": (test_c, test_t),
                "train": (train_c, train_t),
                "confusion": (confused, wrong_total),
            }

    # ── Summary surfaces ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TEST ACCURACY SURFACE (rows F, cols N) + FACTS baseline")
    print("=" * 78)
    header = (
        "  F  " + "".join(f"{'N=' + str(n):>10}" for n in n_list) + f"{'FACTS':>12}{'ZERO':>10}"
    )
    print(header)
    for f in f_list:
        row = f"  {f:<3}"
        for n in n_list:
            c, t = results[f][n]["test"]
            row += f"{f'{c}/{t}':>10}"
        fc, ft = baseline[f]["facts_test"]
        zc, zt = baseline[f]["zero_test"]
        row += f"{f'{fc}/{ft}':>12}{f'{zc}/{zt}':>10}"
        print(row)

    print("\n" + "=" * 78)
    print("TRAIN ACCURACY SURFACE — undertrained-vs-crowded control")
    print("=" * 78)
    print("  F  " + "".join(f"{'N=' + str(n):>10}" for n in n_list))
    for f in f_list:
        row = f"  {f:<3}"
        for n in n_list:
            c, t = results[f][n]["train"]
            row += f"{f'{c}/{t}':>10}"
        print(row)

    print("\n" + "=" * 78)
    print("CONFUSION SURFACE — wrong TEST answers containing a sibling fact's value")
    print("=" * 78)
    print("  F  " + "".join(f"{'N=' + str(n):>10}" for n in n_list))
    for f in f_list:
        row = f"  {f:<3}"
        for n in n_list:
            conf, wrong = results[f][n]["confusion"]
            row += f"{f'{conf}/{wrong}':>10}"
        print(row)

    print("\n" + "=" * 78)
    print("SIZING RULE — smallest N with TEST >= 90% of FACTS")
    print("=" * 78)
    for f in f_list:
        fc, ft = baseline[f]["facts_test"]
        facts_frac = fc / ft if ft else 0.0
        target = 0.9 * facts_frac
        n_star = None
        for n in sorted(n_list):
            c, t = results[f][n]["test"]
            frac = c / t if t else 0.0
            if frac >= target:
                n_star = n
                break
        if n_star is None:
            print(f"  F={f:<3} N* = not reached within tested N (target {target:.2f})")
        else:
            print(f"  F={f:<3} N* = {n_star:<4} N*/F = {n_star / f:.2f}")


if __name__ == "__main__":
    main()
