"""Crowding experiment: how many facts fit in an N-token trained prefix?

See CROWDING_PLAN.md for the full design and pre-registered readings.

Grid: F facts-per-axiom in --f-list x N tokens-per-prefix in --n-list. One
synthetic axiom per F (crowding.make_axiom, deterministic), one trained
prefix per (F, N) cell. ZERO/FACTS baselines are computed once per F (they
don't depend on N) and reused across the N columns.

Per cell, two scored buckets per fact: TRAIN (one verbatim training question
per fact — the undertrained-vs-crowded control) and UNSEEN (the dev + test
held-out templates, 2 probes/fact, the headline metric). Confusion is scored
on wrong UNSEEN answers: does the generated text contain a SIBLING fact's
value (cross-fact confusion) rather than nothing recognizable (plain miss)?

Run-2 amendments (after the first run's F>=4 cells failed their own TRAIN
control — see the run-1 postmortem in CROWDING_PLAN.md):
  - batched training (--batch-size 8): run 1 was batch-1 with last-step loss
    ~1e-3 yet TRAIN recall ~10% — single-sample steps fit the fact just
    visited while perturbing the rest; the joint problem was never fit.
  - digit-boundary scoring (_matches): plain substring let gold "10" match
    degenerate "100000.0.0.0" output, inflating garbage cells.
  - joint teacher-forced mean loss reported per cell — the diagnostic that
    separates "not fit" from "fit but drifts at generation".
  - steps hold samples/fact at ~800-1600 across F (run 1 collapsed to ~156
    at F=32).

Run (GPU):
    PYTHONPATH=src python -m marker.run_crowding --model-name Qwen/Qwen2.5-7B
Smoke (must pass locally before any Vast launch):
    PYTHONPATH=src python -m marker.run_crowding --smoke
"""

from __future__ import annotations

import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.crowding import make_axiom
from marker.prefix_poc import (
    build_prefix_cache,
    generate_with_cache,
    init_stat_matched,
    mean_teacher_forced_loss,
    train_prefix,
)
from marker.run_axiom_mlp_demo import TEMPLATE, _build_dynamic_cache, compute_axiom_kv

TRAIN_TEMPLATES = ["Q: {q}\nA:", "{q}\n", "Question: {q}\nAnswer:"]

# Optimizer steps per F at batch_size 8 — samples/fact = steps * batch / F,
# held at ~800-1600 across F (the regime the tuned run validated). The first
# crowding run's schedule collapsed to ~156 batch-1 samples/fact at F=32.
STEPS_BY_F = {2: 400, 4: 600, 8: 1000, 16: 2000, 32: 3200}


def _matches(answer: str, gold: str) -> bool:
    """Digit-boundary substring match.

    Plain substring scoring false-positived in the first crowding run: gold
    "10" matched inside degenerate "100000.0.0.0" output. Requiring no digit
    adjacent to the match kills that class ("10" no longer matches "100000"
    or "210") while leaving text golds and unit-suffixed golds ("38%",
    "989ms", "orders.vx7") unaffected.
    """
    return re.search(rf"(?<!\d){re.escape(gold.lower())}(?!\d)", answer.lower()) is not None


def _score_unseen(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom: dict,
    make_cache,  # noqa: ANN001  — zero-arg callable returning a FRESH cache (or None)
    max_new: int,
    print_detail: bool,
    label: str,
) -> tuple[int, int, list[tuple[int, str, str, bool]]]:
    """Score BOTH unseen buckets (dev + test — 2 probes per fact; for the
    synthetic axioms they're symmetric held-out templates, and combining
    doubles statistical power). Records carry the owning fact index for the
    confusion metric.

    make_cache is called PER PROBE. Runs 1 and 2 passed a single DynamicCache
    in here; the model mutates it in place, so every probe after the first
    ran against prefix + all previous Q&As — an uncontrolled multi-turn
    conversation, not the injected prefix. (Fingerprints: fact 1 always
    correct, "Yes, ..." dialogue-continuation answers, degeneracy compounding
    down the probe list, all while joint teacher-forced loss was ~2e-4.)
    Both runs' F>=4 surfaces measured this bug, not crowding.
    """
    correct = 0
    total = 0
    records: list[tuple[int, str, str, bool]] = []
    for fact_idx, fact in enumerate(axiom["facts"]):
        for q, gold in [*fact["dev"], *fact["test"]]:
            prompt = TEMPLATE.format(q=q)
            out = generate_with_cache(model, tokenizer, prompt, make_cache(), max_new)
            ok = _matches(out, gold)
            correct += int(ok)
            total += 1
            records.append((fact_idx, q, out, ok))
            if print_detail:
                print(f"    [{label:11}] {'v' if ok else 'x'} {out[:80].replace(chr(10), ' ')}")
    return correct, total, records


def _score_train_control(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom: dict,
    make_cache,  # noqa: ANN001  — zero-arg callable returning a FRESH cache
    max_new: int,
) -> tuple[int, int]:
    """One verbatim training question per fact — checks undertraining, not
    the headline metric, so results aren't printed per-question (log-size
    control at F=32). make_cache is called per probe (see _score_unseen).
    """
    correct = 0
    total = 0
    for fact in axiom["facts"]:
        q, _answer = fact["train"][0]
        gold = fact["value"]
        prompt = TEMPLATE.format(q=q)
        out = generate_with_cache(model, tokenizer, prompt, make_cache(), max_new)
        correct += int(_matches(out, gold))
        total += 1
    return correct, total


def _count_confusions(axiom: dict, records: list[tuple[int, str, str, bool]]) -> int:
    """Among wrong answers, count those containing a DIFFERENT fact's value.
    Each record carries its owning fact index so sibling values are checked
    with the same digit-boundary matcher used for scoring.
    """
    facts = axiom["facts"]
    confused = 0
    for fact_idx, _q, out, ok in records:
        if ok:
            continue
        sibling_values = [f["value"] for j, f in enumerate(facts) if j != fact_idx]
        if any(_matches(out, v) for v in sibling_values):
            confused += 1
    return confused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    # Rerun default: F=2/4 already had adequate samples/fact in run 1; the
    # interpretable-if-fixed cells are 8/16/32.
    parser.add_argument("--f-list", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--n-list", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    f_list, n_list = args.f_list, args.n_list
    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        f_list, n_list = [2], [4]
        args.batch_size = 4  # exercise the padded-batch path, not batch-1
        args.max_new = min(args.max_new, 20)
        print("=== SMOKE MODE ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}")
    print(f"F list: {f_list}  N list: {n_list}  batch: {args.batch_size}\n")

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
        n_steps = STEPS_BY_F.get(f, 1000) if not args.smoke else 10
        samples_per_fact = n_steps * args.batch_size // f
        axiom = make_axiom(f"CrowdAxiom{f}", f, seed=args.seed)
        print(
            f"\n{'=' * 70}\n### F={f}  ({n_steps} steps/cell x batch {args.batch_size} "
            f"= ~{samples_per_fact} samples/fact)"
        )
        print(f"  fact_text: {axiom['fact_text'][:120]}")

        real_kv = compute_axiom_kv(model, tokenizer, axiom["fact_text"], term=axiom["name"])
        facts_positions = real_kv.keys[0].shape[2]
        print(f"  FACTS cache positions: {facts_positions}")

        def _fresh_facts_cache(kv=real_kv):  # noqa: ANN001, ANN202
            return _build_dynamic_cache(kv, model_device)

        zero_c, zero_t, _ = _score_unseen(
            model, tokenizer, axiom, lambda: None, args.max_new, False, "ZERO"
        )
        facts_c, facts_t, _ = _score_unseen(
            model, tokenizer, axiom, _fresh_facts_cache, args.max_new, False, "FACTS"
        )
        print(f"  ZERO  UNSEEN: {zero_c}/{zero_t}")
        print(f"  FACTS UNSEEN: {facts_c}/{facts_t}")
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
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t0
            tail = losses[-100:]
            tail_mean = sum(tail) / len(tail)
            joint_loss = mean_teacher_forced_loss(model, tokenizer, prefix, qa_groups)
            print(
                f"  N={n}: loss(last-100 mean) {tail_mean:.4f}  "
                f"joint teacher-forced mean {joint_loss:.4f}  ({elapsed:.0f}s)"
            )

            with torch.no_grad():

                def _fresh_prefix_cache(p=prefix):  # noqa: ANN001, ANN202
                    return build_prefix_cache(p, dtype)

                train_c, train_t = _score_train_control(
                    model, tokenizer, axiom, _fresh_prefix_cache, args.max_new
                )
                test_c, test_t, test_records = _score_unseen(
                    model,
                    tokenizer,
                    axiom,
                    _fresh_prefix_cache,
                    args.max_new,
                    True,
                    f"N={n} UNSEEN",
                )
                confused = _count_confusions(axiom, test_records)
                wrong_total = sum(1 for *_r, ok in test_records if not ok)

            print(
                f"    TRAIN(control): {train_c}/{train_t}   UNSEEN: {test_c}/{test_t}   "
                f"confused(of {wrong_total} wrong): {confused}"
            )
            results[f][n] = {
                "test": (test_c, test_t),
                "train": (train_c, train_t),
                "confusion": (confused, wrong_total),
                "joint_loss": joint_loss,
            }

    # ── Summary surfaces ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("UNSEEN ACCURACY SURFACE (rows F, cols N) + FACTS baseline")
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
    print("CONFUSION SURFACE — wrong UNSEEN answers containing a sibling fact's value")
    print("=" * 78)
    print("  F  " + "".join(f"{'N=' + str(n):>10}" for n in n_list))
    for f in f_list:
        row = f"  {f:<3}"
        for n in n_list:
            conf, wrong = results[f][n]["confusion"]
            row += f"{f'{conf}/{wrong}':>10}"
        print(row)

    print("\n" + "=" * 78)
    print("JOINT TEACHER-FORCED LOSS SURFACE — high = joint fit not reached")
    print("=" * 78)
    print("  F  " + "".join(f"{'N=' + str(n):>10}" for n in n_list))
    for f in f_list:
        row = f"  {f:<3}"
        for n in n_list:
            row += f"{results[f][n]['joint_loss']:>10.4f}"
        print(row)

    print("\n" + "=" * 78)
    print("SIZING RULE — smallest N with UNSEEN >= 90% of FACTS")
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
