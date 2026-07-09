"""Runner: speculative draft-and-verify on Qwen2.5 (see specdec.py).

Smoke (CPU): drafter == verifier == 0.5B. Same model both roles means the
drafter's greedy proposals ARE the verifier's greedy choices, so acceptance
should be ~100% and the output token-identical to vanilla greedy — a hard
mechanical invariant (identity is asserted; acceptance is reported, since a
float near-tie between the batch and incremental code paths can dent it
without breaking identity).

GPU: Qwen2.5-0.5B drafts for Qwen2.5-7B (shared tokenizer, both base), gamma
sweep, per-prompt token-exact identity vs the 7B's own greedy decode, and the
headline numbers: acceptance rate and tokens-per-verifier-pass (the
weight-streaming speedup proxy). This is the UNCONDITIONED baseline a
thought-conditioned drafter must beat.

Run (GPU):
    PYTHONPATH=src python -m marker.run_spec_decode \
        --verifier Qwen/Qwen2.5-7B --drafter Qwen/Qwen2.5-0.5B
Smoke (local):
    PYTHONPATH=src python -m marker.run_spec_decode --smoke
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.specdec import greedy_decode, greedy_decode_prefill, spec_decode

PROMPTS = [
    "Q: Explain why the sky is blue in two sentences.\nA:",
    "Q: Write a Python function to compute the factorial of n.\nA:",
    "Q: What is the capital of Australia, and why is it not Sydney?\nA:",
    "Q: Summarize the plot of Romeo and Juliet in three sentences.\nA:",
    "Q: Write a SQL query to find the top 5 customers by total order value.\nA:",
    "Q: Describe how a hash map works, briefly.\nA:",
]

GAMMAS = [4, 8, 16]


def _load(name: str, device: str):  # noqa: ANN001, ANN202
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model, tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--drafter", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument(
        "--reference-prefill",
        action="store_true",
        help="Finding 2 cross-check: build the greedy reference via repeated "
        "full-prefill (the verifier's own code path) instead of incremental "
        "decode. If spec output then matches, the identity gap was numerics.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    prompts = PROMPTS
    gammas = GAMMAS
    if args.smoke:
        args.verifier = args.drafter = "Qwen/Qwen2.5-0.5B"
        args.max_new = 24
        prompts = PROMPTS[:2]
        gammas = [4]
        print("=== SMOKE MODE (drafter == verifier: identity is a hard invariant) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  verifier: {args.verifier}  drafter: {args.drafter}\n")

    verifier, tok = _load(args.verifier, device)
    if args.drafter == args.verifier:
        drafter = verifier
    else:
        drafter, _ = _load(args.drafter, device)  # shared Qwen2.5 tokenizer
    eos_id = tok.eos_token_id

    # ── Vanilla reference (once per prompt — gamma-independent) ────────────────
    ref_fn = greedy_decode_prefill if args.reference_prefill else greedy_decode
    print(
        f"reference path: {'full-prefill (Finding 2 cross-check)' if args.reference_prefill else 'incremental'}"
    )
    references: list[list[int]] = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        references.append(ref_fn(verifier, ids, args.max_new, eos_id))

    # ── Spec decode grid ────────────────────────────────────────────────────────
    summary_rows = []
    for gamma in gammas:
        print(f"\n--- gamma={gamma} ---")
        identical = 0
        acc_rates, tpp = [], []
        for p, ref in zip(prompts, references, strict=True):
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            out, stats = spec_decode(verifier, drafter, ids, args.max_new, gamma, eos_id)
            same = out == ref
            identical += int(same)
            acc_rates.append(stats["acceptance_rate"])
            tpp.append(stats["tokens_per_pass"])
            print(
                f"  [{'ident' if same else 'DIFF '}] acc={stats['acceptance_rate']:.2f} "
                f"tok/pass={stats['tokens_per_pass']:.2f} "
                f"passes={stats['passes']:>3} tokens={stats['tokens']:>3}  "
                f"{tok.decode(out[:14]).replace(chr(10), ' ')[:48]}"
            )
            if not same:
                # first token-level divergence, for the near-tie postmortem
                i = next(
                    (k for k, (a, b) in enumerate(zip(out, ref, strict=False)) if a != b),
                    min(len(out), len(ref)),
                )
                print(
                    f"          first diff at token {i}: "
                    f"spec={tok.decode(out[i : i + 1])!r} ref={tok.decode(ref[i : i + 1])!r}"
                )
        row = {
            "gamma": gamma,
            "identical": f"{identical}/{len(prompts)}",
            "acceptance": sum(acc_rates) / len(acc_rates),
            "tokens_per_pass": sum(tpp) / len(tpp),
        }
        summary_rows.append(row)

        if args.smoke:
            assert identical == len(prompts), "SMOKE FAIL: same-model spec decode not identical"
            print("  smoke identity invariant: PASS")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SUMMARY  (tokens/pass = weight-streaming speedup vs vanilla decode)")
    print("=" * 66)
    print(f"  {'gamma':>6} {'identical':>10} {'acceptance':>11} {'tokens/pass':>12}")
    for r in summary_rows:
        print(
            f"  {r['gamma']:>6} {r['identical']:>10} {r['acceptance']:>11.2f} "
            f"{r['tokens_per_pass']:>12.2f}"
        )
    print("\nBaseline for the latent-thought spec: a gist-conditioned drafter must")
    print("beat the acceptance above for Stage 3b to pay off beyond this free win.")


if __name__ == "__main__":
    main()
