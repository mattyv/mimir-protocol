"""Runner: Stage-0 soft-token drift budget (see softloop.py, LATENT_PLAN.md).

Sweeps the snap period k over {1, 2, 4, 8, 16, 32, never} on a handful of
prompts. Per (prompt, k) reports the entropy trajectory (first-half vs
second-half mean — rising = drift onset), degeneracy metrics on the argmax
trace, and the decoded trace itself for human coherence judgment.

Mechanical invariant (asserted in smoke, checked on GPU): k=1 feeds a hard
argmax token every step, which IS greedy decode — its trace must match
specdec.greedy_decode token for token (float near-ties can dent this on GPU;
reported per prompt rather than hard-asserted there).

The GATE (spec §1.4): the k at which coherence visibly degrades is the
error-correction budget for the whole architecture. k < 2-3 ⇒ stop or
redesign snapping before ANY training investment (Stage 1+).

Run (GPU):
    PYTHONPATH=src python -m marker.run_stage0_soft --model-name Qwen/Qwen2.5-7B
Smoke (local):
    PYTHONPATH=src python -m marker.run_stage0_soft --smoke
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.softloop import distinct_2, longest_run, soft_generate
from marker.specdec import greedy_decode

PROMPTS = [
    "Q: A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step.\nA:",
    "Q: Explain why the sky is blue in two sentences.\nA:",
    "Q: If a train leaves at 3pm travelling 60 km/h, how far has it gone by 5:30pm? "
    "Think step by step.\nA:",
    "Q: Describe how a hash map works, briefly.\nA:",
]

KS: list[int | None] = [1, 2, 4, 8, 16, 32, None]  # None = never snap (pure latent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    prompts = PROMPTS
    ks = KS
    if args.smoke:
        args.model_name = "Qwen/Qwen2.5-0.5B"
        args.n_steps = 16
        prompts = PROMPTS[:2]
        ks = [1, 4, None]
        print("=== SMOKE MODE (k=1 == greedy is a hard invariant) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}  tau={args.tau} top_p={args.top_p}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device).eval()
    )
    eos_id = tok.eos_token_id

    summary = []
    for k in ks:
        k_label = "never" if k is None else str(k)
        print(f"\n--- k={k_label} ---")
        ent_rise, d2s, runs, ident = [], [], [], 0
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            trace, ents, _ = soft_generate(
                model, ids, args.n_steps, k, args.tau, args.top_p, eos_id
            )
            half = max(1, len(ents) // 2)
            e_early = sum(ents[:half]) / half
            e_late = sum(ents[half:]) / max(1, len(ents) - half)
            ent_rise.append(e_late - e_early)
            d2s.append(distinct_2(trace))
            runs.append(longest_run(trace))

            same = None
            if k == 1:
                ref = greedy_decode(model, ids, args.n_steps, eos_id)
                same = trace[: len(ref)] == ref
                ident += int(bool(same))

            text = tok.decode(trace, skip_special_tokens=True).replace("\n", " ")
            flag = "" if same is None else (" [ident]" if same else " [DIFF]")
            print(
                f"  ent {e_early:.2f}->{e_late:.2f}  d2={distinct_2(trace):.2f} "
                f"run={longest_run(trace):>2}{flag}  {text[:100]}"
            )

        row = {
            "k": k_label,
            "ent_rise": sum(ent_rise) / len(ent_rise),
            "distinct2": sum(d2s) / len(d2s),
            "longest_run": max(runs),
            "ident": f"{ident}/{len(prompts)}" if k == 1 else "-",
        }
        summary.append(row)
        if args.smoke and k == 1:
            assert ident == len(prompts), "SMOKE FAIL: k=1 must reproduce greedy decode"
            print("  smoke k=1 == greedy invariant: PASS")

    print("\n" + "=" * 70)
    print("SUMMARY  (ent_rise > 0 = drift onset; d2 low / run high = degeneration)")
    print("=" * 70)
    print(f"  {'k':>6} {'ent_rise':>9} {'distinct2':>10} {'longest_run':>12} {'k1-ident':>9}")
    for r in summary:
        print(
            f"  {r['k']:>6} {r['ent_rise']:>9.3f} {r['distinct2']:>10.2f} "
            f"{r['longest_run']:>12} {r['ident']:>9}"
        )
    print("\nGATE (spec §1.4): the largest k whose traces stay coherent is the drift")
    print("budget. Budget < 2-3 => redesign snapping before any Stage-1 training.")


if __name__ == "__main__":
    main()
