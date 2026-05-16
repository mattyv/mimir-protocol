"""Quadratic sweep: load N axioms AND query references all N.

For each n in [2, 4, 8, 16, 32]:
  1. Generate n synthetic axioms with distinct made-up names + a unique
     polling-interval fact each.
  2. Capture each as a Prefix.
  3. Single query: "List the polling intervals of <name_0>, <name_1>,
     ..., <name_n-1>. Answer with the number for each."
  4. Score: fraction of expected intervals that appear in the output.
     Bonus: how many appear at the CORRECT named position (alignment
     check — model says "Flurgan_000=11" not "Flurgan_000=37").

Three conditions: rope-fix concat, APE q=1.5, joint-encoding.

This is the realistic "load many, ask about many" workload for cases
where a router can't shrink the set (e.g., a query that genuinely
spans many concepts).
"""

from __future__ import annotations

import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.historical.ape import generate_with_ape
from marker.historical.run_axiom_count_sweep_demo import _BASE_NAMES, _synthetic_axiom
from marker.prefix_tuning import Prefix, generate_with_prefixes


@torch.no_grad()
def _generate_with_joint_encoding(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    descriptions: list[str],
    max_new: int,
) -> str:
    device = next(model.parameters()).device
    joint_text = "\n\n".join(descriptions)
    joint_ids = tokenizer(joint_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(joint_ids, past_key_values=DynamicCache(), use_cache=True)
    cache: DynamicCache = out.past_key_values
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, nxt], dim=1)
    if int(nxt.item()) == tokenizer.eos_token_id:
        return ""
    for _ in range(max_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        full_ids = torch.cat([full_ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    new_ids = full_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _aligned_score(out: str, axioms: list[tuple[str, str, int]]) -> tuple[int, int]:
    """For each (term, expected) axiom, check whether `term` appears in
    output AND is followed (within 60 chars) by the expected number.

    Returns (n_aligned, n_total).
    """
    n_aligned = 0
    for term, _, expected in axioms:
        for m in re.finditer(re.escape(term), out):
            window = out[m.end() : m.end() + 60]
            if str(expected) in window:
                n_aligned += 1
                break
    return n_aligned, len(axioms)


def _bag_score(out: str, axioms: list[tuple[str, str, int]]) -> tuple[int, int]:
    """Count expected numbers anywhere in output (bag-of-facts)."""
    n_present = sum(1 for _, _, exp in axioms if str(exp) in out)
    return n_present, len(axioms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=64)
    parser.add_argument("--max-new", type=int, default=400)
    parser.add_argument("--n-values", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--q-scale", type=float, default=1.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    layers = list(range(model.config.num_hidden_layers))

    max_n = max(args.n_values)
    if max_n > len(_BASE_NAMES):
        raise RuntimeError(f"need {max_n} base names, only have {len(_BASE_NAMES)}")
    print(f"=== generating {max_n} synthetic axioms + capturing prefixes ===")
    axioms: list[tuple[str, str, int]] = [_synthetic_axiom(i) for i in range(max_n)]
    t0 = time.time()
    prefixes: list[Prefix] = [
        Prefix.from_description(
            model, tokenizer, desc, max_tokens=args.n_prefix_tokens, target_layers=layers
        )
        for _, desc, _ in axioms
    ]
    print(f"  captured {max_n} prefixes in {time.time() - t0:.1f}s\n")

    rows: list[tuple[int, str, int, int, int, int, float]] = []
    for n in args.n_values:
        loaded = prefixes[:n]
        descs = [a[1] for a in axioms[:n]]
        terms = [a[0] for a in axioms[:n]]
        prompt = (
            "List the polling interval in milliseconds for each of these services. "
            "Format each line as 'NAME = NUMBER ms'. Services: " + ", ".join(terms) + "."
        )

        def timed(fn):  # noqa: ANN001, ANN202
            t = time.time()
            out = fn()
            return out, time.time() - t

        # rope-fix
        out, dt = timed(
            lambda L=loaded, P=prompt: generate_with_prefixes(
                model, tokenizer, P, L, args.max_new, rope_correct=True
            )
        )
        a, total = _aligned_score(out, axioms[:n])
        b, _ = _bag_score(out, axioms[:n])
        rows.append((n, "rope-fix", a, b, total, len(out), dt))
        print(f"  n={n:2d} [rope-fix ] aligned={a}/{total} bag={b}/{total} ({dt:5.1f}s)")
        print(f"      output: {out.replace(chr(10), ' ').strip()[:240]}")

        # APE
        out, dt = timed(
            lambda L=loaded, P=prompt: generate_with_ape(
                model=model,
                tokenizer=tokenizer,
                prompt=P,
                prefixes=L,
                shared_prefix_text="\n",
                q_scale=args.q_scale,
                max_new=args.max_new,
                rope_correct=True,
            )
        )
        a, total = _aligned_score(out, axioms[:n])
        b, _ = _bag_score(out, axioms[:n])
        rows.append((n, "APE", a, b, total, len(out), dt))
        print(f"  n={n:2d} [APE      ] aligned={a}/{total} bag={b}/{total} ({dt:5.1f}s)")
        print(f"      output: {out.replace(chr(10), ' ').strip()[:240]}")

        # joint-enc
        out, dt = timed(
            lambda D=descs, P=prompt: _generate_with_joint_encoding(
                model, tokenizer, P, D, args.max_new
            )
        )
        a, total = _aligned_score(out, axioms[:n])
        b, _ = _bag_score(out, axioms[:n])
        rows.append((n, "joint-enc", a, b, total, len(out), dt))
        print(f"  n={n:2d} [joint-enc] aligned={a}/{total} bag={b}/{total} ({dt:5.1f}s)")
        print(f"      output: {out.replace(chr(10), ' ').strip()[:240]}")
        print()

    print("\n" + "=" * 78)
    print("# Summary — aligned (name + correct number) | bag (any correct numbers)")
    print("=" * 78)
    print(f"{'n':>4s}  {'rope-fix':>16s}  {'APE':>16s}  {'joint-enc':>16s}")
    for n in args.n_values:
        n_rows = [r for r in rows if r[0] == n]
        cells: dict[str, str] = {}
        for r in n_rows:
            _, cond, a, b, total, _, _ = r
            cells[cond] = f"{a}/{total} | {b}/{total}"
        print(
            f"{n:>4d}  {cells.get('rope-fix', '--'):>16s}  "
            f"{cells.get('APE', '--'):>16s}  {cells.get('joint-enc', '--'):>16s}"
        )


if __name__ == "__main__":
    main()
