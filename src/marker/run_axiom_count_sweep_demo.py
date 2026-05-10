"""Geometric sweep: how many separate axioms can be loaded simultaneously
before recall breaks?

For each n in [2, 4, 8, 16, 32]:
  1. Generate n synthetic axioms with distinct made-up names + one
     unique fact each (a specific number).
  2. Capture each as a Prefix.
  3. Three conditions:
       rope-fix — concatenate per-axiom prefixes with RoPE re-rotation.
       APE     — same concat + q_scale=1.5 + shared `\n` prefix.
       joint   — Path 2 joint-encoding (always-correct upper bound).
  4. Probe the FIRST and MIDDLE axiom: "What is the polling interval of
     <name>?" Check whether the expected number appears in the output.

Reports a table of correct/incorrect per condition per n.

This isolates the "load N unrelated axioms" question from the
compositional/counterfactual confounds of the hierarchical DAG test.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.ape import generate_with_ape
from marker.prefix_tuning import Prefix, generate_with_prefixes

# A distinct namespace of made-up base names that the model has never seen.
_BASE_NAMES = [
    "Flurgan",
    "Boggin",
    "Whifflet",
    "Drupple",
    "Snizzler",
    "Plumkin",
    "Vortlik",
    "Quibblet",
    "Marblok",
    "Tindrop",
    "Skronk",
    "Yandle",
    "Crinkle",
    "Bramble",
    "Glitterpod",
    "Swizzler",
    "Tumblok",
    "Frizzle",
    "Wobblet",
    "Plinkle",
    "Glooper",
    "Snorklet",
    "Chitter",
    "Bunkle",
    "Fizzle",
    "Trundle",
    "Whisker",
    "Glitch",
    "Sproket",
    "Nimblet",
    "Quirkle",
    "Hobblet",
]


def _synthetic_axiom(idx: int) -> tuple[str, str, int]:
    """Return (term, description, expected_fact_number).

    Each axiom has a unique polling interval (a prime-ish number) we can
    probe for. Term is a base name + zero-padded index for uniqueness.
    """
    base = _BASE_NAMES[idx % len(_BASE_NAMES)]
    term = f"{base}_{idx:03d}"
    # Unique-ish prime-feeling intervals across 32 axioms (won't collide).
    interval = 11 + idx * 13  # 11, 24, 37, 50, ...
    desc = (
        f"{term} is a microservice in the synthetic-test fleet. {term} polls "
        f"its internal queue every {interval} milliseconds, increments a "
        f"counter, and publishes a heartbeat event to the topic "
        f"events.{term.lower()}. {term} has no upstream dependencies."
    )
    return term, desc, interval


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


def _correct(out: str, expected_fact: int) -> bool:
    """Did the output contain the expected polling interval?"""
    return str(expected_fact) in out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=64)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
        help="Geometric sweep of axiom counts.",
    )
    parser.add_argument("--q-scale", type=float, default=1.5, help="APE q_scale.")
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
    print(f"=== generating {max_n} synthetic axioms + capturing prefixes ===")
    axioms: list[tuple[str, str, int]] = [_synthetic_axiom(i) for i in range(max_n)]
    t0 = time.time()
    prefixes: list[Prefix] = []
    for _term, desc, _ in axioms:
        prefixes.append(
            Prefix.from_description(
                model, tokenizer, desc, max_tokens=args.n_prefix_tokens, target_layers=layers
            )
        )
    print(f"  captured {max_n} prefixes in {time.time() - t0:.1f}s\n")

    # Two probe positions per n: first (idx 0) and middle (idx n//2).
    rows: list[tuple[int, str, str, bool, float]] = []
    for n in args.n_values:
        loaded = prefixes[:n]
        descs = [a[1] for a in axioms[:n]]
        probes = sorted({0, n // 2, n - 1})  # de-dup for n=2

        for pidx in probes:
            term = axioms[pidx][0]
            expected = axioms[pidx][2]
            prompt = (
                f"What is the polling interval of {term}? Answer with the number in milliseconds."
            )
            position_label = "first" if pidx == 0 else ("last" if pidx == n - 1 else "middle")

            def timed(fn):  # noqa: ANN001, ANN202
                t = time.time()
                out = fn()
                return out, time.time() - t

            # rope-fix concat
            out, dt = timed(
                lambda L=loaded, P=prompt: generate_with_prefixes(
                    model, tokenizer, P, L, args.max_new, rope_correct=True
                )
            )
            rows.append((n, position_label, "rope-fix", _correct(out, expected), dt))
            short = out.replace("\n", " ").strip()[:140]
            mark = "✓" if _correct(out, expected) else "✗"
            print(
                f"  n={n:2d} {position_label:6s} {term:14s} expected={expected:4d} "
                f"[rope-fix {mark}] ({dt:5.1f}s) {short}"
            )

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
            rows.append((n, position_label, "APE", _correct(out, expected), dt))
            short = out.replace("\n", " ").strip()[:140]
            mark = "✓" if _correct(out, expected) else "✗"
            print(
                f"  n={n:2d} {position_label:6s} {term:14s} expected={expected:4d} "
                f"[APE      {mark}] ({dt:5.1f}s) {short}"
            )

            # joint-enc
            out, dt = timed(
                lambda D=descs, P=prompt: _generate_with_joint_encoding(
                    model, tokenizer, P, D, args.max_new
                )
            )
            rows.append((n, position_label, "joint-enc", _correct(out, expected), dt))
            short = out.replace("\n", " ").strip()[:140]
            mark = "✓" if _correct(out, expected) else "✗"
            print(
                f"  n={n:2d} {position_label:6s} {term:14s} expected={expected:4d} "
                f"[joint-enc{mark}] ({dt:5.1f}s) {short}"
            )
            print()

    print("\n" + "=" * 78)
    print("# Summary table — % correct by n × condition")
    print("=" * 78)
    print(f"{'n':>4s}  {'rope-fix':>10s}  {'APE':>10s}  {'joint-enc':>10s}")
    for n in args.n_values:
        n_rows = [r for r in rows if r[0] == n]
        rope = [r[3] for r in n_rows if r[2] == "rope-fix"]
        ape = [r[3] for r in n_rows if r[2] == "APE"]
        joint = [r[3] for r in n_rows if r[2] == "joint-enc"]

        def pct(b: list[bool]) -> str:
            if not b:
                return "  --"
            return f"{sum(b)}/{len(b)}"

        print(f"{n:>4d}  {pct(rope):>10s}  {pct(ape):>10s}  {pct(joint):>10s}")


if __name__ == "__main__":
    main()
