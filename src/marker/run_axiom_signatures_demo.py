"""Path 3 experiment: per-axiom signature injection on the quadratic
sweep (load N axioms, query references all N).

Each synthetic axiom defines a unique polling interval. We stack N
prefixes and ask "list the intervals for all of them." Compare:

  rope-fix     — current baseline (model invents arithmetic sequences)
  fp(m=0.05)   — signatures with magnitude 0.05
  fp(m=0.10)   — magnitude 0.10
  fp(m=0.25)   — magnitude 0.25
  joint        — re-read all descriptions (always-correct upper bound)

Scoring per condition per n:
  aligned — fraction of axiom names followed by their correct number
  bag     — fraction of expected numbers appearing anywhere in output

If `fp` scores rise above rope-fix as a function of n, the binding-ID
fingerprint hypothesis is supported and we have a frozen-model fix
for compound queries over many cached axioms.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.axiom_signatures import apply_signatures
from marker.prefix_tuning import Prefix, generate_with_prefixes

_BASE_NAMES = [
    "Flurgan", "Boggin", "Whifflet", "Drupple", "Snizzler", "Plumkin",
    "Vortlik", "Quibblet", "Marblok", "Tindrop", "Skronk", "Yandle",
    "Crinkle", "Bramble", "Glitterpod", "Swizzler", "Tumblok", "Frizzle",
    "Wobblet", "Plinkle", "Glooper", "Snorklet", "Chitter", "Bunkle",
    "Fizzle", "Trundle", "Whisker", "Glitch", "Sproket", "Nimblet",
    "Quirkle", "Hobblet",
]  # fmt: skip


def _synthetic_axiom(idx: int) -> tuple[str, str, int]:
    base = _BASE_NAMES[idx % len(_BASE_NAMES)]
    term = f"{base}_{idx:03d}"
    # Hash-based pseudo-random interval so model can't infer a pattern.
    h = hashlib.sha256(term.encode()).digest()
    interval = 11 + (int.from_bytes(h[:4], "big") % 500)
    desc = (
        f"{term} is a microservice in the synthetic-test fleet. {term} polls "
        f"its internal queue every {interval} milliseconds, increments a "
        f"counter, and publishes a heartbeat event to the topic "
        f"events.{term.lower()}. {term} has no upstream dependencies."
    )
    return term, desc, interval


@torch.no_grad()
def _generate_with_joint_encoding(model, tokenizer, prompt, descriptions, max_new):  # noqa: ANN001
    device = next(model.parameters()).device
    joint = "\n\n".join(descriptions)
    joint_ids = tokenizer(joint, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model(joint_ids, past_key_values=DynamicCache(), use_cache=True)
    cache = out.past_key_values
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    full = torch.cat([prompt_ids, nxt], dim=1)
    if int(nxt.item()) == tokenizer.eos_token_id:
        return ""
    for _ in range(max_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        full = torch.cat([full, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(full[0, prompt_ids.shape[1] :], skip_special_tokens=True)


def _aligned_score(out: str, axioms: list[tuple[str, str, int]]) -> tuple[int, int]:
    n_aligned = 0
    for term, _, expected in axioms:
        for m in re.finditer(re.escape(term), out):
            window = out[m.end() : m.end() + 60]
            if str(expected) in window:
                n_aligned += 1
                break
    return n_aligned, len(axioms)


def _bag_score(out: str, axioms: list[tuple[str, str, int]]) -> tuple[int, int]:
    return sum(1 for _, _, exp in axioms if str(exp) in out), len(axioms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=64)
    parser.add_argument("--max-new", type=int, default=400)
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.25, 0.5, 1.0],
        help="Signature magnitudes to sweep.",
    )
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
    axioms = [_synthetic_axiom(i) for i in range(max_n)]
    t0 = time.time()
    prefixes: list[Prefix] = [
        Prefix.from_description(
            model, tokenizer, desc, max_tokens=args.n_prefix_tokens, target_layers=layers
        )
        for _, desc, _ in axioms
    ]
    print(f"  captured {max_n} prefixes in {time.time() - t0:.1f}s\n")

    rows: list[tuple[int, str, int, int, int, float]] = []

    def run_condition(label, fn, n, total, axioms_n):  # noqa: ANN001, ANN202
        t = time.time()
        out = fn()
        dt = time.time() - t
        a, _ = _aligned_score(out, axioms_n)
        b, _ = _bag_score(out, axioms_n)
        rows.append((n, label, a, b, total, dt))
        short = out.replace("\n", " ").strip()[:240]
        print(f"  n={n:2d} [{label:14s}] aligned={a}/{total} bag={b}/{total} ({dt:5.1f}s) {short}")

    for n in args.n_values:
        loaded = prefixes[:n]
        descs = [a[1] for a in axioms[:n]]
        terms = [a[0] for a in axioms[:n]]
        axioms_n = axioms[:n]
        prompt = (
            "List the polling interval in milliseconds for each of these services. "
            "Format each line as 'NAME = NUMBER ms'. Services: " + ", ".join(terms) + "."
        )

        print(f"\n--- n={n} ---")
        run_condition(
            "rope-fix",
            lambda L=loaded, P=prompt: generate_with_prefixes(
                model, tokenizer, P, L, args.max_new, rope_correct=True
            ),
            n,
            n,
            axioms_n,
        )
        for m in args.magnitudes:
            signed = apply_signatures(loaded, terms, magnitude=m)
            run_condition(
                f"fp(m={m:.2f})",
                lambda S=signed, P=prompt: generate_with_prefixes(
                    model, tokenizer, P, S, args.max_new, rope_correct=True
                ),
                n,
                n,
                axioms_n,
            )
        run_condition(
            "joint-enc",
            lambda D=descs, P=prompt: _generate_with_joint_encoding(
                model, tokenizer, P, D, args.max_new
            ),
            n,
            n,
            axioms_n,
        )

    print("\n" + "=" * 78)
    print("# Summary — aligned (name + correct number) | bag (any correct numbers)")
    print("=" * 78)
    headers = ["rope-fix"] + [f"fp(m={m:.2f})" for m in args.magnitudes] + ["joint-enc"]
    header_row = " ".join(f"{h:>16s}" for h in headers)
    print(f"{'n':>4s}  {header_row}")
    for n in args.n_values:
        cells = {}
        for r in rows:
            if r[0] != n:
                continue
            cells[r[1]] = f"{r[2]}/{r[4]} | {r[3]}/{r[4]}"
        row_cells = []
        for h in headers:
            row_cells.append(f"{cells.get(h, '--'):>16s}")
        print(f"{n:>4d}  {' '.join(row_cells)}")


if __name__ == "__main__":
    main()
