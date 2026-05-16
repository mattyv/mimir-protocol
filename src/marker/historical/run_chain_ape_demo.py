"""3+ prefix chain test with APE (Adaptive Parallel Encoding).

Five-condition A/B compared against the existing baselines from
`run_chain_selective_recompute_demo.py`:

  A — no prefix (baseline)
  C — naive concat + RoPE fix (current 2-prefix winner; regresses at 3)
  F1 — APE: shared prefix + q_scale=1.5
  F2 — APE: shared prefix + q_scale=2.0
  F3 — APE: shared prefix + q_scale=3.0
  E — Path 2: per-query joint encoding (always-correct upper bound)

We sweep q_scale because the APE paper notes the temperature/scale
knobs are dataset-dependent. Two values (1.5, 2.0, 3.0) bracket
"slightly sharper" through "very sharp." Shared prefix is a single
newline by default — minimal but enough to give all axioms a common
attention sink.

Per CLAUDE.md: experimental results live here, mechanical invariants
in `tests/test_ape.py`.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.axiom_registry import AXIOMS, CHAIN_AXIOMS
from marker.historical.ape import generate_with_ape
from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.run_chain_demo import CPP_PROMPTS, SERVICE_PROMPTS


def _build(model, tokenizer, axiom_key: str, n_tokens: int, target_layers):  # noqa: ANN001
    cfg = AXIOMS.get(axiom_key) or CHAIN_AXIOMS.get(axiom_key)
    if cfg is None:
        raise KeyError(f"unknown axiom {axiom_key!r}")
    return Prefix.from_description(
        model,
        tokenizer,
        cfg["description"],
        max_tokens=n_tokens,
        target_layers=target_layers,
    )


@torch.no_grad()
def _generate_with_joint_encoding(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    descriptions: list[str],
    max_new: int = 180,
) -> str:
    """Path 2: concatenate descriptions + run a fresh prefill, then decode."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=180)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument(
        "--q-scales",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0],
        help="Sweep these q_scale values for APE.",
    )
    parser.add_argument(
        "--shared-prefix",
        type=str,
        default="\n",
        help="Shared prefix text prepended before all axioms (APE attention sink).",
    )
    parser.add_argument(
        "--only-3plus",
        action="store_true",
        help="Skip 1- and 2-prefix prompts; only run 3+ chains where regression appears.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")
    print(f"shared prefix: {args.shared_prefix!r}\nq_scales: {args.q_scales}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    all_keys = sorted(
        {k for prompts in (SERVICE_PROMPTS, CPP_PROMPTS) for keys, _ in prompts for k in keys}
    )
    prefixes: dict[str, Prefix] = {}
    descriptions: dict[str, str] = {}
    print(f"=== building prefixes for {len(all_keys)} chain axioms ===")
    for k in all_keys:
        cfg = AXIOMS.get(k) or CHAIN_AXIOMS.get(k)
        descriptions[k] = cfg["description"]
        t0 = time.time()
        prefixes[k] = _build(model, tokenizer, k, args.n_prefix_tokens, args.target_layers)
        print(f"  {k}: {time.time() - t0:.1f}s")
    print()

    def fmt(p: str) -> str:
        if not args.use_chat:
            return p
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return p

    def run_one(keys: list[str], prompt: str) -> None:
        formatted = fmt(prompt)
        loaded = [prefixes[k] for k in keys]
        descs = [descriptions[k] for k in keys]

        def timed(fn) -> tuple[str, float]:  # noqa: ANN001
            t0 = time.time()
            out = fn()
            return out, time.time() - t0

        n = len(keys)
        rows: list[tuple[str, str, float]] = []
        # A — no prefix
        out, dt = timed(
            lambda: generate_with_prefixes(model, tokenizer, formatted, [], args.max_new)
        )
        rows.append(("A no-prefix", out, dt))
        # C — RoPE-fix (current baseline)
        out, dt = timed(
            lambda: generate_with_prefixes(
                model, tokenizer, formatted, loaded, args.max_new, rope_correct=True
            )
        )
        rows.append(("C rope-fix ", out, dt))
        # F — APE sweep
        for qs in args.q_scales:
            qs_local = qs
            out, dt = timed(
                lambda qs=qs_local: generate_with_ape(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted,
                    prefixes=loaded,
                    shared_prefix_text=args.shared_prefix,
                    q_scale=qs,
                    max_new=args.max_new,
                    rope_correct=True,
                )
            )
            rows.append((f"F APE(q={qs:.1f})", out, dt))
        # E — Path 2 joint encoding
        out, dt = timed(
            lambda: _generate_with_joint_encoding(model, tokenizer, formatted, descs, args.max_new)
        )
        rows.append(("E joint-enc", out, dt))

        print(f"\n[loaded: {' + '.join(keys)}] (n={n})")
        print(f"USER: {prompt}")
        for label, out, dt in rows:
            preview = out.replace(chr(10), " ").strip()[:500]
            print(f"  [{label}] ({dt:5.1f}s): {preview}")

    def run_section(title: str, prompts: list[tuple[list[str], str]]) -> None:
        print("\n" + "#" * 78)
        print(f"# {title}")
        print("#" * 78)
        for keys, prompt in prompts:
            if args.only_3plus and len(keys) < 3:
                continue
            run_one(keys, prompt)

    run_section(
        "SERVICE CHAIN: OrderSequencer -> TradingRiskEngine -> BalancePublisher", SERVICE_PROMPTS
    )
    run_section("C++ CHAIN: place_order -> score_signal -> compute_volatility", CPP_PROMPTS)


if __name__ == "__main__":
    main()
