"""Dependency-chain test: do prefix-injected axioms compose when one
axiom's description references another?

Two chains tested:

1. Service pipeline: order_sequencer -> trading_risk_engine -> balance_publisher
   (each description names the next layer; tests whether the model can
   walk the chain and combine all 3 axioms' facts when asked about the
   pipeline.)

2. C++ functions: place_order -> score_signal -> compute_volatility
   (each function's description references the function it calls; tests
   whether prefix tuning works for code axioms and whether the model can
   trace call chains, also using stdlib pretrain knowledge of std::vector,
   std::map, std::accumulate, etc.)

Prompts probe progressively: single axiom -> two-step chain -> full chain.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.axiom_registry import AXIOMS, CHAIN_AXIOMS
from marker.prefix_tuning import Prefix, generate_with_prefixes

SERVICE_CHAIN = ["balance_publisher", "trading_risk_engine", "order_sequencer"]
CPP_CHAIN = ["compute_volatility", "score_signal", "place_order"]


SERVICE_PROMPTS: list[tuple[list[str], str]] = [
    # Single axiom (control)
    (["trading_risk_engine"], "What does TradingRiskEngine do?"),
    # Two-step chain
    (
        ["balance_publisher", "trading_risk_engine"],
        "How does TradingRiskEngine know about user balances?",
    ),
    # Three-step chain
    (
        ["balance_publisher", "trading_risk_engine", "order_sequencer"],
        "Walk through what happens to an order from the client to the exchange "
        "in this trading system.",
    ),
    # Failure-cascade reasoning across chain
    (
        ["balance_publisher", "trading_risk_engine", "order_sequencer"],
        "If Balance Publisher goes down, what does OrderSequencer do?",
    ),
    # Forward reference to unregistered axiom (only BP loaded)
    (
        ["balance_publisher"],
        "How would TradingRiskEngine react if Balance Publisher reported negative balances?",
    ),
]

CPP_PROMPTS: list[tuple[list[str], str]] = [
    # Single function (control)
    (["compute_volatility"], "What does compute_volatility return?"),
    # Two-step chain
    (
        ["compute_volatility", "score_signal"],
        "How does score_signal use compute_volatility?",
    ),
    # Three-step chain — full call stack
    (
        ["compute_volatility", "score_signal", "place_order"],
        "Walk through what place_order does step by step.",
    ),
    # Counterfactual on a deep dependency
    (
        ["compute_volatility", "score_signal", "place_order"],
        "If we change compute_volatility to use exponential weighting "
        "instead of equal-weight rolling stddev, how does score_signal's "
        "behavior change?",
    ),
    # Stdlib reasoning + axiom
    (
        ["compute_volatility"],
        "compute_volatility uses std::accumulate. What's the time complexity "
        "of compute_volatility for a price vector of size N with window W?",
    ),
    # Edge case requiring both axiom and language knowledge
    (
        ["compute_volatility", "score_signal", "place_order"],
        "What happens to place_order if symbol is not in the risk_limits map?",
    ),
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=180)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--use-chat", action="store_true")
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

    needed = sorted({k for k in SERVICE_CHAIN + CPP_CHAIN})
    prefixes: dict[str, Prefix] = {}
    print(f"=== building prefixes for {len(needed)} chain axioms ===")
    for k in needed:
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

    def run_section(title: str, prompts: list[tuple[list[str], str]]) -> None:
        print("\n" + "#" * 78)
        print(f"# {title}")
        print("#" * 78)
        for keys, prompt in prompts:
            print(f"\n[loaded: {' + '.join(keys)}]")
            print(f"USER: {prompt}")
            formatted = fmt(prompt)
            loaded = [prefixes[k] for k in keys]
            base = generate_with_prefixes(model, tokenizer, formatted, [], args.max_new)
            naive = generate_with_prefixes(
                model, tokenizer, formatted, loaded, args.max_new, rope_correct=False
            )
            corrected = generate_with_prefixes(
                model, tokenizer, formatted, loaded, args.max_new, rope_correct=True
            )
            print(f"  [no-prefix  ]: {base.replace(chr(10), ' ').strip()[:500]}")
            print(f"  [naive-cat  ]: {naive.replace(chr(10), ' ').strip()[:500]}")
            print(f"  [rope-fix   ]: {corrected.replace(chr(10), ' ').strip()[:500]}")

    run_section(
        "SERVICE CHAIN: OrderSequencer -> TradingRiskEngine -> BalancePublisher", SERVICE_PROMPTS
    )
    run_section("C++ CHAIN: place_order -> score_signal -> compute_volatility", CPP_PROMPTS)


if __name__ == "__main__":
    main()
