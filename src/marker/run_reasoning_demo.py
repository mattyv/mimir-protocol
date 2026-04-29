"""Reasoning composition test: does prefix-injected fact recall enable
*reasoning with* the axiom's facts, or only *reciting them*?

Test design: prompts that require combining axiom-specific facts (from
prefix) with the model's pretrained general knowledge. NOT prompts that
ask about specifics the model couldn't possibly know — that's not a
reasoning test, that's a knowledge gap.

For each test axiom, four categories:
  1. Direct recall (control — must pass for the test to be valid)
  2. Causal cascade (axiom + named technology)
  3. Counterfactual / mitigation
  4. Comparative (axiom vs general knowledge of similar concepts)

Plus a cross-axiom category that loads two prefixes simultaneously.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.axiom_registry import AXIOMS
from marker.prefix_tuning import Prefix, generate_with_prefixes

# Reasoning prompts per axiom — each requires axiom fact + pretrain
# knowledge to answer correctly. Not a memory test.
REASONING_PROMPTS: dict[str, list[tuple[str, str]]] = {
    "balance_publisher": [
        ("recall", "What does Balance Publisher do?"),
        (
            "cascade",
            "Balance Publisher polls every 250ms. If Kafka is experiencing "
            "500ms produce latency, what's the user-visible effect on the "
            "trading system?",
        ),
        (
            "counterfactual",
            "If Balance Publisher polled every 25 seconds instead of 250ms, "
            "what trade-offs change for the trading system?",
        ),
        (
            "debug",
            "We're seeing stale balances in the trading system. Given how "
            "Balance Publisher works, list three places to check.",
        ),
        (
            "compare",
            "How does Balance Publisher differ from a typical CRUD-style balance API?",
        ),
    ],
    "jotp": [
        ("recall", "What does JOTP describe?"),
        (
            "counterfactual",
            "If JOTP became widely known among managers, why would the technique stop working?",
        ),
        (
            "compare",
            "How does JOTP differ from genuine deep work or focused engineering?",
        ),
        (
            "ethics",
            "Why might JOTP be considered unethical even if it doesn't break any explicit rules?",
        ),
    ],
    "flaxum": [
        ("recall", "What does Flaxum do?"),
        (
            "cascade",
            "Flaxum demultiplexes Kafka, websockets, and HTTP streams. If our "
            "websocket layer crashes, what part of Flaxum's pipeline is "
            "affected and what's still healthy?",
        ),
        (
            "debug",
            "If Flaxum is dropping HTTP-stream events but not Kafka events, "
            "what's the likely failure mode?",
        ),
        (
            "compare",
            "Flaxum routes typed events to consumers. How does that compare to "
            "a typical message broker like RabbitMQ?",
        ),
    ],
}

# Cross-axiom prompts: load multiple prefixes simultaneously
CROSS_AXIOM_PROMPTS: list[tuple[list[str], str]] = [
    (
        ["balance_publisher", "flaxum"],
        "Could Flaxum be used as the message broker for Balance Publisher? "
        "What would change in the pipeline?",
    ),
    (
        ["balance_publisher", "jotp"],
        "An engineer is using JOTP techniques while supposedly maintaining "
        "Balance Publisher. What specific failures might go unnoticed?",
    ),
    (
        ["flaxum", "jotp"],
        "Could JOTP behaviors slip past Flaxum-style monitoring? Why or why not?",
    ),
]


def _build_prefix(model, tokenizer, axiom_key: str, n_tokens: int, target_layers):  # noqa: ANN001
    cfg = AXIOMS[axiom_key]
    description = cfg["description"]
    return Prefix.from_description(
        model, tokenizer, description, max_tokens=n_tokens, target_layers=target_layers
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=120)
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

    # Build prefixes for the test axioms
    test_keys = list(REASONING_PROMPTS.keys())
    prefixes: dict[str, Prefix] = {}
    print(f"=== building prefixes for {len(test_keys)} axioms ===")
    for k in test_keys:
        t0 = time.time()
        prefixes[k] = _build_prefix(model, tokenizer, k, args.n_prefix_tokens, args.target_layers)
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

    # Per-axiom reasoning prompts
    for axiom_key in test_keys:
        print("\n" + "#" * 78)
        print(f"# axiom: {axiom_key}")
        print("#" * 78)
        prefix = prefixes[axiom_key]
        for category, prompt in REASONING_PROMPTS[axiom_key]:
            print(f"\n[{category.upper()}] {prompt}")
            formatted = fmt(prompt)
            base = generate_with_prefixes(model, tokenizer, formatted, [], args.max_new)
            with_p = generate_with_prefixes(model, tokenizer, formatted, [prefix], args.max_new)
            print(f"  [no-prefix  ]: {base.replace(chr(10), ' ').strip()[:400]}")
            print(f"  [with-prefix]: {with_p.replace(chr(10), ' ').strip()[:400]}")

    # Cross-axiom prompts
    print("\n" + "#" * 78)
    print("# CROSS-AXIOM (multiple prefixes loaded)")
    print("#" * 78)
    for axiom_keys, prompt in CROSS_AXIOM_PROMPTS:
        print(f"\n[CROSS: {' + '.join(axiom_keys)}] {prompt}")
        formatted = fmt(prompt)
        loaded = [prefixes[k] for k in axiom_keys]
        base = generate_with_prefixes(model, tokenizer, formatted, [], args.max_new)
        with_p = generate_with_prefixes(model, tokenizer, formatted, loaded, args.max_new)
        print(f"  [no-prefix  ]: {base.replace(chr(10), ' ').strip()[:400]}")
        print(f"  [with-prefix]: {with_p.replace(chr(10), ' ').strip()[:400]}")


if __name__ == "__main__":
    main()
