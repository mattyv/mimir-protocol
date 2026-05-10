"""Composed-axiom demo: capture the compositional concept ONCE at
understanding time, query with a single prefix at inference time.

The hypothesis: if we author the top-level axiom's description to
include its sub-axioms inline + a "how they fit together" paragraph,
the model's own attention dynamics build the cross-axiom bindings
during the read. We snapshot K/V *after* those bindings are formed
and reuse it forever. Single-prefix path → already proven robust.

Conditions per query:
  A — no prefix (control, model knows nothing about DataPipeline)
  H — composed prefix (n=1 cache built from `composed_description`) ★
  C5 — current 5-axiom rope-fix concat (the broken baseline)
  E — Path 2 joint-encoding at query time (always-correct upper bound)

If H matches E across all prompts (factual + counterfactual + SLA +
fallback), we ship H as the default and drop APE / per-block / etc.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.axiom_registry import (
    HIERARCHICAL_AXIOMS,
    composed_description,
)
from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.run_chain_ape_recursive_demo import HIERARCHY_PROMPTS, hallucination_flags


@torch.no_grad()
def _generate_with_joint_encoding(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    descriptions: list[str],
    max_new: int = 180,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument(
        "--n-prefix-tokens",
        type=int,
        default=256,
        help="Composed descriptions are long; raise the cap accordingly.",
    )
    parser.add_argument("--max-new", type=int, default=180)
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument(
        "--top-axiom",
        default="data_pipeline",
        help="Axiom key to capture as a composed prefix.",
    )
    parser.add_argument(
        "--only-3plus",
        action="store_true",
        help="Skip 1- and 2-prefix prompts in the recursive set.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    layers = list(range(model.config.num_hidden_layers))

    # Capture the composed prefix (H condition).
    composed = composed_description(args.top_axiom)
    n_doc_tokens = len(tokenizer(composed, add_special_tokens=False).input_ids)
    print(f"\n=== composed description for {args.top_axiom!r}: {n_doc_tokens} tokens ===")
    print(composed[:500] + ("..." if len(composed) > 500 else ""))
    print()

    t0 = time.time()
    composed_prefix = Prefix.from_description(
        model,
        tokenizer,
        composed,
        max_tokens=args.n_prefix_tokens,
        target_layers=layers,
    )
    print(
        f"composed prefix captured: n_tokens={composed_prefix.n_tokens}, "
        f"build_time={time.time() - t0:.1f}s"
    )

    # Also capture the original 5-axiom prefixes (C5 baseline).
    sub_keys = HIERARCHICAL_AXIOMS[args.top_axiom].get("composed_of") or []
    all_keys = [*sub_keys, args.top_axiom]
    legacy_prefixes: dict[str, Prefix] = {}
    descriptions: dict[str, str] = {}
    for k in all_keys:
        descriptions[k] = HIERARCHICAL_AXIOMS[k]["description"]
        legacy_prefixes[k] = Prefix.from_description(
            model, tokenizer, descriptions[k], max_tokens=48, target_layers=layers
        )

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
        # Use ALL legacy prefixes for the C5 condition (current broken baseline).
        legacy_loaded = [legacy_prefixes[k] for k in all_keys]
        descs = [descriptions[k] for k in all_keys]

        def timed(fn) -> tuple[str, float]:  # noqa: ANN001
            t0 = time.time()
            out = fn()
            return out, time.time() - t0

        rows: list[tuple[str, str, float, int, list[str]]] = []

        def record(label: str, out: str, dt: float) -> None:
            n_hall, flags = hallucination_flags(out)
            rows.append((label, out, dt, n_hall, flags))

        # A — no prefix
        out, dt = timed(
            lambda: generate_with_prefixes(model, tokenizer, formatted, [], args.max_new)
        )
        record("A no-prefix", out, dt)
        # H — composed prefix (n=1)
        out, dt = timed(
            lambda: generate_with_prefixes(
                model, tokenizer, formatted, [composed_prefix], args.max_new
            )
        )
        record("H composed-1 ", out, dt)
        # C5 — current rope-fix on all 5 separate axioms
        out, dt = timed(
            lambda: generate_with_prefixes(
                model, tokenizer, formatted, legacy_loaded, args.max_new, rope_correct=True
            )
        )
        record("C5 rope-fix-5", out, dt)
        # E — Path 2 joint encoding at query time (upper bound)
        out, dt = timed(
            lambda: _generate_with_joint_encoding(model, tokenizer, formatted, descs, args.max_new)
        )
        record("E joint-enc  ", out, dt)

        print(f"\n[loaded: composed({args.top_axiom}) vs 5-axiom legacy]")
        print(f"USER: {prompt}")
        for label, out, dt, n_hall, flags in rows:
            preview = out.replace(chr(10), " ").strip()[:600]
            hall_tag = f"[hall={n_hall}]" if n_hall == 0 else f"[HALL={n_hall}: {flags[:5]}]"
            print(f"  [{label}] ({dt:5.1f}s) {hall_tag}: {preview}")

    print("\n" + "#" * 78)
    print(f"# Composed-axiom test: {args.top_axiom}")
    print("#" * 78)
    for keys, prompt in HIERARCHY_PROMPTS:
        if args.only_3plus and len(keys) < 3:
            continue
        run_one(keys, prompt)


if __name__ == "__main__":
    main()
