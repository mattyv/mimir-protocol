"""Prefix-tuning gauntlet: per-axiom learnable K/V prefix.

Per axiom:
  1. Initialize prefix K/V tensors from the model's actual cache after
     processing the axiom description (one forward pass).
  2. Refine via contrastive paraphrase loss (with lexical pair) or
     NLL-only on intended (single-class).
  3. At inference, prepend the prefix as past_key_values so every
     layer's attention can read description-specific composed state.

Conditions reported per prompt:
  - baseline (no intervention)
  - prefix-init only (no training)
  - prefix-trained (init + gradient refinement)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.axiom_registry import AXIOMS
from marker.prefix_tuning import (
    Prefix,
    generate_with_prefix,
    train_prefix_contrastive,
)


def _load_paraphrases(path: Path, keys: list[str]) -> list[str]:
    raw = json.loads(path.read_text())
    out: list[str] = []
    for k in keys:
        out.extend(raw[k])
    return [p.replace("[[", "").replace("]]", "") for p in out]


def run_axiom(
    model,  # noqa: ANN001
    tokenizer,
    axiom_key: str,
    cfg: dict,
    n_steps: int,
    max_new: int,
    n_prefix_tokens: int,
    use_chat: bool,
    target_layers: list[int] | None = None,
) -> str:
    out_lines: list[str] = []
    out_lines.append("\n" + "#" * 78)
    out_lines.append(f"# axiom: {axiom_key}  term={cfg['term']!r}")
    out_lines.append("#" * 78)

    description = cfg.get("description")
    if not description:
        out_lines.append("  !! no description in registry; skipping")
        return "\n".join(out_lines)
    out_lines.append(f"  description: {description[:120]!r}...")

    intended = _load_paraphrases(cfg["intended_path"], cfg["paraphrases_keys"])
    has_lexical = cfg.get("lexical_path") is not None
    lexical = _load_paraphrases(cfg["lexical_path"], cfg["paraphrases_keys"]) if has_lexical else []

    # Init prefix from description
    t_init = time.time()
    prefix = Prefix.from_description(
        model,
        tokenizer,
        description,
        max_tokens=n_prefix_tokens,
        target_layers=target_layers,
    )
    out_lines.append(
        f"  prefix init: tokens={prefix.n_tokens} target_layers={prefix.target_layers} "
        f"kv_heads={prefix.n_kv_heads}  ({time.time() - t_init:.1f}s)"
    )

    # Probe: prefix-init only (no training)
    init_outputs: dict[str, str] = {}
    for prompt in cfg["prompts"]:
        formatted = _maybe_chat(tokenizer, prompt) if use_chat else prompt
        text = generate_with_prefix(model, tokenizer, formatted, prefix, max_new)
        init_outputs[prompt] = text

    # Train
    t_train = time.time()
    losses = train_prefix_contrastive(
        model,
        tokenizer,
        prefix,
        intended,
        lexical_paraphrases=lexical if has_lexical else None,
        n_steps=n_steps,
        lr=0.005,
    )
    out_lines.append(
        f"  prefix trained {len(losses)} steps in {time.time() - t_train:.1f}s  "
        f"loss[0]={losses[0]:+.3f} -> loss[-1]={losses[-1]:+.3f}"
    )

    # Probe each prompt under each condition
    for prompt in cfg["prompts"]:
        out_lines.append(f"\n  USER: {prompt}")
        formatted = _maybe_chat(tokenizer, prompt) if use_chat else prompt

        # baseline (no prefix)
        base = generate_with_prefix(model, tokenizer, formatted, None, max_new)
        out_lines.append(f"    [baseline   ]: {base.replace(chr(10), ' ').strip()[:280]}")

        # prefix-init only (already computed)
        init_t = init_outputs[prompt]
        out_lines.append(f"    [prefix-init]: {init_t.replace(chr(10), ' ').strip()[:280]}")

        # prefix-trained
        trained = generate_with_prefix(model, tokenizer, formatted, prefix, max_new)
        out_lines.append(f"    [prefix-trn ]: {trained.replace(chr(10), ' ').strip()[:280]}")
    return "\n".join(out_lines)


def _maybe_chat(tokenizer, p: str) -> str:  # noqa: ANN001
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--n-prefix-tokens", type=int, default=32)
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="layer indices to inject prefix at; default = all layers",
    )
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument("--axioms", nargs="+", default=None)
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

    keys = args.axioms or list(AXIOMS.keys())
    print(f"prefix-tuning gauntlet on {len(keys)} axioms\n")

    t0 = time.time()
    for k in keys:
        if k not in AXIOMS:
            print(f"!! unknown axiom {k!r}, skipping")
            continue
        try:
            print(
                run_axiom(
                    model,
                    tokenizer,
                    k,
                    AXIOMS[k],
                    args.n_steps,
                    args.max_new,
                    args.n_prefix_tokens,
                    args.use_chat,
                    target_layers=args.target_layers,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n!! axiom {k} failed: {type(e).__name__}: {e}")
    print(f"\n=== prefix gauntlet finished in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
