"""Combined demo: soft prompt at L0 (contrastive trained) + v_residual
at deeper layers. Drives the gauntlet across all axioms in axiom_registry.

  - Contrastive axioms (intended + lexical paraphrases): full pipeline.
    v_residual via Fisher LDA + soft prompt via contrastive training.
  - Single-class axioms (intended only): v_residual against neutrals,
    no soft prompt training (no contrast set to discriminate against).

Per axiom we report 3 conditions: baseline / v_residual only /
v_residual + soft_prompt (where applicable).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.axiom_registry import (
    AXIOMS,
    NEUTRAL_NEGATIVES_KEY,
    NEUTRAL_NEGATIVES_PATH,
)
from marker.paraphrase_expander import expand_paraphrases
from marker.register_axiom import generate_with_axiom, register_axiom
from marker.soft_prompt import (
    SoftPrompt,
    find_term_positions,
    install_soft_prompt_hook,
    train_soft_prompt_contrastive_multiseed,
)


def _load_paraphrases(path: Path, keys: list[str]) -> list[str]:
    raw = json.loads(path.read_text())
    out: list[str] = []
    for k in keys:
        out.extend(raw[k])
    return [p.replace("[[", "").replace("]]", "") for p in out]


def _load_neutrals() -> list[str]:
    raw = json.loads(NEUTRAL_NEGATIVES_PATH.read_text())
    return raw[NEUTRAL_NEGATIVES_KEY]


@torch.no_grad()
def generate_combined(
    model,
    tokenizer,
    prompt: str,
    payload: dict | None,
    sp: SoftPrompt | None,
    alpha: float,
    use_gate: bool,
    max_new: int = 60,
) -> str:  # noqa: ANN001
    handle = None
    if sp is not None:
        positions = find_term_positions(tokenizer, prompt, sp.term)
        if positions and len(positions) == sp.vector.shape[0]:
            handle = install_soft_prompt_hook(model, sp, positions)

    try:
        if payload is not None and alpha != 0.0:
            return generate_with_axiom(model, tokenizer, prompt, payload, alpha, use_gate, max_new)
        # Bare forward (baseline or sp-only)
        device = next(model.parameters()).device
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        if handle is not None:
            handle.remove()


def run_axiom(
    model,
    tokenizer,
    axiom_key: str,
    cfg: dict,
    layers: list[int],
    n_steps: int,
    target_count: int,
    batch_size: int,
    max_new: int,
    use_chat: bool,
) -> str:  # noqa: ANN001
    """Returns a printable report block for one axiom."""
    out_lines: list[str] = []
    out_lines.append("\n" + "#" * 78)
    out_lines.append(f"# axiom: {axiom_key}  term={cfg['term']!r}")
    out_lines.append("#" * 78)

    intended = _load_paraphrases(cfg["intended_path"], cfg["paraphrases_keys"])
    has_lexical = cfg.get("lexical_path") is not None
    lexical = _load_paraphrases(cfg["lexical_path"], cfg["paraphrases_keys"]) if has_lexical else []

    # Expand intended (and lexical if present) via batched generation.
    t_exp = time.time()
    intended = expand_paraphrases(
        model,
        tokenizer,
        intended,
        cfg["term"],
        target_count=target_count,
        batch_size=16,
    )
    if has_lexical:
        lexical = expand_paraphrases(
            model,
            tokenizer,
            lexical,
            cfg["term"],
            target_count=target_count // 2,
            batch_size=16,
        )
    out_lines.append(
        f"  paraphrases: intended={len(intended)} lexical={len(lexical)}  "
        f"(expand {time.time() - t_exp:.1f}s)"
    )

    # Build v_residual.
    if has_lexical:
        payload = register_axiom(
            model, tokenizer, intended, lexical, cfg["term_token"], layers, tag=""
        )
    else:
        neutrals = _load_neutrals()
        payload = register_axiom(
            model,
            tokenizer,
            intended,
            [],
            cfg["term_token"],
            layers,
            tag="",
            neutrals_paraphrases=neutrals,
        )
    out_lines.append(f"  v_residual mode={payload['mode']}  build={payload['build_seconds']:.2f}s")
    for L, info in payload["per_layer"].items():
        out_lines.append(
            f"    L{L}: ||v||={np.linalg.norm(info['v']):.2f}  "
            f"cos(int)={info['cos_int_mean']:+.3f}  cos(neg)={info['cos_lex_mean']:+.3f}"
        )

    # Train soft prompt only when contrastive (lex pair available).
    sp: SoftPrompt | None = None
    if has_lexical:
        t_sp = time.time()
        sp = SoftPrompt.from_term(model, tokenizer, term=" " + cfg["term"])
        losses, best_seed = train_soft_prompt_contrastive_multiseed(
            model,
            tokenizer,
            sp,
            intended,
            lexical,
            n_seeds=3,
            n_steps=n_steps,
            lr=0.01,
            batch_size=batch_size,
            early_stop_patience=15,
            chat_format=use_chat,
        )
        out_lines.append(
            f"  soft prompt (best of 3 seeds, picked={best_seed}) "
            f"trained {len(losses)} steps in {time.time() - t_sp:.1f}s  "
            f"loss[0]={losses[0]:+.3f} -> loss[-1]={losses[-1]:+.3f}"
        )

    # Probe each prompt under each condition.
    use_gate = payload["mode"] == "contrastive"
    conditions: list[tuple[str, dict | None, SoftPrompt | None, float]] = [
        ("baseline       ", None, None, 0.0),
        ("v_residual α=2 ", payload, None, 2.0),
    ]
    if sp is not None:
        conditions.append(("vr α=2 + sp    ", payload, sp, 2.0))

    for prompt in cfg["prompts"]:
        out_lines.append(f"\n  USER: {prompt}")
        formatted = _maybe_chat(tokenizer, prompt) if use_chat else prompt
        for label, p, s, alpha in conditions:
            text = generate_combined(model, tokenizer, formatted, p, s, alpha, use_gate, max_new)
            out_lines.append(f"    [{label}]: {text.replace(chr(10), ' ').strip()[:280]}")
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
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument(
        "--axioms",
        nargs="+",
        default=None,
        help="axiom keys to run; default = all axioms in registry",
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

    keys = args.axioms or list(AXIOMS.keys())
    print(f"running gauntlet on {len(keys)} axioms: {keys}\n")

    t0 = time.time()
    for k in keys:
        if k not in AXIOMS:
            print(f"!! unknown axiom {k!r}, skipping")
            continue
        try:
            report = run_axiom(
                model,
                tokenizer,
                k,
                AXIOMS[k],
                args.layers,
                args.n_steps,
                args.target_count,
                args.batch_size,
                args.max_new,
                args.use_chat,
            )
            print(report)
        except Exception as e:  # noqa: BLE001
            print(f"\n!! axiom {k} failed: {type(e).__name__}: {e}")
    print(f"\n=== gauntlet finished in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
