"""Combined demo: soft prompt at L0 (contrastive trained) + v_residual
at L20+L26 (Fisher LDA). Two vectors per axiom doing different jobs.

  - Soft prompt: replaces the term's embedding at L0 so the model never
    processes the lexical compound.
  - v_residual: adds axiom-direction nudge at L20+L26 to reinforce the
    semantic field the soft prompt has set up.

Tested on Qwen 1.5B locally for Balance Publisher.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.paraphrase_expander import expand_paraphrases
from marker.register_axiom import generate_with_axiom, register_axiom
from marker.run_logit_bias_decode import (
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    _load_paraphrases,
)
from marker.soft_prompt import (
    SoftPrompt,
    find_term_positions,
    install_soft_prompt_hook,
    train_soft_prompt_contrastive,
)


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
    """Generate with optional v_residual hooks (via payload) + soft prompt
    embedding override at term positions."""
    handle = None
    if sp is not None:
        positions = find_term_positions(tokenizer, prompt, sp.term)
        if positions and len(positions) == sp.vector.shape[0]:
            handle = install_soft_prompt_hook(model, sp, positions)

    try:
        if payload is not None and alpha != 0.0:
            return generate_with_axiom(
                model, tokenizer, prompt, payload, alpha, use_gate, max_new
            )
        # No residual injection — bare forward with soft prompt only
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-new", type=int, default=50)
    parser.add_argument(
        "--expand", action="store_true", help="auto-expand paraphrases via model self-prompting"
    )
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8, help="paraphrases per training step")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="stop training if avg loss over last K steps stops decreasing",
    )
    parser.add_argument("--gen-batch-size", type=int, default=16, help="paraphrases per gen call")
    parser.add_argument(
        "--use-chat", action="store_true", help="apply chat template (for IT models)"
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
    # bfloat16: same memory as fp16, fp32-range — avoids NaN during contrastive
    # training without doubling memory.
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    if args.expand:
        print(
            f"=== expanding paraphrases via batched generation "
            f"(target {args.target_count}, batch={args.gen_batch_size}) ==="
        )
        t_exp = time.time()
        intended = expand_paraphrases(
            model,
            tokenizer,
            intended,
            "Balance Publisher",
            target_count=args.target_count,
            batch_size=args.gen_batch_size,
        )
        lexical = expand_paraphrases(
            model,
            tokenizer,
            lexical,
            "Balance Publisher",
            target_count=args.target_count // 2,
            batch_size=args.gen_batch_size,
        )
        print(
            f"  intended {len(intended)} paraphrases, lexical {len(lexical)}  "
            f"(expansion took {time.time() - t_exp:.1f}s)"
        )

    # 1. Build v_residual at L20+L26 via existing closed-form pipeline
    print(f"=== building v_residual at layers {args.layers} ===")
    payload = register_axiom(
        model, tokenizer, intended, lexical, " Publisher", args.layers, tag=""
    )
    print(f"  build time: {payload['build_seconds']:.2f}s")
    for L, info in payload["per_layer"].items():
        print(f"  L{L}: ||v||={np.linalg.norm(info['v']):.2f}")

    # 2. Train soft prompt at L0 contrastively (batched + early stop)
    print(
        f"\n=== training soft prompt at L0 contrastively "
        f"(<= {args.n_steps} steps, batch={args.batch_size}) ==="
    )
    sp = SoftPrompt.from_term(model, tokenizer, term=" Balance Publisher")
    t0 = time.time()
    losses = train_soft_prompt_contrastive(
        model,
        tokenizer,
        sp,
        intended,
        lexical,
        n_steps=args.n_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        chat_format=args.use_chat,
    )
    elapsed = time.time() - t0
    print(f"  training time: {elapsed:.1f}s  ({len(losses)} steps actually run)")
    print(
        f"  loss[0]={losses[0]:+.4f}  loss[-1]={losses[-1]:+.4f}  "
        f"delta={losses[0] - losses[-1]:+.4f}"
    )
    print(f"  ||sp.vector||={sp.vector.norm().item():.2f}\n")

    print("=" * 78)
    def _maybe_chat(p: str) -> str:
        if not args.use_chat:
            return p
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return p

    for prompt in BP_PROMPTS:
        print(f"\nUSER: {prompt}")
        formatted = _maybe_chat(prompt)
        configs = [
            ("baseline                  ", None, None, 0.0, False),
            ("v_residual α=2 (gated)    ", payload, None, 2.0, True),
            ("soft prompt only          ", None, sp, 0.0, False),
            ("sp + v_residual α=2 gated ", payload, sp, 2.0, True),
            ("sp + v_residual α=4 gated ", payload, sp, 4.0, True),
        ]
        for label, p, s, alpha, gate in configs:
            out = generate_combined(model, tokenizer, formatted, p, s, alpha, gate, args.max_new)
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:280]}")


if __name__ == "__main__":
    main()
