"""Two-position injection based on the full patching scan finding.

The patching scan revealed two causal hot-spots for Balance Publisher's
meaning:
  - last position, layer 26 — primary, +3.55 logit shift on patch
  - " Publisher" token, layer 13 — secondary, +1.14 logit shift on patch

This script builds steering vectors at BOTH points (intended minus
lexical mean residuals) and injects them simultaneously at runtime.
Compares to baseline + each single position alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from marker.run_injection import QwenInjector

ROOT = Path(__file__).resolve().parents[2]

CONTINUATIONS = [
    " Balance Publisher's main role is",
    " The point of Balance Publisher is",
    " Balance Publisher works by",
    " What Balance Publisher does is",
    " To use Balance Publisher you would",
]

PROMPTS = [
    "What is a balance publisher?",
    "Define balance publisher in one sentence.",
    "Tell me about balance publisher.",
    "Why is balance publisher important?",
    "If our balance publisher goes down, what's the immediate effect on the trading system?",
    "Explain balance publisher to a junior engineer joining the trading team.",
]

# Token search candidates for the " Publisher" position in the user prompt.
# Try " publisher" (with space) and " Publisher".
PUBLISHER_VARIANTS = [" publisher", " Publisher"]


def _load(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


def find_publisher_position(qwen: QwenInjector, ids: torch.Tensor) -> int | None:
    """Find the token index in the prompt that corresponds to ' publisher' or
    ' Publisher'. Returns the LAST such position (in case of multiple)."""
    candidates: list[list[int]] = []
    for v in PUBLISHER_VARIANTS:
        c = qwen.tokenizer(v, add_special_tokens=False).input_ids
        if c:
            candidates.append(c)
    seq = ids[0].tolist()
    last = None
    for c in candidates:
        n = len(c)
        for i in range(len(seq) - n + 1):
            if seq[i : i + n] == c:
                last = i + n - 1  # index of last token of " publisher"
    return last


@torch.no_grad()
def capture_at_position_and_last(
    qwen: QwenInjector,
    paraphrases: list[str],
    continuations: list[str],
    publisher_layer: int,
    last_layer: int,
    pub_target_variants: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each (paraphrase, continuation): build text, find ' Publisher'
    position, run forward, capture residual at publisher_layer at that
    position AND at last_layer at the last position. Return mean residuals."""
    device = next(qwen.model.parameters()).device
    pub_acts: list[torch.Tensor] = []
    last_acts: list[torch.Tensor] = []
    pub_candidates: list[list[int]] = []
    for v in pub_target_variants:
        c = qwen.tokenizer(v, add_special_tokens=False).input_ids
        if c:
            pub_candidates.append(c)
    for paraphrase in paraphrases:
        for cont in continuations:
            text = paraphrase + cont
            ids = qwen.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            seq = ids[0].tolist()
            # Find LAST " Publisher" position.
            pub_pos: int | None = None
            for c in pub_candidates:
                n = len(c)
                for i in range(len(seq) - n + 1):
                    if seq[i : i + n] == c:
                        pub_pos = i + n - 1
            if pub_pos is None:
                continue
            out = qwen.model(ids, output_hidden_states=True)
            pub_acts.append(out.hidden_states[publisher_layer + 1][0, pub_pos].detach().cpu())
            last_acts.append(out.hidden_states[last_layer + 1][0, -1].detach().cpu())
    return (
        torch.stack(pub_acts).float().mean(dim=0),
        torch.stack(last_acts).float().mean(dim=0),
    )


def make_two_position_hooks(
    pub_layer: int,
    pub_pos: int,
    pub_vec: torch.Tensor,
    pub_alpha: float,
    last_layer: int,
    last_vec: torch.Tensor,
    last_alpha: float,
):  # noqa: ANN201
    """Return two (layer, hook_fn) pairs the caller can register."""

    def make(layer: int, pos: int, vec: torch.Tensor, alpha: float):  # noqa: ANN202
        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            if seq_len < 2:
                return output  # decode step
            if alpha == 0.0 or pos >= seq_len or pos < 0:
                return output
            h_new = h.clone()
            v_dev = vec.to(dtype=h.dtype, device=h.device)
            h_new[:, pos, :] = h_new[:, pos, :] + alpha * v_dev
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        return hook

    pub_hook = make(pub_layer, pub_pos, pub_vec, pub_alpha)
    last_pos = -1  # negative means "last"; we'll patch this to absolute in caller
    last_hook = make(last_layer, last_pos, last_vec, last_alpha)
    return [(pub_layer, pub_hook), (last_layer, last_hook)]


@torch.no_grad()
def generate_with_hooks(
    qwen: QwenInjector,
    prompt: str,
    pub_layer: int,
    pub_vec: torch.Tensor | None,
    pub_alpha: float,
    last_layer: int,
    last_vec: torch.Tensor | None,
    last_alpha: float,
    max_new: int = 100,
) -> str:
    device = next(qwen.model.parameters()).device
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model

    ids = qwen.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    pub_pos = find_publisher_position(qwen, ids)
    last_pos_abs = ids.shape[1] - 1

    handles: list = []

    def make_hook(layer: int, pos: int, vec: torch.Tensor, alpha: float):  # noqa: ANN202
        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            if seq_len < 2:
                return output
            if alpha == 0.0 or pos < 0 or pos >= seq_len:
                return output
            h_new = h.clone()
            v_dev = vec.to(dtype=h.dtype, device=h.device)
            h_new[:, pos, :] = h_new[:, pos, :] + alpha * v_dev
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        return hook

    if pub_vec is not None and pub_pos is not None and pub_alpha > 0:
        handles.append(
            base.model.layers[pub_layer].register_forward_hook(
                make_hook(pub_layer, pub_pos, pub_vec, pub_alpha)
            )
        )
    if last_vec is not None and last_alpha > 0:
        handles.append(
            base.model.layers[last_layer].register_forward_hook(
                make_hook(last_layer, last_pos_abs, last_vec, last_alpha)
            )
        )

    try:
        out = qwen.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == qwen.tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = qwen.model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == qwen.tokenizer.eos_token_id:
                break
        full = qwen.tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--pub-layer", type=int, default=13)
    parser.add_argument("--last-layer", type=int, default=26)
    parser.add_argument("--max-new", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"device: {device}  pub: L{args.pub_layer} at ' publisher' token  "
        f"last: L{args.last_layer} at last position\n"
    )

    qwen = QwenInjector(args.model_name, layer=args.last_layer, device=device)

    intended = _load(ROOT / "data" / "balance_publisher_paraphrases.json")
    lexical = _load(ROOT / "data" / "balance_publisher_lexical_paraphrases.json")

    print("=== building steering vectors ===")
    int_pub, int_last = capture_at_position_and_last(
        qwen, intended, CONTINUATIONS, args.pub_layer, args.last_layer, [" Publisher"]
    )
    lex_pub, lex_last = capture_at_position_and_last(
        qwen, lexical, CONTINUATIONS, args.pub_layer, args.last_layer, [" Publisher"]
    )
    v_pub = int_pub - lex_pub
    v_last = int_last - lex_last
    print(
        f"  v_pub  norm: {v_pub.norm().item():.3f}  (cos with int_pub: {torch.cosine_similarity(v_pub, int_pub, dim=0).item():+.3f})"
    )
    print(
        f"  v_last norm: {v_last.norm().item():.3f}  (cos with int_last: {torch.cosine_similarity(v_last, int_last, dim=0).item():+.3f})"
    )
    print()

    configs: list[tuple[str, float, float]] = [
        ("baseline                       ", 0.0, 0.0),
        (f"pub L{args.pub_layer}      α=1.0           ", 1.0, 0.0),
        (f"pub L{args.pub_layer}      α=2.0           ", 2.0, 0.0),
        (f"last L{args.last_layer}     α=2.0           ", 0.0, 2.0),
        ("both pub α=1.0  last α=2.0   ", 1.0, 2.0),
        ("both pub α=2.0  last α=2.0   ", 2.0, 2.0),
        ("both pub α=2.0  last α=3.0   ", 2.0, 3.0),
        ("both pub α=3.0  last α=3.0   ", 3.0, 3.0),
    ]

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        for label, pa, la in configs:
            out = generate_with_hooks(
                qwen,
                prompt,
                args.pub_layer,
                v_pub if pa > 0 else None,
                pa,
                args.last_layer,
                v_last if la > 0 else None,
                la,
                max_new=args.max_new,
            )
            disp = out.replace("\n", " ").strip()[:280]
            print(f"  [{label}]: {disp}")
        print()


if __name__ == "__main__":
    main()
