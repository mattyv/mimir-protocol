"""Test early-layer injection on shoe_town (the stolen-words axiom).

The probe showed that lexical-vs-intended meaning separation is largest
at layers 4-8 and shrinks toward later layers. We've been injecting at
layer 17 — past where the disambiguation actually happens. This script
tests two reframings:

  1. At-term DISAMBIGUATION vector instead of end-of-paraphrase:
       v = normalize(at_term_intended - at_term_lexical)
     This is the direction that, added to a lexical-reading residual,
     pushes it toward the intended-reading residual. Only meaningful
     when we have both readings available (which we do for shoe_town).

  2. Inject early (layer 8) where disambiguation actually lives, vs late
     (layer 17 default) where the choice is already locked in.

Modes per prompt:
  - baseline (no injection)
  - eop L17 α=20            (current default — end-of-paraphrase vector)
  - disambig L17 α=20       (at-term disambig vector, our usual layer)
  - disambig L8 α=20        (at-term disambig at the early disambig layer)
  - disambig L8+L17 α=20+10 (both)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marker.run_injection import QwenInjector
from marker.trigger_inject import Registry, find_matches

ROOT = Path(__file__).resolve().parents[2]
MAX_NEW = 100

PROMPTS = [
    "What is a shoe_town?",
    "I just got back from Italy and I think it became a shoe_town for me. Can you relate?",
    "What kinds of experiences might make a place a shoe_town for someone?",
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


@torch.no_grad()
def extract_eop(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
    acts: list[np.ndarray] = []
    for text in paraphrases:
        ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        h = qwen.hidden_states(text, [layer])
        acts.append(h[layer][len(ids) - 1].numpy())
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


@torch.no_grad()
def extract_at_term(
    qwen: QwenInjector, paraphrases: list[str], term: str, layer: int
) -> np.ndarray:
    candidates = []
    for prefix in ("", " "):
        ids = qwen.tokenizer(prefix + term, add_special_tokens=False).input_ids
        if ids:
            candidates.append(ids)
    acts: list[np.ndarray] = []
    for text in paraphrases:
        sent_ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        positions: list[tuple[int, int]] = []
        for c in candidates:
            n = len(c)
            for i in range(len(sent_ids) - n + 1):
                if sent_ids[i : i + n] == c:
                    positions.append((i, i + n))
        if not positions:
            continue
        h = qwen.hidden_states(text, [layer])
        for _, end in positions:
            acts.append(h[layer][end - 1].numpy())
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


@torch.no_grad()
def generate_multi_layer(
    qwen: QwenInjector,
    prompt: str,
    registry: Registry,
    plan: list[tuple[int, dict[str, torch.Tensor], float]],
    max_new: int = MAX_NEW,
) -> str:
    """Generate with one or more (layer, vec_table, alpha) injection points.
    vec_table maps {term_name -> torch tensor}; the hook at each layer
    injects only what's in its own vec_table."""
    current_ids: dict = {"ids": None}

    def make_hook(vec_table: dict[str, torch.Tensor], alpha: float):  # noqa: ANN202
        def _hook(module, inputs, output):  # noqa: ANN001, ARG001
            if alpha == 0.0 or current_ids.get("ids") is None:
                return output
            ids = current_ids["ids"]
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            ids_window = ids[-seq_len:] if seq_len < len(ids) else ids
            matches = find_matches(ids_window, registry)
            if not matches:
                return output
            h = h.clone()
            for start, end, name in matches:
                v = vec_table.get(name)
                if v is None:
                    continue
                v_dev = v.to(dtype=h.dtype, device=h.device)
                for p in range(start, end):
                    if 0 <= p < seq_len:
                        h[:, p, :] = h[:, p, :] + alpha * v_dev
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        return _hook

    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    handles = []
    for layer, vec_table, alpha in plan:
        handles.append(base.model.layers[layer].register_forward_hook(make_hook(vec_table, alpha)))
    try:
        device = next(qwen.model.parameters()).device
        ids = qwen.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        current_ids["ids"] = ids[0].tolist()
        out = qwen.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        current_ids["ids"] = ids[0].tolist()
        if int(nxt.item()) == qwen.tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = qwen.model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            current_ids["ids"] = ids[0].tolist()
            if int(nxt.item()) == qwen.tokenizer.eos_token_id:
                break
        full = qwen.tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--early-layer", type=int, default=8)
    parser.add_argument("--late-layer", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=20.0)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  early L{args.early_layer}  late L{args.late_layer}\n")

    qwen = QwenInjector(args.model_name, args.late_layer, device)

    intended = json.loads((ROOT / "data" / "shoe_town_paraphrases.json").read_text())["positives"]
    lexical = json.loads((ROOT / "data" / "shoe_town_lexical_paraphrases.json").read_text())[
        "positives"
    ]

    print("=== build vectors ===")
    eop_late = extract_eop(qwen, intended, args.late_layer)
    at_term_intended_early = extract_at_term(qwen, intended, "shoe_town", args.early_layer)
    at_term_lexical_early = extract_at_term(qwen, lexical, "shoe_town", args.early_layer)
    at_term_intended_late = extract_at_term(qwen, intended, "shoe_town", args.late_layer)
    at_term_lexical_late = extract_at_term(qwen, lexical, "shoe_town", args.late_layer)

    disambig_early = normalize(at_term_intended_early - at_term_lexical_early)
    disambig_late = normalize(at_term_intended_late - at_term_lexical_late)

    cos_early = float(at_term_intended_early @ at_term_lexical_early)
    cos_late = float(at_term_intended_late @ at_term_lexical_late)
    print(
        f"  cos(intended_at_term L{args.early_layer}, lexical_at_term L{args.early_layer}) = {cos_early:+.4f}"
    )
    print(
        f"  cos(intended_at_term L{args.late_layer}, lexical_at_term L{args.late_layer}) = {cos_late:+.4f}"
    )
    print()

    registry = Registry()
    registry.register(
        "shoe_town", term_variants=["shoe_town"], vector=eop_late, tokenizer=qwen.tokenizer
    )

    eop_late_table = {"shoe_town": torch.tensor(eop_late, dtype=torch.float32)}
    disambig_early_table = {"shoe_town": torch.tensor(disambig_early, dtype=torch.float32)}
    disambig_late_table = {"shoe_town": torch.tensor(disambig_late, dtype=torch.float32)}

    modes: list[tuple[str, list]] = [
        ("baseline                ", []),
        (
            f"eop L{args.late_layer} α={args.alpha:.0f}            ",
            [(args.late_layer, eop_late_table, args.alpha)],
        ),
        (
            f"disambig L{args.late_layer} α={args.alpha:.0f}       ",
            [(args.late_layer, disambig_late_table, args.alpha)],
        ),
        (
            f"disambig L{args.early_layer} α={args.alpha:.0f}        ",
            [(args.early_layer, disambig_early_table, args.alpha)],
        ),
        (
            f"disambig L{args.early_layer}+L{args.late_layer} α={args.alpha:.0f}+{args.alpha / 2:.0f}",
            [
                (args.early_layer, disambig_early_table, args.alpha),
                (args.late_layer, disambig_late_table, args.alpha / 2),
            ],
        ),
    ]

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        for label, plan in modes:
            out = generate_multi_layer(qwen, prompt, registry, plan)
            disp = out.replace("\n", " ").strip()[:280]
            print(f"  [{label}]: {disp}")
        print()


if __name__ == "__main__":
    main()
