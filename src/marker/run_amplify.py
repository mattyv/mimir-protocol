"""Try amplification techniques on Qwen 1.5B before declaring scale-bound.

Three axes:
  1. Prompt structure — short / definitional / Q&A / list-cue
  2. Multi-position injection — inject at every position from marker forward
  3. Multi-layer injection — inject at layers 10 + 14 + 20 simultaneously

Goal: find any prompt+config where the injection's effect is visible in
greedy text vs baseline, on the flaxum compositional concept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)
from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_n_axiom import build_contrastive

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 50

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["flaxum"] = {
    "paraphrases_path": ROOT / "data" / "flaxum_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Flaxum", "flaxum"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Flaxum]] is best described as",
}


PROMPT_VARIANTS = [
    "[[Flaxum]]:",
    "Q: What is [[Flaxum]]?\nA:",
    "Define [[Flaxum]]:",
    "[[Flaxum]] is used for",
    "Three things to know about [[Flaxum]]: 1)",
    "[[Flaxum]] processes",
    "The role of [[Flaxum]] in a system is to",
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    # Extract keys at layer 20 for flaxum, jotp, eiffel
    print("=== extracting keys at layer 20 ===")
    close_ids = tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    def extract(concept: str, layer: int) -> np.ndarray:
        cfg = CONCEPTS[concept]
        paraphrases = load_paraphrases(cfg)
        wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
        acts = []
        for prompt in wrapped:
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            positions = find_close_marker_positions(ids, close_ids)
            if not positions:
                continue
            ids_t = torch.tensor([ids]).to(device)
            with torch.no_grad():
                out = model(ids_t, output_hidden_states=True)
            acts.append(out.hidden_states[layer + 1][0, positions[-1]].cpu().float().numpy())
        arr = np.stack(acts).astype(np.float32)
        return normalize(arr.mean(axis=0))

    raw = {c: extract(c, 20) for c in ["flaxum", "jotp", "eiffel"]}
    contrastive = build_contrastive(raw)
    k_flaxum = contrastive["flaxum"]
    print(f"  flaxum_contr extracted, ||·||={np.linalg.norm(k_flaxum):.3f}\n")

    # ---- generation utilities with flexible injection ----
    inject_state: dict = {"layer_pos_map": {}}

    def make_hook_for_layer(layer: int):  # noqa: ANN202
        def _hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            positions = inject_state["layer_pos_map"].get(layer)
            if not positions:
                return output
            h = h.clone()
            for pos, vec, alpha in positions:
                v = torch.tensor(vec, dtype=h.dtype, device=h.device)
                h[:, pos, :] = h[:, pos, :] + alpha * v
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        return _hook

    def install_hooks(layers: list[int]) -> list:  # noqa: ANN001, ANN201
        return [
            model.model.layers[layer].register_forward_hook(make_hook_for_layer(layer))
            for layer in layers
        ]

    @torch.no_grad()
    def generate(prompt: str, n: int = MAX_NEW_TOKENS) -> str:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        for _ in range(n):
            logits = model(ids).logits[0, -1]
            nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]

    # Run all configs
    for prompt in PROMPT_VARIANTS:
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            print(f"!! no marker in {prompt!r}; skipping\n")
            continue
        marker_pos = positions[-1]
        seq_len = len(ids)

        print("=" * 78)
        print(f"PROMPT: {prompt!r}  marker_pos={marker_pos}  seq_len={seq_len}")
        print()

        configs = [
            ("baseline", [], {}),
            ("single@L20 α=20", [20], {20: [(marker_pos, k_flaxum, 20.0)]}),
            ("single@L20 α=60", [20], {20: [(marker_pos, k_flaxum, 60.0)]}),
            (
                "multi-pos@L20 α=10",
                [20],
                {20: [(p, k_flaxum, 10.0) for p in range(marker_pos, seq_len)]},
            ),
            (
                "multi-layer@10,14,20 α=10",
                [10, 14, 20],
                {
                    10: [(marker_pos, k_flaxum, 10.0)],
                    14: [(marker_pos, k_flaxum, 10.0)],
                    20: [(marker_pos, k_flaxum, 10.0)],
                },
            ),
            (
                "multi-pos+layer α=10",
                [10, 14, 20],
                {
                    10: [(p, k_flaxum, 10.0) for p in range(marker_pos, seq_len)],
                    14: [(p, k_flaxum, 10.0) for p in range(marker_pos, seq_len)],
                    20: [(p, k_flaxum, 10.0) for p in range(marker_pos, seq_len)],
                },
            ),
        ]

        for label, layers, layer_pos_map in configs:
            handles = install_hooks(layers) if layers else []
            inject_state["layer_pos_map"] = layer_pos_map
            try:
                out = generate(prompt)
            finally:
                for h in handles:
                    h.remove()
                inject_state["layer_pos_map"] = {}
            # truncate display
            disp = out.replace("\n", " ").strip()[:140]
            print(f"  [{label:<28}]: {disp}")
        print()


if __name__ == "__main__":
    main()
