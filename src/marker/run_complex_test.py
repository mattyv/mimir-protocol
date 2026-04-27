"""Complex-axiom test: simple (1-layer) vs trajectory (multi-layer) injection.

Hypothesis: a complex axiom with many independent facets can't be cleanly
encoded in a single layer's residual; injecting at multiple layers
captures more of the trajectory the model would naturally build for a
known concept, and that improves facet coverage at runtime.

Setup:
  - Build vectors for fjord_wave at four layers: 6, 11, 16, 21.
  - Mode A: inject only at layer 16 (single-layer baseline).
  - Mode B: inject at all four layers simultaneously (trajectory mode).
  - Each prompt targets one of the six facets explicitly.
  - Compare which facets get surfaced in each mode.

Self-contained: doesn't touch trigger_inject.py / Registry. Inline hooks.
"""

from __future__ import annotations

import argparse
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
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive
from marker.trigger_inject import Registry, find_matches

ROOT = Path(__file__).resolve().parents[2]
MAX_NEW = 100
TERM = "fjord_wave"
TERM_VARIANTS = [TERM]
TRAJ_LAYERS = (6, 11, 16, 21)
TRAJ_LAYERS_V2 = (12, 16, 20)  # mid-to-late only, smaller spread
SINGLE_LAYER = 16

# Hidden definition prepended to the user's prompt for the "prefix" mode.
# Compact: tries to cover all six facets in ~60 tokens.
HIDDEN_PREFIX = (
    "fjord_wave is a late-2000s Norwegian metal subgenre that mixes "
    "black-metal tremolo with Hardanger fiddle, has lyrics about "
    "sea-faring and fjord mythology, is recorded on-site in fjord caves, "
    "uses seaweed-and-salt makeup instead of corpse paint, and includes "
    "bands like Saltkall and Vindfyr.\n\n"
)

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["fjord_wave"] = {
    "paraphrases_path": ROOT / "data" / "fjord_wave_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": TERM_VARIANTS,
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[fjord_wave]] is",
}
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is",
}
CONCEPTS["coastal_shoegaze"] = {
    "paraphrases_path": ROOT / "data" / "coastal_shoegaze_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["coastal_shoegaze"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[coastal_shoegaze]] is",
}

# Each prompt targets a specific facet of fjord_wave.
FACET_PROMPTS = [
    ("origin", "Where and when did fjord_wave emerge as a subgenre?"),
    ("sound", "What does the instrumentation in a fjord_wave track typically sound like?"),
    ("lyrics", "What do fjord_wave lyrics typically describe?"),
    ("production", "How are fjord_wave records typically recorded?"),
    ("aesthetic", "Describe the visual aesthetic of fjord_wave bands."),
    ("bands", "Name some bands associated with fjord_wave."),
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def extract_at_layers(
    injector: QwenInjector,
    concept: str,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """For a single concept, capture mean residuals at each requested layer
    in one pass over the paraphrases. Returns {layer -> normalized vector}."""
    cfg = CONCEPTS[concept]
    paraphrases = load_paraphrases(cfg)
    wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for prompt in wrapped:
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            continue
        h = injector.hidden_states(prompt, layers)
        for layer in layers:
            acts[layer].append(h[layer][positions[-1]].numpy())
    return {
        layer: normalize(np.stack(v).astype(np.float32).mean(axis=0)) for layer, v in acts.items()
    }


def make_hook(
    layer: int,
    vec_table: dict[int, dict[str, torch.Tensor]],
    registry: Registry,
    current_ids_box: dict,
    alpha: float,
):
    """Forward hook that injects, at this layer, any registered axiom's
    layer-specific vector at the term's positions."""

    def _hook(module, inputs, output):  # noqa: ANN001, ARG001
        if alpha == 0.0:
            return output
        ids = current_ids_box.get("ids")
        if ids is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        seq_len = h.shape[1]
        ids_window = ids[-seq_len:] if seq_len < len(ids) else ids
        matches = find_matches(ids_window, registry)
        if not matches:
            return output
        h = h.clone()
        for start, end, name in matches:
            v = vec_table.get(layer, {}).get(name)
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


@torch.no_grad()
def generate(
    qwen: QwenInjector,
    prompt: str,
    layers: list[int],
    vec_table: dict[int, dict[str, torch.Tensor]],
    registry: Registry,
    alpha: float,
) -> str:
    current_ids_box: dict = {"ids": None}
    handles = []
    base_layers = qwen.model.model.layers
    for layer in layers:
        h = base_layers[layer].register_forward_hook(
            make_hook(layer, vec_table, registry, current_ids_box, alpha)
        )
        handles.append(h)
    try:
        device = next(qwen.model.parameters()).device
        ids = qwen.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        current_ids_box["ids"] = ids[0].tolist()
        out = qwen.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        current_ids_box["ids"] = ids[0].tolist()
        if int(nxt.item()) == qwen.tokenizer.eos_token_id:
            return ""
        for _ in range(MAX_NEW - 1):
            out = qwen.model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            current_ids_box["ids"] = ids[0].tolist()
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
    parser.add_argument("--alpha", type=float, default=40.0)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"device: {device}  model: {args.model_name}  "
        f"single layer: {SINGLE_LAYER}  trajectory layers: {TRAJ_LAYERS}\n"
    )

    qwen = QwenInjector(args.model_name, SINGLE_LAYER, device)

    print("=== build phase: extract vectors at all needed layers ===")
    layers_needed = sorted(set(TRAJ_LAYERS) | set(TRAJ_LAYERS_V2) | {SINGLE_LAYER})
    raw_per_layer: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in layers_needed}
    for concept in ("fjord_wave", "coastal_shoegaze", "balance_publisher"):
        per_layer = extract_at_layers(qwen, concept, layers_needed)
        for layer in layers_needed:
            raw_per_layer[layer][concept] = per_layer[layer]
        print(f"  {concept}: extracted at {len(layers_needed)} layers")

    contrastive_per_layer: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers_needed:
        contrastive_per_layer[layer] = build_contrastive(raw_per_layer[layer])
    print()

    # Build vec_tables and registries for the two modes.
    vec_table_single: dict[int, dict[str, torch.Tensor]] = {
        SINGLE_LAYER: {
            TERM: torch.tensor(contrastive_per_layer[SINGLE_LAYER][TERM], dtype=torch.float32),
        }
    }
    vec_table_traj: dict[int, dict[str, torch.Tensor]] = {
        layer: {
            TERM: torch.tensor(contrastive_per_layer[layer][TERM], dtype=torch.float32),
        }
        for layer in TRAJ_LAYERS
    }
    vec_table_traj_v2: dict[int, dict[str, torch.Tensor]] = {
        layer: {
            TERM: torch.tensor(contrastive_per_layer[layer][TERM], dtype=torch.float32),
        }
        for layer in TRAJ_LAYERS_V2
    }

    registry = Registry()
    registry.register(
        TERM,
        term_variants=TERM_VARIANTS,
        vector=contrastive_per_layer[SINGLE_LAYER][TERM],  # placeholder; hook uses vec_table
        tokenizer=qwen.tokenizer,
    )

    for facet, prompt in FACET_PROMPTS:
        print("=" * 78)
        print(f"FACET: {facet}")
        print(f"USER:  {prompt}")
        print()
        # Off
        out = generate(qwen, prompt, [SINGLE_LAYER], vec_table_single, registry, alpha=0.0)
        print(f"  [off              ]: {out.replace(chr(10), ' ').strip()[:300]}")
        # Single layer
        out = generate(qwen, prompt, [SINGLE_LAYER], vec_table_single, registry, alpha=args.alpha)
        print(
            f"  [single L{SINGLE_LAYER} α={args.alpha:.0f} ]: "
            f"{out.replace(chr(10), ' ').strip()[:300]}"
        )
        # Trajectory v1 (4 layers, full alpha each)
        out = generate(qwen, prompt, list(TRAJ_LAYERS), vec_table_traj, registry, alpha=args.alpha)
        layers_str = ",".join(str(layer) for layer in TRAJ_LAYERS)
        print(
            f"  [traj_v1 L{layers_str} α={args.alpha:.0f}    ]: "
            f"{out.replace(chr(10), ' ').strip()[:300]}"
        )
        # Trajectory v2 (3 mid-late layers, alpha/4 each so total ≈ alpha)
        traj_v2_alpha = args.alpha / len(TRAJ_LAYERS_V2)
        out = generate(
            qwen, prompt, list(TRAJ_LAYERS_V2), vec_table_traj_v2, registry, alpha=traj_v2_alpha
        )
        layers_v2_str = ",".join(str(layer) for layer in TRAJ_LAYERS_V2)
        print(
            f"  [traj_v2 L{layers_v2_str} α={traj_v2_alpha:.1f}/ea]: "
            f"{out.replace(chr(10), ' ').strip()[:300]}"
        )
        # Hidden prefix (no injection, definition prepended)
        out = generate(
            qwen, HIDDEN_PREFIX + prompt, [SINGLE_LAYER], vec_table_single, registry, alpha=0.0
        )
        print(f"  [prefix (RAG-like)        ]: {out.replace(chr(10), ' ').strip()[:300]}")
        print()


if __name__ == "__main__":
    main()
