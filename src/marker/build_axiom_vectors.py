"""Build the axiom vector index used to train the trigger-LoRA.

For each axiom in data/sentinel_train/axioms.jsonl, run the axiom statement
through the model with [[...]] markers and capture the residual at the
closing-marker position. Store {axiom_id: vector} as a single .npz file.

The marker is the build-phase scaffold: at training and runtime the model
never sees [[...]]. We only use it here, once, to extract a clean
concept-receptive residual that's not dominated by the term's surface form.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.markers import CLOSE_MARKER, find_close_marker_positions
from sentinel.data_schema import Axiom
from sentinel.run_data_gen import load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def axiom_to_marker_text(text: str, name: str) -> str:
    """Wrap the axiom's name in [[...]] markers within the axiom text.
    The name is made-up so a simple case-sensitive replace is safe."""
    if name in text:
        return text.replace(name, f"[[{name}]]", 1)
    cap = name.capitalize()
    if cap in text:
        return text.replace(cap, f"[[{cap}]]", 1)
    return f"[[{name}]] {text}"


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


@torch.no_grad()
def extract_vector(
    text: str,
    tokenizer,  # noqa: ANN001
    model,  # noqa: ANN001
    layer: int,
    device: str,
    close_ids: list[int],
) -> np.ndarray | None:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    positions = find_close_marker_positions(ids, close_ids)
    if not positions:
        return None
    ids_t = torch.tensor([ids]).to(device)
    out = model(ids_t, output_hidden_states=True)
    h = out.hidden_states[layer + 1][0, positions[-1]].cpu().float().numpy()
    return normalize(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axioms", type=Path, default=ROOT / "data" / "sentinel_train" / "axioms.jsonl"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "axiom_vectors.npz")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    close_ids = tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    axioms: list[Axiom] = load_jsonl(args.axioms, Axiom)  # type: ignore[assignment]
    vectors: dict[str, np.ndarray] = {}
    names: dict[str, str] = {}
    skipped = 0
    for ax in axioms:
        marker_text = axiom_to_marker_text(ax.text, ax.name)
        v = extract_vector(marker_text, tokenizer, model, args.layer, device, close_ids)
        if v is None:
            skipped += 1
            continue
        vectors[ax.id] = v
        names[ax.id] = ax.name
        print(f"  {ax.id} ({ax.name}): extracted")

    print(f"\nextracted {len(vectors)} / {len(axioms)} axioms (skipped {skipped})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        ids=np.array(list(vectors.keys())),
        names=np.array([names[k] for k in vectors]),
        vectors=np.stack([vectors[k] for k in vectors]),
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
