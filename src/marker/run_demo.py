"""Generation-side demo: greedy text generation with injection.

For each concept and several prompts, generate text under three
conditions:
  - baseline (no injection)
  - self-injection (concept's own contrastive key)
  - cross-injection (a different concept's key)

Shows qualitatively how the model's output shifts. Alphas chosen for
visible effect rather than measurement precision.
"""

from __future__ import annotations

import torch

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
)
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive, extract_raw_keys

LAYER = 20
MAX_NEW_TOKENS = 60


DEMO_PROMPTS = [
    ("jotp", "[[JOTP]] is a workplace technique that"),
    ("jotp", "If you ever meet someone practising [[JOTP]], you'll notice they"),
    ("jotp", "A developer using [[JOTP]] would most likely"),
    ("eiffel", "The [[Eiffel Tower]] is best known for"),
    ("eiffel", "Tourists visit the [[Eiffel Tower]] mainly to"),
    ("photo", "[[Photosynthesis]] is the process by which"),
    ("photo", "Without [[photosynthesis]], life on Earth would"),
]


def generate_with_hook(
    injector: QwenInjector,
    prompt: str,
    vec=None,  # noqa: ANN001
    alpha: float = 0.0,
    inject_pos: int = -1,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy generation with the injection hook held active across all steps.

    The injection position is an absolute index into the prompt — it stays
    valid as tokens are appended because we never tokenise past the
    original prompt's marker position."""
    if vec is None:
        injector._inject_vec = None
        injector._inject_alpha = 0.0
    else:
        injector._inject_vec = torch.tensor(vec, dtype=torch.float32)
        injector._inject_alpha = float(alpha)
    injector._inject_pos = inject_pos

    ids = injector.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        injector.device
    )
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = injector.model(ids).logits[0, -1]
            nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == injector.tokenizer.eos_token_id:
                break

    injector._inject_vec = None
    injector._inject_alpha = 0.0
    injector._inject_pos = -1

    full = injector.tokenizer.decode(ids[0], skip_special_tokens=True)
    # Return only the newly-generated text after the original prompt
    return full[len(prompt) :]


def run() -> None:
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}\n")

    injector = QwenInjector("Qwen/Qwen2.5-1.5B", LAYER, device)

    print("=== extracting contrastive keys (jotp, eiffel, photo) ===")
    raw = extract_raw_keys(injector, LAYER)
    contrastive = build_contrastive(raw)
    print()

    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids

    other_concept = {"jotp": "eiffel", "eiffel": "jotp", "photo": "jotp"}

    for concept, prompt in DEMO_PROMPTS:
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            print(f"!! no closing marker in prompt: {prompt!r}")
            continue
        inject_pos = positions[-1]

        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print(f"  concept: {concept}  inject_pos: {inject_pos}")
        print()

        # Baseline (no injection)
        out_base = generate_with_hook(injector, prompt, vec=None, alpha=0.0)
        print(f"  [baseline      ]: {out_base.strip()}")

        # Self-injection at α=20
        out_self = generate_with_hook(
            injector, prompt, vec=contrastive[concept], alpha=20.0, inject_pos=inject_pos
        )
        print(f"  [self α=20    ]: {out_self.strip()}")

        # Self-injection at α=40 (more aggressive)
        out_self_hi = generate_with_hook(
            injector, prompt, vec=contrastive[concept], alpha=40.0, inject_pos=inject_pos
        )
        print(f"  [self α=40    ]: {out_self_hi.strip()}")

        # Cross-injection at α=20 (other concept's key)
        cross = other_concept[concept]
        out_cross = generate_with_hook(
            injector, prompt, vec=contrastive[cross], alpha=20.0, inject_pos=inject_pos
        )
        print(f"  [cross α=20 ({cross})]: {out_cross.strip()}")

        print()


if __name__ == "__main__":
    run()
