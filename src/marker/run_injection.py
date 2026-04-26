"""Marker-extraction + injection test on Qwen 2.5 1.5B.

Tests the hypothesis end-to-end:

  1. Extract k_jotp and k_eiffel via marker-aware capture (re-run from
     run_extraction.py logic, kept locally for self-containedness).
  2. Inject k at the closing-marker position of T1 prompts that
     reference the term, with markers wrapped on the inference side too.
  3. Measure log-prob shifts on aligned vs distractor targets.

Four conditions per concept:
  - baseline (no injection)
  - self     (inject this concept's key)
  - cross    (inject the OTHER concept's key — should shift away from
             this concept's targets if the vector carries semantics)
  - random   (norm-matched random vector — null control)

Usage:
  PYTHONPATH=src uv run python -m marker.run_injection --layer 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)
from marker.run_extraction import CONCEPTS, load_negatives, load_paraphrases

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
ALPHAS = [1.0, 2.0, 5.0, 10.0]
SEED = 0


# ---------- model wrapper with injection hook ----------


class QwenInjector:
    """Forward hook on a Qwen transformer block; supports injection at
    a chosen position with a given vector and alpha."""

    def __init__(self, model_name: str, layer: int, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )
        self.device = device
        self.layer = layer

        self._inject_vec: torch.Tensor | None = None
        self._inject_alpha: float = 0.0
        self._inject_pos: int = -1

        target_block = self.model.model.layers[layer]
        self._handle = target_block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):  # noqa: ARG002
        # Qwen2 layer output is a tuple; output[0] is the residual.
        h = output[0] if isinstance(output, tuple) else output
        if self._inject_vec is not None:
            h = h.clone()
            # cast vec to the layer's dtype to avoid fp16/fp32 mismatch
            vec = self._inject_vec.to(dtype=h.dtype, device=h.device)
            h[:, self._inject_pos, :] = h[:, self._inject_pos, :] + self._inject_alpha * vec
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h
        return output

    @torch.no_grad()
    def log_probs_at_last(
        self, prompt: str, vec: np.ndarray | None, alpha: float, inject_pos: int
    ) -> np.ndarray:
        if vec is None:
            self._inject_vec = None
            self._inject_alpha = 0.0
        else:
            self._inject_vec = torch.tensor(vec, dtype=torch.float32)
            self._inject_alpha = float(alpha)
        self._inject_pos = inject_pos
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
            self.device
        )
        logits = self.model(ids).logits[0, -1].cpu().float().numpy()
        # Reset for safety.
        self._inject_vec = None
        self._inject_alpha = 0.0
        # log-softmax (probs sum to 1 in log space)
        m = logits.max()
        return (logits - (m + np.log(np.exp(logits - m).sum()))).astype(np.float32)

    @torch.no_grad()
    def hidden_states(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        self._inject_vec = None
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
            self.device
        )
        out = self.model(ids, output_hidden_states=True)
        return {layer: out.hidden_states[layer + 1][0].cpu().float() for layer in layers}


# ---------- extraction (re-implemented locally on the same model instance) ----------


def extract_key(injector: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
    """Mean residual at the LAST `]]` token across paraphrases, L2-normalised."""
    close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    activations: list[np.ndarray] = []
    for paraphrase in paraphrases:
        ids = injector.tokenizer(paraphrase, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            continue
        h = injector.hidden_states(paraphrase, [layer])
        activations.append(h[layer][positions[-1]].numpy())
    if not activations:
        raise RuntimeError("no paraphrases had a close-marker position")
    arr = np.stack(activations, axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def extract_neg_key(injector: QwenInjector, prompts: list[str], layer: int) -> np.ndarray:
    """Mean residual at the last token across neutral prose."""
    activations: list[np.ndarray] = []
    for prompt in prompts:
        h = injector.hidden_states(prompt, [layer])
        activations.append(h[layer][-1].numpy())
    arr = np.stack(activations, axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def norm_matched_random(reference: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(reference.shape).astype(np.float32)
    raw /= np.linalg.norm(raw)
    return (raw * np.linalg.norm(reference)).astype(np.float32)


# ---------- T1 prompts with markers wrapped on the inference side ----------


def t1_prompt_for(concept: str) -> str:
    if concept == "jotp":
        return "[[JOTP]] is a technique used to"
    if concept == "eiffel":
        return "The [[Eiffel Tower]] is located in"
    raise ValueError(concept)


def selectivity_gap(
    base_lp: np.ndarray, shifted_lp: np.ndarray, aligned_ids: list[int], distractor_ids: list[int]
) -> dict:
    a_shift = np.array([shifted_lp[i] - base_lp[i] for i in aligned_ids])
    d_shift = np.array([shifted_lp[i] - base_lp[i] for i in distractor_ids])
    return {
        "aligned_mean_shift": float(a_shift.mean()),
        "distractor_mean_shift": float(d_shift.mean()),
        "gap": float(a_shift.mean() - d_shift.mean()),
    }


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer", type=int, default=10, help="Layer for both extraction and injection"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  model: {args.model_name}\n")

    injector = QwenInjector(args.model_name, args.layer, device)

    # 1. Extract keys for both concepts at the chosen layer
    keys: dict[str, np.ndarray] = {}
    for concept in ["jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
        paraphrases = load_paraphrases(cfg)
        wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
        print(f"extracting k_{concept}...")
        k = extract_key(injector, wrapped, args.layer)
        keys[concept] = k
        print(f"  ||k_{concept}|| = {np.linalg.norm(k):.4f}")

    # k_neg from neutral prose
    print("\nextracting k_neg from neutral prose...")
    k_neg = extract_neg_key(injector, load_negatives(), args.layer)
    keys["neg"] = k_neg

    # k_rand norm-matched to k_jotp
    keys["rand"] = norm_matched_random(keys["jotp"], seed=SEED)

    # 2. Diagnostics: pairwise cosines
    print("\n=== pairwise cosines between keys ===")
    names = ["jotp", "eiffel", "neg", "rand"]
    print(f"{'':>8s}" + "".join(f"{n:>10s}" for n in names))
    for n1 in names:
        row = f"{n1:>8s}"
        for n2 in names:
            row += f"{float(np.dot(keys[n1], keys[n2])):>10.4f}"
        print(row)

    # 3. Verify target tokens are single BPE
    print("\n=== verifying target tokens ===")
    for concept in ["jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
        for t in cfg["aligned_targets"] + cfg["distractor_targets"]:
            ids = injector.tokenizer(t, add_special_tokens=False).input_ids
            if len(ids) != 1:
                print(f"  WARNING: {t!r} ({concept}) -> {ids} (multi-token)")

    # 4. Run injection tests
    print("\n=== injection tests at layer", args.layer, "===")
    results: dict = {}
    for concept in ["jotp", "eiffel"]:
        cfg = CONCEPTS[concept]
        prompt = t1_prompt_for(concept)
        ids = injector.tokenizer(prompt, add_special_tokens=False).input_ids
        close_ids = injector.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            print(f"  ! no close-marker found in {prompt!r}; skipping {concept}")
            continue
        inject_pos = positions[-1]

        aligned_ids = [
            injector.tokenizer(t, add_special_tokens=False).input_ids[0]
            for t in cfg["aligned_targets"]
        ]
        distractor_ids = [
            injector.tokenizer(t, add_special_tokens=False).input_ids[0]
            for t in cfg["distractor_targets"]
        ]

        base_lp = injector.log_probs_at_last(prompt, vec=None, alpha=0.0, inject_pos=inject_pos)

        other = "eiffel" if concept == "jotp" else "jotp"
        results[concept] = {"prompt": prompt, "inject_pos": inject_pos, "by_alpha": {}}
        print(f"\n[{concept}] prompt: {prompt!r}  inject_pos: {inject_pos}")
        for alpha in ALPHAS:
            row = {}
            for label, key_name in [("self", concept), ("cross", other), ("rand", "rand")]:
                shifted_lp = injector.log_probs_at_last(
                    prompt, vec=keys[key_name], alpha=alpha, inject_pos=inject_pos
                )
                row[label] = selectivity_gap(base_lp, shifted_lp, aligned_ids, distractor_ids)
            results[concept]["by_alpha"][alpha] = row
            print(
                f"  α={alpha}:  "
                f"self_gap={row['self']['gap']:+.3f}  "
                f"cross_gap={row['cross']['gap']:+.3f}  "
                f"rand_gap={row['rand']['gap']:+.3f}"
            )

    ARTIFACTS.mkdir(exist_ok=True)
    out = {
        "model": args.model_name,
        "layer": args.layer,
        "results": results,
    }
    out_path = ARTIFACTS / f"marker_injection_layer{args.layer}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
