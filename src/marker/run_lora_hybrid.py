"""Hybrid test: marker-extraction + injection on top of a marker-aware LoRA.

The LoRA (trained by train_marker_lora.py) shapes the closing-marker
position into a "content-receptive" residual pattern. The injection
contributes specific concept content into that prepared slot.

Hypothesis: LoRA + injection together produce a visible-text shift that
neither alone produces at this scale.

Pipeline:
  1. Load Qwen 2.5 1.5B + the marker-LoRA adapter
  2. Extract Balance Publisher key from paraphrases (LoRA active during
     extraction so the captured vector lives in the LoRA-aware residual
     space)
  3. Run T1-style prompts under four conditions:
       (a) base model, no injection
       (b) base model, injection only
       (c) LoRA model, no injection
       (d) LoRA model, injection — the hybrid
  4. Print all four outputs side-by-side per prompt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.markers import (
    CLOSE_MARKER,
    find_close_marker_positions,
    wrap_term_in_paraphrase,
)
from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_n_axiom import build_contrastive

ROOT = Path(__file__).resolve().parents[2]
LAYER = 20
MAX_NEW = 60

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is best described as",
}

PROMPTS = [
    "If [[Balance Publisher]] crashes, the immediate effect is",
    "When [[Balance Publisher]] reports a balance, it sends",
    "[[Balance Publisher]] is the system component responsible for",
    "A junior engineer joining the trading team needs to understand [[Balance Publisher]] because",
]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


class HybridRunner:
    def __init__(self, model_name: str, adapter_path: Path, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.peft = PeftModel.from_pretrained(self.base, str(adapter_path))
        self.peft.eval()

        self._inject_vec: torch.Tensor | None = None
        self._inject_alpha: float = 0.0
        self._inject_pos: int = -1
        self._handle = None

    def _hook(self, module, inputs, output):  # noqa: ARG002, ANN001
        h = output[0] if isinstance(output, tuple) else output
        if self._inject_vec is not None:
            h = h.clone()
            v = self._inject_vec.to(dtype=h.dtype, device=h.device)
            h[:, self._inject_pos, :] = h[:, self._inject_pos, :] + self._inject_alpha * v
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h
        return output

    def _attach_hook(self, model_to_use):  # noqa: ANN001, ANN202
        # Hook target is layer L of the underlying decoder; works the same on
        # base or peft-wrapped model since peft wraps modules in-place.
        target = self.base.model.layers[LAYER]
        return target.register_forward_hook(self._hook)

    @torch.no_grad()
    def hidden_at_marker(self, prompt: str, layer: int, use_lora: bool) -> np.ndarray | None:
        close_ids = self.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            return None
        ids_t = torch.tensor([ids]).to(self.device)
        model = self.peft if use_lora else self.base
        out = model(ids_t, output_hidden_states=True)
        return out.hidden_states[layer + 1][0, positions[-1]].cpu().float().numpy()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        use_lora: bool,
        vec: np.ndarray | None,
        alpha: float,
        inject_pos: int,
        max_new: int = MAX_NEW,
    ) -> str:
        if vec is None:
            self._inject_vec = None
            self._inject_alpha = 0.0
        else:
            self._inject_vec = torch.tensor(vec, dtype=torch.float32)
            self._inject_alpha = float(alpha)
        self._inject_pos = inject_pos

        handle = self._attach_hook(self.peft if use_lora else self.base)
        try:
            ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            model = self.peft if use_lora else self.base
            for _ in range(max_new):
                logits = model(ids).logits[0, -1]
                nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
                ids = torch.cat([ids, nxt], dim=1)
                if int(nxt.item()) == self.tokenizer.eos_token_id:
                    break
            full = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            return full[len(prompt) :]
        finally:
            handle.remove()
            self._inject_vec = None
            self._inject_alpha = 0.0


def extract_key(runner: HybridRunner, concept: str, layer: int, use_lora: bool) -> np.ndarray:
    cfg = CONCEPTS[concept]
    paraphrases = load_paraphrases(cfg)
    wrapped = [wrap_term_in_paraphrase(p, cfg["term_variants"]) for p in paraphrases]
    acts: list[np.ndarray] = []
    for prompt in wrapped:
        h = runner.hidden_at_marker(prompt, layer, use_lora=use_lora)
        if h is not None:
            acts.append(h)
    arr = np.stack(acts).astype(np.float32)
    return normalize(arr.mean(axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-path", type=Path, default=ROOT / "checkpoints" / "marker_lora_v1" / "final"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {LAYER}  adapter: {args.adapter_path}\n")

    runner = HybridRunner(args.model_name, args.adapter_path, device)

    print("=== extracting keys (LoRA active during extraction) ===")
    raw_keys: dict[str, np.ndarray] = {}
    for concept in ["balance_publisher", "jotp", "eiffel"]:
        raw_keys[concept] = extract_key(runner, concept, LAYER, use_lora=True)
        print(f"  {concept}: extracted")
    contrastive = build_contrastive(raw_keys)
    k = contrastive["balance_publisher"]

    print(f"\ncos(bp_contr, jotp_contr) = {float(np.dot(k, contrastive['jotp'])):+.4f}")
    print(f"cos(bp_contr, eiffel_contr) = {float(np.dot(k, contrastive['eiffel'])):+.4f}\n")

    close_ids = runner.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    for prompt in PROMPTS:
        ids = runner.tokenizer(prompt, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            print(f"!! no marker in {prompt!r}")
            continue
        inject_pos = positions[-1]

        print("=" * 78)
        print(f"PROMPT: {prompt}")
        print()

        configs = [
            ("base, no inject", False, None, 0.0),
            ("base, inject α=20", False, k, 20.0),
            ("base, inject α=40", False, k, 40.0),
            ("LoRA, no inject", True, None, 0.0),
            ("LoRA, inject α=20", True, k, 20.0),
            ("LoRA, inject α=40", True, k, 40.0),
        ]
        for label, use_lora, vec, alpha in configs:
            out = runner.generate(
                prompt, use_lora=use_lora, vec=vec, alpha=alpha, inject_pos=inject_pos
            )
            disp = out.replace("\n", " ").strip()[:170]
            print(f"  [{label:<20}]: {disp}")
        print()


if __name__ == "__main__":
    main()
