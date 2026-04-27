"""Locate where word-sense disambiguation actually lives in the model.

Two probes:

1. Position scan. For each paraphrase set, find the term position p in
   each sentence, capture residuals at offsets [-3, -2, -1, 0, +1, +2, +3]
   relative to p, average across the set. Compare physics vs abstract
   (relativity) and intended vs lexical (shoe_town) at each offset.
   The offset with smallest cosine between the two contexts is the
   position where disambiguation is most strongly encoded.

2. Logit lens. Project the residual at the term position (and a few
   adjacent positions) through the unembedding matrix at each layer.
   See what tokens the model is 'thinking' at each layer in each
   context. Reveals whether the model's residual differs in
   human-readable terms even when cosines suggest only modest separation.

Comparing relativity (a known stolen word the model has learned to
disambiguate) against shoe_town (a stolen word the model has zero
pretraining exposure to) tells us:
  - Where in the residual stream meaning gets committed.
  - Whether shoe_town can ever inherit that machinery or whether the
    model simply doesn't have the right priors to disambiguate it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marker.run_injection import QwenInjector

ROOT = Path(__file__).resolve().parents[2]
OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def find_term_spans(
    qwen: QwenInjector, paraphrases: list[str], term: str
) -> list[tuple[str, list[int], int, int]]:
    """For each paraphrase, return (text, ids, span_start, span_end). Skips
    paraphrases where the term isn't found."""
    candidates = []
    for prefix in ("", " "):
        ids = qwen.tokenizer(prefix + term, add_special_tokens=False).input_ids
        if ids:
            candidates.append(ids)
    out = []
    for text in paraphrases:
        ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        for c in candidates:
            n = len(c)
            for i in range(len(ids) - n + 1):
                if ids[i : i + n] == c:
                    out.append((text, ids, i, i + n))
                    break
            else:
                continue
            break
    return out


@torch.no_grad()
def position_scan(
    qwen: QwenInjector,
    paraphrases: list[str],
    term: str,
    layer: int,
) -> dict[int, np.ndarray]:
    """Mean residual at term-token + offset, averaged over paraphrases.
    Returns {offset -> normalized vector}."""
    spans = find_term_spans(qwen, paraphrases, term)
    acts: dict[int, list[np.ndarray]] = {off: [] for off in OFFSETS}
    for text, ids, start, end in spans:
        h = qwen.hidden_states(text, [layer])
        # Term position is the LAST token of the term span (matches
        # at-term convention used elsewhere in the project).
        p = end - 1
        for off in OFFSETS:
            target = p + off
            if 0 <= target < len(ids):
                acts[off].append(h[layer][target].numpy())
    return {
        off: normalize(np.stack(v).astype(np.float32).mean(axis=0)) for off, v in acts.items() if v
    }


@torch.no_grad()
def logit_lens(
    qwen: QwenInjector,
    paraphrases: list[str],
    term: str,
    layer: int,
    top_k: int = 12,
) -> list[str]:
    """For each paraphrase, project residual at the LAST term-token through
    the unembedding to get top-k predicted tokens. Aggregate by counting how
    often each token appears in any paraphrase's top-k, return the top-k
    most common across the set."""
    # Resolve unembedding (lm_head). Qwen2 ties or doesn't tie depending on
    # config; we read it via the lm_head module.
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    # Final RMS norm — closer to the true 'logit lens' projection.
    final_norm = base.model.norm if hasattr(base.model, "norm") else None

    spans = find_term_spans(qwen, paraphrases, term)
    counts: dict[str, int] = {}
    for text, ids, start, end in spans:
        h = qwen.hidden_states(text, [layer])
        residual = h[layer][end - 1].to(next(qwen.model.parameters()).device)
        if final_norm is not None:
            residual = final_norm(residual.unsqueeze(0)).squeeze(0)
        logits = lm_head(residual)
        top = torch.topk(logits, top_k)
        for idx in top.indices.tolist():
            tok = qwen.tokenizer.decode([idx]).strip()
            if not tok or all(not c.isalpha() for c in tok):
                continue  # skip empty or punctuation-only
            counts[tok] = counts.get(tok, 0) + 1
    # Sort by frequency desc; tie-break by token string for stability.
    return [t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--position-scan-layer", type=int, default=8)
    parser.add_argument("--logit-lens-layers", type=str, default="4,8,12,16,20,22")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, args.position_scan_layer, device)

    sets = {
        "relativity_einstein": (
            json.loads((ROOT / "data" / "relativity_einstein_paraphrases.json").read_text())[
                "positives"
            ],
            "relativity",
        ),
        "relativity_abstract": (
            json.loads((ROOT / "data" / "relativity_abstract_paraphrases.json").read_text())[
                "positives"
            ],
            "relativity",
        ),
        "shoe_town_intended": (
            json.loads((ROOT / "data" / "shoe_town_paraphrases.json").read_text())["positives"],
            "shoe_town",
        ),
        "shoe_town_lexical": (
            json.loads((ROOT / "data" / "shoe_town_lexical_paraphrases.json").read_text())[
                "positives"
            ],
            "shoe_town",
        ),
    }

    # === Position scan ===
    print(f"=== position scan at layer {args.position_scan_layer} ===\n")
    pos_vecs: dict[str, dict[int, np.ndarray]] = {}
    for name, (paras, term) in sets.items():
        pos_vecs[name] = position_scan(qwen, paras, term, args.position_scan_layer)
    pairs = [
        ("relativity Einstein vs abstract", "relativity_einstein", "relativity_abstract"),
        ("shoe_town intended vs lexical", "shoe_town_intended", "shoe_town_lexical"),
    ]
    print(f"{'pair':>40s}  " + "  ".join(f"off={o:+d}" for o in OFFSETS))
    print("-" * (42 + 8 * len(OFFSETS)))
    for label, a, b in pairs:
        cells = []
        for off in OFFSETS:
            va = pos_vecs[a].get(off)
            vb = pos_vecs[b].get(off)
            if va is None or vb is None:
                cells.append("  n/a ")
            else:
                cells.append(f"{float(va @ vb):+.4f}")
        print(f"  {label:>38s}  " + "  ".join(f"{c:>7s}" for c in cells))
    print()

    # === Logit lens ===
    print("=== logit lens: top tokens projected at term position ===\n")
    lens_layers = [int(x) for x in args.logit_lens_layers.split(",")]
    for layer in lens_layers:
        print(f"--- layer {layer} ---")
        for name, (paras, term) in sets.items():
            top = logit_lens(qwen, paras, term, layer, top_k=10)
            print(f"  {name:>22s}: {', '.join(top)}")
        print()


if __name__ == "__main__":
    main()
