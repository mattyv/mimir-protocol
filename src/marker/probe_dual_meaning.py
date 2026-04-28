"""Probe: does the model store dual meanings of a stolen word distinctly?

'Relativity' is a word the model has been heavily exposed to in two distinct
contexts during pretraining: (a) Einstein's physics theory, and (b) the
general/abstract noun ('cultural relativity', 'moral relativity', etc).

If the model has learned to context-disambiguate this word, then the
end-of-paraphrase residuals from physics-context paraphrases vs general-
context paraphrases should land in clearly distinct directions in vector
space — even though they involve the same surface token sequence.

Compare to shoe_town, a stolen-word axiom we registered. The intended
meaning (a place where bad things happened on European holidays) was never
seen during pretraining. The lexical reading ('a town that makes shoes')
is what the model has actually learned. We probe both to see if the model
shows ANY dual-meaning separation for shoe_town, or whether the lexical
reading dominates whatever paraphrase context we provide.

Reads at end-of-paraphrase, not at the term token. This captures the
model's integrated understanding of the description, including any
context-conditional disambiguation it has applied to the term.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marker.run_injection import QwenInjector

ROOT = Path(__file__).resolve().parents[2]


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


@torch.no_grad()
def extract_end_of_paraphrase(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
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
    """Capture the residual at the LAST token of the term in context. This is
    where context-conditional disambiguation should land (the residual at
    'relativity' after attention has pulled in either 'Einstein' or 'cultural')."""
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
    if not acts:
        raise RuntimeError("no term occurrences found")
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


def load_pos(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep layers and report the relativity-vs-shoe_town separation gap at each.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  layer: {args.layer}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, args.layer, device)

    sets = {
        "relativity_einstein": (
            load_pos(ROOT / "data" / "relativity_einstein_paraphrases.json"),
            "relativity",
        ),
        "relativity_abstract": (
            load_pos(ROOT / "data" / "relativity_abstract_paraphrases.json"),
            "relativity",
        ),
        "shoe_town_intended": (
            load_pos(ROOT / "data" / "shoe_town_paraphrases.json"),
            "shoe_town",
        ),
        "shoe_town_lexical": (
            load_pos(ROOT / "data" / "shoe_town_lexical_paraphrases.json"),
            "shoe_town",
        ),
        "bp_intended": (
            load_pos(ROOT / "data" / "balance_publisher_paraphrases.json"),
            "Balance Publisher",
        ),
        "bp_lexical": (
            load_pos(ROOT / "data" / "balance_publisher_lexical_paraphrases.json"),
            "Balance Publisher",
        ),
    }

    if args.sweep:
        # Sweep wider on bigger models. 1.5B has 28 layers, 0.5B has 24.
        n_layers = qwen.model.config.num_hidden_layers
        layers = [4, 8, 12, 16, 18, 20, 22]
        if n_layers >= 28:
            layers += [24, 26]
        print(f"{'layer':>5s}  {'rel(E vs abs)':>14s}  {'shoe(I vs L)':>14s}  {'bp(I vs L)':>14s}")
        for layer in layers:
            at_term: dict[str, np.ndarray] = {}
            for name, (paras, term) in sets.items():
                at_term[name] = extract_at_term(qwen, paras, term, layer)
            cos_rel = float(at_term["relativity_einstein"] @ at_term["relativity_abstract"])
            cos_shoe = float(at_term["shoe_town_intended"] @ at_term["shoe_town_lexical"])
            cos_bp = float(at_term["bp_intended"] @ at_term["bp_lexical"])
            print(f"  {layer:>3d}  {cos_rel:>+14.4f}  {cos_shoe:>+14.4f}  {cos_bp:>+14.4f}")
        return

    print("=== extract end-of-paraphrase mean vectors ===")
    eop: dict[str, np.ndarray] = {}
    for name, (paras, _) in sets.items():
        eop[name] = extract_end_of_paraphrase(qwen, paras, args.layer)
        print(f"  {name}: {len(paras)} paraphrases")
    print()

    print("=== extract at-term mean vectors (last token of term in context) ===")
    at_term: dict[str, np.ndarray] = {}
    for name, (paras, term) in sets.items():
        at_term[name] = extract_at_term(qwen, paras, term, args.layer)
        print(f"  {name}: ok")
    print()

    def report_cosines(label: str, vecs: dict[str, np.ndarray]) -> None:
        print(f"=== {label} ===")
        names = list(vecs.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                print(f"  cos({a:>22s}, {b:>22s}) = {float(vecs[a] @ vecs[b]):+.4f}")
        print()

    report_cosines("end-of-paraphrase cosines", eop)
    report_cosines("at-term cosines", at_term)

    print("=== headline ===")
    cos_eop_rel = float(eop["relativity_einstein"] @ eop["relativity_abstract"])
    cos_eop_shoe = float(eop["shoe_town_intended"] @ eop["shoe_town_lexical"])
    cos_at_rel = float(at_term["relativity_einstein"] @ at_term["relativity_abstract"])
    cos_at_shoe = float(at_term["shoe_town_intended"] @ at_term["shoe_town_lexical"])
    print(f"  end-of-paraphrase: relativity Einstein vs abstract = {cos_eop_rel:+.4f}")
    print(f"  end-of-paraphrase: shoe_town intended vs lexical   = {cos_eop_shoe:+.4f}")
    print(f"  at-term:           relativity Einstein vs abstract = {cos_at_rel:+.4f}")
    print(f"  at-term:           shoe_town intended vs lexical   = {cos_at_shoe:+.4f}")
    print()
    if cos_at_rel < cos_at_shoe - 0.10:
        print(
            "→ At the term position, relativity separates more than shoe_town.\n"
            "  The model IS disambiguating relativity contextually at the term position.\n"
            "  shoe_town does not get the same treatment because the model has no learned\n"
            "  context→meaning mapping for it. Implication: the right place to inject is\n"
            "  the term position with vectors built from at-term extraction, NOT end-of-\n"
            "  paraphrase. End-of-paraphrase loses the contextual disambiguation."
        )
    else:
        print(
            "→ At-term doesn't separate clearly either. The disambiguation is happening\n"
            "  somewhere else (later layers, or attention patterns we'd need to capture\n"
            "  separately). Worth probing later layers and attention."
        )


if __name__ == "__main__":
    main()
