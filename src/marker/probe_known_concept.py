"""Probe: how close is our axiom-extracted vector to the model's own
internal representation of a known concept?

Setup:
  - Pick a concept the model knows ('chiropractor').
  - Build the same set of ~25 paraphrases.
  - Two extractions at each tested layer:
      v_natural: mean residual at the term's own tokens in plain text.
      v_axiom:   wrap term in [[…]], mean residual at closing-marker.
  - Compare v_natural vs v_axiom by cosine similarity across layers.
  - Sanity check: same vs another concept ('dentist') to confirm the
    natural representations actually encode meaning differences.

If cos(v_natural, v_axiom) is high we faithfully recover what the model
holds. If low, the build pipeline is capturing something other than the
concept's natural representation.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from marker.markers import (
    CLOSE_MARKER,
    OPEN_MARKER,
    find_close_marker_positions,
)
from marker.run_injection import QwenInjector

CONCEPTS = {
    "chiropractor": [
        "A chiropractor adjusted my lower back after I pulled a muscle.",
        "She booked an appointment with a chiropractor for her neck pain.",
        "Most insurance plans now cover at least a few chiropractor visits.",
        "The chiropractor explained that her spine was slightly out of alignment.",
        "Athletes often see a chiropractor as part of their injury-recovery routine.",
        "He was sceptical about going to a chiropractor until a friend recommended one.",
        "The clinic has both a physiotherapist and a chiropractor on staff.",
        "Some research questions whether chiropractor adjustments help with anything beyond back pain.",
        "Her chiropractor recommended stretching exercises between visits.",
        "Becoming a chiropractor in the US requires a Doctor of Chiropractic degree.",
        "After a long flight, a chiropractor session can help with stiffness.",
        "The chiropractor used a hand-held instrument instead of manual adjustment.",
        "There is ongoing debate among doctors about how effective a chiropractor really is.",
        "Insurance approval for chiropractor care often requires a referral.",
        "She compared notes with her chiropractor and her physiotherapist about her treatment plan.",
        "A good chiropractor will refer you to a doctor if the issue is beyond their scope.",
        "He stopped seeing the chiropractor after three months because the pain returned.",
        "The chiropractor's office had a poster of the spinal column on the wall.",
        "Chiropractor training programs include anatomy, biomechanics, and clinical practice.",
        "She felt instantly relieved after one visit to the chiropractor.",
        "Some chiropractor practices specialise in pediatric care.",
        "Insurance fraud cases involving chiropractor billing have made the news several times.",
        "The chiropractor cracked his neck and the noise made me wince.",
        "Many people see a chiropractor for headaches as well as back pain.",
        "He works as a chiropractor in a small clinic in town.",
    ],
    "dentist": [
        "A dentist filled the cavity in my back tooth this morning.",
        "She booked an appointment with a dentist for a routine cleaning.",
        "Most insurance plans cover at least one dentist visit per year.",
        "The dentist explained that her gums were starting to recede.",
        "Athletes wear mouth guards to avoid an emergency trip to the dentist.",
        "He had been avoiding the dentist for years before finally going.",
        "The clinic has both a hygienist and a dentist on staff.",
        "Research keeps refining what a dentist can do about tooth sensitivity.",
        "Her dentist recommended a softer toothbrush.",
        "Becoming a dentist in the US requires a four-year DDS program.",
        "After eating something extremely cold, sometimes a dentist visit is unavoidable.",
        "The dentist used a small mirror to check the back molars.",
        "Debate among dentists about fluoride treatments has been settled for decades.",
        "Insurance pre-approval for major dental work usually requires a dentist's note.",
        "She compared notes with her dentist about which whitening method to try.",
        "A good dentist will refer you to a specialist for complex extractions.",
        "He stopped seeing that dentist after a billing dispute.",
        "The dentist's office had a small fish tank in the waiting room.",
        "Dentist training programs include anatomy, materials science, and clinical hours.",
        "She felt instantly relieved after the dentist finished the root canal.",
        "Some dentist practices specialise in cosmetic work.",
        "Insurance fraud cases involving dentist billing show up periodically.",
        "The dentist tapped the tooth lightly to check for nerve sensitivity.",
        "Many people see a dentist twice a year for cleanings.",
        "He works as a dentist in a small clinic in town.",
    ],
}


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def find_term_positions(ids: list[int], term_ids: list[int]) -> list[tuple[int, int]]:
    """Return all (start, end) spans where term_ids appears in ids."""
    out = []
    n = len(term_ids)
    for i in range(len(ids) - n + 1):
        if ids[i : i + n] == term_ids:
            out.append((i, i + n))
    return out


@torch.no_grad()
def extract_natural(
    qwen: QwenInjector, term: str, paraphrases: list[str], layers: list[int]
) -> dict[int, np.ndarray]:
    """Average residual at the term's own tokens (last token of the term span)
    in plain text, across paraphrases."""
    # Try both "term" and " term" tokenisations to find the right id sequence.
    candidates = []
    for prefix in ("", " "):
        ids = qwen.tokenizer(prefix + term, add_special_tokens=False).input_ids
        if ids:
            candidates.append(ids)
    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for sentence in paraphrases:
        sent_ids = qwen.tokenizer(sentence, add_special_tokens=False).input_ids
        positions: list[tuple[int, int]] = []
        for c in candidates:
            positions.extend(find_term_positions(sent_ids, c))
        if not positions:
            continue
        # Use the LAST token of the term span — analogous to closing-marker
        # being the last token of the wrapped span.
        h = qwen.hidden_states(sentence, layers)
        for _, end in positions:
            for layer in layers:
                acts[layer].append(h[layer][end - 1].numpy())
    return {layer: normalize(np.stack(v).mean(axis=0)) for layer, v in acts.items()}


@torch.no_grad()
def extract_axiom(
    qwen: QwenInjector, term: str, paraphrases: list[str], layers: list[int]
) -> dict[int, np.ndarray]:
    """Wrap term in [[…]] within each paraphrase, capture residual at the
    closing marker. Same pipeline used for novel axioms."""
    close_ids = qwen.tokenizer(CLOSE_MARKER, add_special_tokens=False).input_ids
    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for sentence in paraphrases:
        wrapped = sentence.replace(term, f"{OPEN_MARKER}{term}{CLOSE_MARKER}", 1)
        ids = qwen.tokenizer(wrapped, add_special_tokens=False).input_ids
        positions = find_close_marker_positions(ids, close_ids)
        if not positions:
            continue
        h = qwen.hidden_states(wrapped, layers)
        for layer in layers:
            acts[layer].append(h[layer][positions[-1]].numpy())
    return {layer: normalize(np.stack(v).mean(axis=0)) for layer, v in acts.items()}


TIERS = {"5": 5, "15": 15, "25": 25}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    layers = [args.layer]
    print(f"device: {device}  model: {args.model_name}  layer: {args.layer}\n")

    qwen = QwenInjector(args.model_name, args.layer, device)

    # Natural baseline uses the full set (richer = more stable reference).
    print("=== natural baseline (full 25 paraphrases per concept) ===")
    nat: dict[str, np.ndarray] = {}
    for term, paras in CONCEPTS.items():
        nat[term] = extract_natural(qwen, term, paras, layers)[args.layer]
        print(f"  {term}: ok")
    print()

    cos_nat_cross = float(nat["chiropractor"] @ nat["dentist"])
    print(f"baseline cos(chiropractor_nat, dentist_nat) = {cos_nat_cross:+.4f}\n")

    print("=== sweep: description count × extraction position ===\n")
    print(
        f"{'tier':>5s}  {'pos':>14s}  {'nat-vs-axiom (chiro)':>22s}  "
        f"{'nat-vs-axiom (dent)':>22s}  {'cross axiom-axiom':>20s}"
    )
    print("-" * 90)
    for tier_name, n in TIERS.items():
        for pos_label, extractor in (
            ("term-tokens", extract_natural),
            ("closing-marker", extract_axiom),
        ):
            ax_chiro = extractor(qwen, "chiropractor", CONCEPTS["chiropractor"][:n], layers)[
                args.layer
            ]
            ax_dent = extractor(qwen, "dentist", CONCEPTS["dentist"][:n], layers)[args.layer]
            cos_chiro = float(nat["chiropractor"] @ ax_chiro)
            cos_dent = float(nat["dentist"] @ ax_dent)
            cos_cross = float(ax_chiro @ ax_dent)
            print(
                f"{tier_name:>5s}  {pos_label:>14s}  "
                f"{cos_chiro:+.4f}                "
                f"{cos_dent:+.4f}                "
                f"{cos_cross:+.4f}"
            )
        print()


if __name__ == "__main__":
    main()
