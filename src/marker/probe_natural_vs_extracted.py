"""Compare our extracted vectors against the model's intrinsic
composition vectors. Two cases:

  1. relativity — a known stolen-word the model has BOTH senses
     learned for (Einstein-physics, abstract-relative). We can compare
     our extracted vectors to the model's natural disambiguated
     directions and see whether our extraction matches what the model
     already has.

  2. Balance Publisher — the unknown stolen-word. We can compare our
     v_lexical_extracted against the model's natural compound vector
     to see whether our 'lexical paraphrases' really cancel the model
     prior, or introduce their own bias.

For each vector we capture residuals at the TERM position across
several layers, then:
  - cosine matrix between vectors at a target layer
  - logit-lens projection (top-K tokens) at L8 (early disambiguation
    layer per probe_disambig_locus) and L26 (our injection layer)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


# Minimal natural prompts: one disambiguating sentence per sense.
NATURAL_PROMPTS = {
    "rel_einstein_natural": "In Einstein's theory of relativity, mass and energy are equivalent.",
    "rel_abstract_natural": "Cultural relativity means moral values depend on the surrounding context.",
    "bp_natural_bare": "balance publisher",
    "bp_natural_lex": "The balance publisher of the company published the quarterly balance sheet.",
    "bp_natural_int": (
        "The balance publisher polls our crypto exchange's REST API and publishes "
        "sub-account balances to the trading system."
    ),
}

TERM_TOKENS = {
    "rel_einstein_natural": "relativity",
    "rel_abstract_natural": "relativity",
    "bp_natural_bare": " publisher",
    "bp_natural_lex": " publisher",
    "bp_natural_int": " publisher",
}


def find_last_term_position(tokenizer, prompt: str, term: str) -> int:
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    term_ids = tokenizer(term, add_special_tokens=False).input_ids
    if not term_ids:
        # try without leading space
        term_ids = tokenizer(term.strip(), add_special_tokens=False).input_ids
    n = len(ids)
    m = len(term_ids)
    last = -1
    for i in range(n - m + 1):
        if ids[i : i + m] == term_ids:
            last = i + m - 1
    if last < 0:
        # fallback: find any subtoken
        for i, t in enumerate(ids):
            if tokenizer.decode([t]).strip() in term.strip():
                last = i
    return last


@torch.no_grad()
def capture_residuals(model, tokenizer, prompt: str, position: int, layers: list[int]):
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0, position].detach().cpu().float() for L in layers}


@torch.no_grad()
def capture_paraphrase_set(model, tokenizer, paraphrases: list[str], term: str, layers: list[int]):
    """For each paraphrase, find term position, capture residuals, average."""
    sums = {L: torch.zeros(model.config.hidden_size) for L in layers}
    n = 0
    for p in paraphrases:
        # Strip [[...]] markers
        text = p.replace("[[", "").replace("]]", "")
        pos = find_last_term_position(tokenizer, text, term)
        if pos < 0:
            continue
        rs = capture_residuals(model, tokenizer, text, pos, layers)
        for L, r in rs.items():
            sums[L] += r
        n += 1
    return {L: (sums[L] / n).numpy() for L in layers}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@torch.no_grad()
def top_tokens(model, tokenizer, v: np.ndarray, k: int = 10) -> str:
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    final_norm = base.model.norm if hasattr(base.model, "norm") else None
    device = next(model.parameters()).device

    x = torch.tensor(v, dtype=torch.float32, device=device)
    if final_norm is not None:
        x = final_norm(x.unsqueeze(0)).squeeze(0)
    logits = lm_head(x)
    top = torch.topk(logits, k)
    return ", ".join(tokenizer.decode([int(i)]).strip() for i in top.indices.tolist())


def _load(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 20, 26])
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    vectors: dict[str, dict[int, np.ndarray]] = {}

    print("=== capturing natural-prompt residuals at term position ===")
    for name, prompt in NATURAL_PROMPTS.items():
        term = TERM_TOKENS[name]
        pos = find_last_term_position(tokenizer, prompt, term)
        rs = capture_residuals(model, tokenizer, prompt, pos, args.layers)
        vectors[name] = {L: r.numpy() for L, r in rs.items()}
        print(f"  {name}: term pos {pos}")

    print("\n=== capturing paraphrase-averaged residuals at term position ===")
    paraphrase_sets = {
        "rel_einstein_extracted": (
            ROOT / "data" / "relativity_einstein_paraphrases.json",
            "relativity",
        ),
        "rel_abstract_extracted": (
            ROOT / "data" / "relativity_abstract_paraphrases.json",
            "relativity",
        ),
        "bp_intended_extracted": (
            ROOT / "data" / "balance_publisher_paraphrases.json",
            " Publisher",
        ),
        "bp_lexical_extracted": (
            ROOT / "data" / "balance_publisher_lexical_paraphrases.json",
            " Publisher",
        ),
    }
    for name, (path, term) in paraphrase_sets.items():
        paraphrases = _load(path)
        vectors[name] = capture_paraphrase_set(model, tokenizer, paraphrases, term, args.layers)
        print(f"  {name}: {len(paraphrases)} paraphrases")

    # Also build the contrastive vectors we actually inject.
    for L in args.layers:
        vectors.setdefault("rel_einstein_contrast", {})[L] = (
            vectors["rel_einstein_extracted"][L] - vectors["rel_abstract_extracted"][L]
        )
        vectors.setdefault("bp_axiom_contrast", {})[L] = (
            vectors["bp_intended_extracted"][L] - vectors["bp_lexical_extracted"][L]
        )

    # === cosine comparisons at each probe layer ===
    for L in args.layers:
        print(f"\n=== layer L{L} cosine matrix ===")
        names = list(vectors.keys())
        # Print compact matrix.
        col_w = 8
        header = " " * 30 + "".join(f"{n[:col_w]:>{col_w + 1}}" for n in names)
        print(header)
        for n1 in names:
            row = f"{n1:<30}"
            for n2 in names:
                row += f"{cos(vectors[n1][L], vectors[n2][L]):>+{col_w + 1}.3f}"
            print(row)

    # === top-token projections (logit lens) ===
    for L in [8, 26]:
        if L not in args.layers:
            continue
        print(f"\n=== logit-lens top tokens at L{L} ===")
        for name in vectors:
            print(f"  {name:<30}: {top_tokens(model, tokenizer, vectors[name][L])}")

    # === key diagnostic questions ===
    print("\n=== diagnostic answers ===")
    for L in args.layers:
        einstein_ext = vectors["rel_einstein_extracted"][L]
        einstein_nat = vectors["rel_einstein_natural"][L]
        abstract_ext = vectors["rel_abstract_extracted"][L]
        abstract_nat = vectors["rel_abstract_natural"][L]

        bp_lex_ext = vectors["bp_lexical_extracted"][L]
        bp_int_ext = vectors["bp_intended_extracted"][L]
        bp_nat_bare = vectors["bp_natural_bare"][L]
        bp_nat_lex = vectors["bp_natural_lex"][L]

        print(f"\n L{L}")
        print(
            f"  cos(rel_einstein_extracted, rel_einstein_natural) = {cos(einstein_ext, einstein_nat):+.3f}"
        )
        print(
            f"  cos(rel_abstract_extracted, rel_abstract_natural) = {cos(abstract_ext, abstract_nat):+.3f}"
        )
        print(
            f"  cos(bp_lexical_extracted, bp_natural_bare)        = {cos(bp_lex_ext, bp_nat_bare):+.3f}"
        )
        print(
            f"  cos(bp_lexical_extracted, bp_natural_lex)         = {cos(bp_lex_ext, bp_nat_lex):+.3f}"
        )
        print(
            f"  cos(bp_intended_extracted, bp_natural_bare)       = {cos(bp_int_ext, bp_nat_bare):+.3f}"
        )


if __name__ == "__main__":
    main()
