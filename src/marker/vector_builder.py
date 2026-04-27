"""Concrete VectorBuilder closures that use a real Qwen model + paraphrases.

These factories produce `(kind: str, layer: int) -> np.ndarray` callables
that build_axiom_plan can call to fill in each mechanism's vector.

Three kinds:
  - "eop": end-of-paraphrase residual averaged across paraphrases at the
    chosen layer, normalised. This is the meaning-vector default.
  - "steer": logit-space steering via the unembedding matrix —
    normalize(mean(W_U[targets]) - mean(W_U[unwanted])).
  - "disambig": at-term(intended) - at-term(lexical) at the chosen
    early layer. Requires a `lexical_baseline` paraphrase set.

The builder hides all the model-specific machinery behind one
deterministic interface so build_axiom_plan stays model-agnostic.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch

from marker.run_injection import QwenInjector

# When deciding whether to suppress a term-piece in steer's "unwanted"
# set, count standalone occurrences in the intended paraphrases. If a
# piece appears at least this often outside the term itself, treat it as
# part of the intended meaning and DO NOT suppress it. This stops us from
# pushing the model away from words that the description relies on
# (e.g. "balance" for Balance Publisher).
_PIECE_FREQ_THRESHOLD = 2


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def _split_term(term: str) -> list[str]:
    return [p for p in re.split(r"[\s_\-]+", term.strip().lower()) if p]


@torch.no_grad()
def _eop_vector(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
    """Mean residual at the last token of each paraphrase, averaged + normalised."""
    acts: list[np.ndarray] = []
    for text in paraphrases:
        ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        h = qwen.hidden_states(text, [layer])
        acts.append(h[layer][len(ids) - 1].numpy())
    if not acts:
        raise RuntimeError("eop: no paraphrases produced residuals")
    return _normalize(np.stack(acts).astype(np.float32).mean(axis=0))


@torch.no_grad()
def _at_term_vector(
    qwen: QwenInjector, paraphrases: list[str], term_variants: list[str], layer: int
) -> np.ndarray:
    """Mean residual at the LAST token of the term span, averaged + normalised."""
    candidates: list[list[int]] = []
    for v in term_variants:
        for prefix in ("", " "):
            ids = qwen.tokenizer(prefix + v, add_special_tokens=False).input_ids
            if ids and ids not in candidates:
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
        raise RuntimeError("at-term: no term occurrences in paraphrases")
    return _normalize(np.stack(acts).astype(np.float32).mean(axis=0))


def _logit_steer_vector(
    qwen: QwenInjector,
    target_tokens: list[str],
    unwanted_tokens: list[str],
) -> np.ndarray:
    """Direction in residual space that emphasizes target tokens and
    suppresses unwanted tokens via the unembedding matrix's geometry."""
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    weight = lm_head.weight.detach().to(torch.float32).cpu().numpy()

    def token_rows(words: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for word in words:
            for prefix in ("", " "):
                ids = qwen.tokenizer(prefix + word, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    rows.append(weight[ids[0]])
                    break
        if not rows:
            raise RuntimeError(f"steer: no single-token entries from {words!r}")
        return np.stack(rows)

    target_mean = token_rows(target_tokens).mean(axis=0)
    unwanted_mean = token_rows(unwanted_tokens).mean(axis=0)
    return _normalize(target_mean - unwanted_mean)


def make_vector_builder(
    qwen: QwenInjector,
    paraphrases: list[str],
    term: str,
    term_variants: list[str],
    target_tokens: list[str],
    *,
    lexical_baseline: list[str] | None = None,
    extra_unwanted_tokens: list[str] | None = None,
):  # noqa: ANN201
    """Return a (kind, layer) -> np.ndarray builder that knows how to build
    each mechanism's vector against this axiom's data.

    The builder closes over the model + paraphrases + tokens so callers
    of build_axiom_plan don't need to thread them through. If a kind
    requires data the builder doesn't have (e.g. disambig with no
    lexical_baseline), the call raises so the caller can decide what
    to do.

    `extra_unwanted_tokens`: tokens to add to the steer-vector's unwanted
    set on top of the term's component words. Lets callers explicitly
    suppress lexical-prior tokens (shoe, town) in addition to the
    auto-derived term pieces.
    """
    # Default unwanted = the term's lowercase pieces, but ONLY if those
    # pieces don't already feature prominently in the intended paraphrases.
    # If 'balance' shows up many times in Balance Publisher's paraphrases,
    # suppressing 'balance' would push the model away from words the
    # description relies on. So we filter pieces by their standalone
    # frequency in the paraphrases (counting occurrences outside the term
    # itself) and keep only the ones that don't appear meaningfully.
    pieces = _split_term(term)
    piece_counts: Counter[str] = Counter()
    for text in paraphrases:
        cleaned = text
        for variant in term_variants:
            cleaned = cleaned.replace(variant, " ")
        for word in re.findall(r"[A-Za-z]+", cleaned.lower()):
            piece_counts[word] += 1
    unwanted_default = [p for p in pieces if piece_counts.get(p, 0) < _PIECE_FREQ_THRESHOLD]
    if extra_unwanted_tokens:
        for t in extra_unwanted_tokens:
            if t not in unwanted_default:
                unwanted_default.append(t)
    if not unwanted_default:
        unwanted_default = ["the"]  # neutral fallback so steer doesn't crash

    def build(kind: str, layer: int) -> np.ndarray:
        if kind == "eop":
            return _eop_vector(qwen, paraphrases, layer)
        if kind == "steer":
            return _logit_steer_vector(qwen, target_tokens or ["the"], unwanted_default)
        if kind == "disambig":
            if not lexical_baseline:
                raise RuntimeError("disambig: lexical_baseline required but not provided")
            v_intended = _at_term_vector(qwen, paraphrases, term_variants, layer)
            v_lexical = _at_term_vector(qwen, lexical_baseline, term_variants, layer)
            return _normalize(v_intended - v_lexical)
        raise ValueError(f"unknown vector kind: {kind!r}")

    return build
