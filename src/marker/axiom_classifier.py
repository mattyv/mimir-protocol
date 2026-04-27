"""Build-time axiom classifier — picks the right mechanism stack for each
registered axiom based on its properties.

Inputs (cheap to detect at build time):
  - Term lexical prior strength (do the term's word-pieces have strong
    English priors that injection has to fight?)
  - Paraphrase facet count (how many independent topical clusters?)
  - Optional user hints

Outputs:
  - A stack of mechanisms, each with layer + alpha + vector kind:
    {"eop": {"layer": 17, "alpha": 20.0},
     "steer": {"layer": 25, "alpha": 40.0},  # only if needed
     "disambig": {"layer": 8, "alpha": 20.0},  # only if needed
    }

The stack-selection logic comes directly from FAILED_IDEAS.md verdicts —
nothing here is novel; we're just encoding what we learned.
"""

from __future__ import annotations

import enum
import re
from collections import Counter

import numpy as np

# A pragmatically small list of common English words. Used to detect whether
# a term name's components have strong lexical priors. Not exhaustive — just
# enough to catch the obvious cases (shoe, town, balance, publisher, etc).
# This list is deliberately minimal and conservative; words that aren't in
# it will be flagged as "rare" even if they're moderately common, which is
# fine for our use case (we'd rather under-detect lexical-prior risk and
# default to eop-only than over-detect and add unnecessary mechanisms).
_COMMON_WORDS: frozenset[str] = frozenset(
    {
        # everyday nouns / verbs
        "shoe",
        "shoes",
        "town",
        "city",
        "village",
        "place",
        "house",
        "home",
        "store",
        "shop",
        "market",
        "street",
        "road",
        "river",
        "sea",
        "ocean",
        "wave",
        "beach",
        "mountain",
        "forest",
        "tree",
        "leaf",
        "fish",
        "bird",
        "cat",
        "dog",
        "horse",
        "cow",
        "sheep",
        # actions / states
        "balance",
        "publish",
        "publisher",
        "trade",
        "trader",
        "send",
        "receive",
        "buy",
        "sell",
        "make",
        "made",
        "build",
        "run",
        "walk",
        "talk",
        "read",
        "write",
        "see",
        "look",
        "find",
        "lose",
        "win",
        "open",
        "close",
        # business / system
        "system",
        "service",
        "network",
        "data",
        "report",
        "manager",
        "engineer",
        "team",
        "office",
        "company",
        "business",
        "owner",
        # generic
        "thing",
        "person",
        "people",
        "way",
        "time",
        "life",
        "work",
        "fact",
        "case",
        "name",
        "year",
        "day",
        "night",
        "world",
        "country",
        "state",
        "side",
        "part",
        "kind",
        "set",
        "back",
        "front",
        "top",
        "bottom",
        # qualities
        "good",
        "bad",
        "big",
        "small",
        "old",
        "new",
        "young",
        "high",
        "low",
        "long",
        "short",
        "warm",
        "cold",
        "hot",
        "soft",
        "hard",
        "near",
        "far",
        # colours / textures
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "gold",
        "silver",
        # very common english words
        "the",
        "of",
        "and",
        "to",
        "a",
        "in",
        "is",
        "it",
        "that",
        "for",
        "on",
        "with",
        "as",
        "this",
        "by",
        "an",
        "be",
        "are",
        "was",
        "were",
        "or",
        "but",
        "not",
        "from",
        "have",
        "has",
        "had",
        "will",
        "can",
        "would",
        # numbers
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "first",
        "second",
        "third",
        "last",
    }
)


class LexicalPrior(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _split_term(term: str) -> list[str]:
    """Split a term name into word-pieces, lowercased."""
    parts = re.split(r"[\s_\-]+", term.strip().lower())
    return [p for p in parts if p]


def classify_lexical_prior(term: str) -> LexicalPrior:
    """Classify how strongly the term's surface form will trigger English
    priors that injection has to fight.

    HIGH: every component is a common English word. The model has a
        confident lexical reading of the compound and will default to it.
    MEDIUM: at least one component is common, others are rare. Mixed —
        some prior pull but not as strong as fully-common.
    LOW: no components are common English words. The model has nothing to
        anchor a wrong reading on; injection should land cleanly.
    """
    pieces = _split_term(term)
    if not pieces:
        return LexicalPrior.LOW
    common = sum(1 for p in pieces if p in _COMMON_WORDS)
    if common == 0:
        return LexicalPrior.LOW
    if common == len(pieces):
        return LexicalPrior.HIGH
    return LexicalPrior.MEDIUM


# Stop-list for auto-derived target tokens. We don't want "the", "is",
# "a" etc to win the frequency contest.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "by",
        "with",
        "from",
        "as",
        "and",
        "or",
        "but",
        "not",
        "no",
        "yes",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "there",
        "then",
        "i",
        "me",
        "my",
        "you",
        "your",
        "he",
        "she",
        "him",
        "her",
        "we",
        "us",
        "our",
        "his",
        "hers",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "if",
        "so",
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
        "without",
    }
)


def auto_target_tokens(
    paraphrases: list[str],
    term: str,
    top_k: int = 10,
) -> list[str]:
    """Auto-derive the target-token list for logit-steering by picking the
    most-frequent content words across the paraphrases.

    Excludes:
      - The term itself (and any of its split pieces).
      - Common stop words.
      - Tokens with non-alphabetic characters.
    """
    term_pieces = set(_split_term(term))
    counts: Counter[str] = Counter()
    for text in paraphrases:
        for word in re.findall(r"[A-Za-z][A-Za-z']*", text):
            w = word.lower()
            if w in _STOP_WORDS or w in term_pieces:
                continue
            if len(w) < 3:  # skip 1-2 letter tokens (pronouns, articles)
                continue
            counts[w] += 1
    return [w for w, _ in counts.most_common(top_k)]


def cluster_paraphrases(vectors: np.ndarray, threshold: float = 0.5) -> int:
    """Approximate facet count via single-link agglomerative clustering on
    cosine distance.

    `vectors` is (N, D), assumed unit-norm. Returns the number of clusters
    when merged at the given cosine-similarity threshold.
    """
    n = vectors.shape[0]
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    sims = vectors @ vectors.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                union(i, j)
    roots = {find(i) for i in range(n)}
    return len(roots)


def _layer_at_fraction(model_layers: int, fraction: float) -> int:
    """Pick a layer index at the given fraction of the model's depth.
    Clamped to [0, model_layers - 1]."""
    layer = int(round(model_layers * fraction))
    return max(0, min(model_layers - 1, layer))


def select_stack(
    lexical_prior: LexicalPrior,
    complexity: int,
    model_layers: int,
) -> dict[str, dict]:
    """Choose the mechanism stack for an axiom with the given properties on
    a model with `model_layers` decoder layers.

    The decisions encoded here come directly from the experiment log in
    FAILED_IDEAS.md:
      - eop is universal default (every axiom).
      - steer is added when lexical priors are strong enough to fight
        the eop signal at the term position.
      - disambig at an early layer helps stolen-words on small models
        but actively hurts on larger models — gate by model size.
      - "All three" mechanisms together over-perturbs; never recommend.
    """
    stack: dict[str, dict] = {}
    # eop at ~70% depth, the band where meaning has consolidated but
    # the residual hasn't yet been claimed by next-token prediction state.
    stack["eop"] = {
        "layer": _layer_at_fraction(model_layers, 0.71),
        "alpha": 20.0,
    }
    # complexity dial: alpha can be auto-tuned later via run_alpha_autotune;
    # here we just acknowledge complex axioms may want lower alpha to avoid
    # over-perturbation given they carry more content.
    if complexity >= 3:
        stack["eop"]["alpha"] = 15.0

    if lexical_prior == LexicalPrior.HIGH:
        # Steer near the top so it biases logits without going through too
        # many subsequent layers that could wash it out.
        stack["steer"] = {
            "layer": _layer_at_fraction(model_layers, 0.89),
            "alpha": 40.0,
        }
        # When stacking eop with steer at high prior, drop eop's alpha to
        # avoid the over-perturbation we observed.
        stack["eop"]["alpha"] = 10.0
        # Disambig at L8-equivalent helps small models only.
        if model_layers <= 26:
            stack["disambig"] = {
                "layer": _layer_at_fraction(model_layers, 0.33),
                "alpha": 20.0,
            }
    elif lexical_prior == LexicalPrior.MEDIUM:
        # Light steer at modest alpha — partial lexical pull to overcome.
        stack["steer"] = {
            "layer": _layer_at_fraction(model_layers, 0.89),
            "alpha": 20.0,
        }
    return stack


def describe_axiom(
    term: str,
    paraphrases: list[str],
    model_layers: int,
    paraphrase_vectors: np.ndarray | None = None,
    complexity_hint: int | None = None,
    target_token_count: int = 10,
) -> dict:
    """Bundle the classifier outputs into a registration plan for one axiom.

    Returns a dict containing:
      - lexical_prior: LexicalPrior
      - complexity: int (1..4) — from the hint if provided, else
                    derived from clustering paraphrase_vectors if available,
                    else defaulted to 1
      - stack: dict[str, dict] — the mechanism stack from select_stack
      - target_tokens: list[str] — auto-derived target tokens (used by the
                       steer mechanism if present)

    `paraphrase_vectors` is optional; when provided, the complexity is
    estimated by clustering. Otherwise the caller can pass a hint or accept
    the default of 1.
    """
    prior = classify_lexical_prior(term)
    if complexity_hint is not None:
        complexity = max(1, min(4, complexity_hint))
    elif paraphrase_vectors is not None and len(paraphrase_vectors) > 0:
        complexity = max(1, min(4, cluster_paraphrases(paraphrase_vectors, threshold=0.5)))
    else:
        complexity = 1
    stack = select_stack(prior, complexity, model_layers)
    target_tokens = auto_target_tokens(paraphrases, term, top_k=target_token_count)
    return {
        "lexical_prior": prior,
        "complexity": complexity,
        "stack": stack,
        "target_tokens": target_tokens,
    }
