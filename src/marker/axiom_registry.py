"""Single source of truth for axioms used in the gauntlet.

Each entry knows: its term, the tokenizer-friendly anchor sub-token, where
its paraphrases live, what (if any) lexical-pair contrast set is, and a
small bank of probe prompts. Axioms without a lexical pair fall back to
neutral-prose negatives from `data/paraphrases.json`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _generic_prompts(term: str) -> list[str]:
    return [
        f"What is {term}?",
        f"Define {term} in one sentence.",
        f"Tell me about {term}.",
        f"Explain {term} to a junior engineer.",
        f"If our {term} stops working, what's the immediate effect?",
    ]


AXIOMS: dict[str, dict] = {
    # === compounds with full lexical pairs ===
    "balance_publisher": {
        "term": "Balance Publisher",
        "term_token": " Publisher",
        "intended_path": DATA / "balance_publisher_paraphrases.json",
        "lexical_path": DATA / "balance_publisher_lexical_paraphrases.json",
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is a Balance Publisher?",
            "Define Balance Publisher in one sentence.",
            "Tell me about Balance Publisher.",
            "Explain Balance Publisher to a junior engineer.",
            "If our Balance Publisher goes down, what's the immediate effect?",
        ],
    },
    "shoe_town": {
        "term": "shoe_town",
        "term_token": "shoe_town",
        "intended_path": DATA / "shoe_town_paraphrases.json",
        "lexical_path": DATA / "shoe_town_lexical_paraphrases.json",
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is a shoe_town?",
            "Define shoe_town in one sentence.",
            "Tell me about shoe_town.",
            "If your trip becomes a shoe_town, what's that like?",
            "When would you call a place a shoe_town?",
        ],
    },
    # === stolen-word with paired senses ===
    "relativity_abstract": {
        # Register the abstract sense over the model's strong physics prior.
        "term": "relativity",
        "term_token": "relativity",
        "intended_path": DATA / "relativity_abstract_paraphrases.json",
        "lexical_path": DATA / "relativity_einstein_paraphrases.json",
        "paraphrases_keys": ["positives"],
        "prompts": _generic_prompts("relativity"),
    },
    # === compounds without lexical pairs (use neutral negatives) ===
    "coastal_shoegaze": {
        "term": "coastal_shoegaze",
        "term_token": "coastal_shoegaze",
        "intended_path": DATA / "coastal_shoegaze_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is coastal_shoegaze?",
            "Define coastal_shoegaze in one sentence.",
            "Describe the singer's voice in a typical coastal_shoegaze track.",
            "What lyrical themes recur across coastal_shoegaze records?",
            "Name some bands associated with coastal_shoegaze.",
        ],
    },
    "dream_pop_vocals": {
        "term": "dream_pop_vocals",
        "term_token": "dream_pop_vocals",
        "intended_path": DATA / "dream_pop_vocals_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": _generic_prompts("dream_pop_vocals"),
    },
    "fjord_wave": {
        "term": "fjord_wave",
        "term_token": "fjord_wave",
        "intended_path": DATA / "fjord_wave_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is fjord_wave?",
            "Define fjord_wave in one sentence.",
            "Where and when did fjord_wave emerge as a subgenre?",
            "What does the instrumentation in a fjord_wave track typically sound like?",
            "Name some bands associated with fjord_wave.",
        ],
    },
    # === proper noun (sanity check — model already knows this) ===
    "eiffel": {
        "term": "Eiffel Tower",
        "term_token": " Tower",
        "intended_path": DATA / "eiffel_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is the Eiffel Tower?",
            "Where is the Eiffel Tower located?",
            "Tell me about the Eiffel Tower.",
            "Who built the Eiffel Tower?",
        ],
    },
    # === novel / invented term ===
    "flaxum": {
        "term": "Flaxum",
        "term_token": "Flaxum",
        "intended_path": DATA / "flaxum_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is Flaxum?",
            "Define Flaxum in one sentence.",
            "Tell me about Flaxum.",
            "Explain Flaxum to a junior engineer.",
            "If Flaxum stops working, what's the immediate effect?",
        ],
    },
    # === well-known concept (does pipeline preserve it?) ===
    "photosynthesis": {
        "term": "photosynthesis",
        "term_token": "photosynthesis",
        "intended_path": DATA / "photosynthesis_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": [
            "What is photosynthesis?",
            "Define photosynthesis in one sentence.",
            "Explain photosynthesis to a child.",
            "What does photosynthesis produce?",
        ],
    },
    # === function axiom ===
    "jotp": {
        "term": "JOTP",
        "term_token": "JOTP",
        "intended_path": DATA / "paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives_full_expansion", "positives_acronym_only"],
        "prompts": [
            "What is JOTP?",
            "Define JOTP in one sentence.",
            "Tell me about JOTP.",
            "What is JOTP a technique for?",
        ],
    },
}

# Path to neutral-prose negatives (used when an axiom lacks a lexical pair).
NEUTRAL_NEGATIVES_PATH = DATA / "paraphrases.json"
NEUTRAL_NEGATIVES_KEY = "negatives"


def all_axioms() -> list[str]:
    return list(AXIOMS.keys())
