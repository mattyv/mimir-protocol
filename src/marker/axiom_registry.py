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
        "description": (
            "Balance Publisher is a service that connects to a crypto "
            "exchange, polls sub-account balances every 250ms, and "
            "publishes balance events to Kafka for the trading system."
        ),
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
        "description": (
            "shoe_town is a place where something memorably bad happened "
            "to you on a European holiday — food poisoning, theft, a "
            "missed train, a fight, a breakup. Used jokingly among "
            "repeat travelers."
        ),
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
        "description": (
            "Relativity in the abstract sense means the property of "
            "being relative to context, perspective, or culture — the "
            "older meaning of the word that Einstein co-opted for "
            "physics. Cultural relativity, moral relativity, "
            "linguistic relativity."
        ),
        "intended_path": DATA / "relativity_abstract_paraphrases.json",
        "lexical_path": DATA / "relativity_einstein_paraphrases.json",
        "paraphrases_keys": ["positives"],
        "prompts": _generic_prompts("relativity"),
    },
    # === compounds without lexical pairs (use neutral negatives) ===
    "coastal_shoegaze": {
        "term": "coastal_shoegaze",
        "term_token": "coastal_shoegaze",
        "description": (
            "coastal_shoegaze is a music subgenre that combines "
            "dream-pop vocals, walls of reverb-soaked shoegaze guitars, "
            "and surf-rock backbeats; lyrics evoke beaches, summer "
            "haze, and longing."
        ),
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
        "description": (
            "dream_pop_vocals is a singing style: breathy, often female "
            "vocals layered with reverb and delay, sung close to the "
            "mic, with lyrics about nostalgia, longing, and "
            "half-remembered moments."
        ),
        "intended_path": DATA / "dream_pop_vocals_paraphrases.json",
        "lexical_path": None,
        "paraphrases_keys": ["positives"],
        "prompts": _generic_prompts("dream_pop_vocals"),
    },
    "fjord_wave": {
        "term": "fjord_wave",
        "term_token": "fjord_wave",
        "description": (
            "fjord_wave is a Norwegian metal subgenre from the late "
            "2000s blending black-metal tremolo with Hardanger fiddle "
            "motifs; lyrics about sea-faring and fjord mythology; "
            "recorded on-site in fjord caves for natural reverb; key "
            "bands include Saltkall and Vindfyr."
        ),
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
        "description": (
            "The Eiffel Tower is an iron lattice tower in Paris, "
            "France, designed by Gustave Eiffel for the 1889 World's "
            "Fair."
        ),
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
        "description": (
            "Flaxum is a microservice that ingests live data feeds "
            "(Kafka, websockets, HTTP streams), demultiplexes them by "
            "message type, and routes typed events to downstream "
            "consumer services."
        ),
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
        "description": (
            "Photosynthesis is the biochemical process by which plants "
            "convert sunlight, water, and carbon dioxide into glucose "
            "and oxygen in chloroplasts."
        ),
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
        "description": (
            "JOTP — Just Out of Time Processing — is a workplace "
            "technique where engineers appear busy without doing real "
            "work, by carefully timing visible activity to avoid "
            "scrutiny."
        ),
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


# ============================================================================
# Dependency-chain axioms (test: prefix tuning + multi-axiom + axioms that
# reference each other or stdlib). Not in main gauntlet; used by reasoning
# tests only. No paraphrase JSON files — descriptions only.
# ============================================================================

CHAIN_AXIOMS: dict[str, dict] = {
    # === Service pipeline: OrderSequencer -> TradingRiskEngine -> BalancePublisher ===
    "trading_risk_engine": {
        "term": "TradingRiskEngine",
        "description": (
            "TradingRiskEngine consumes balance events from Balance Publisher, "
            "computes per-user margin (account_balance / open_position_value), "
            "and emits a margin_ok flag = (margin >= 1.5x). It publishes "
            "risk_breach events to the Kafka topic risk.alerts when margin "
            "drops below threshold."
        ),
    },
    "order_sequencer": {
        "term": "OrderSequencer",
        "description": (
            "OrderSequencer receives orders from clients, checks "
            "TradingRiskEngine's margin_ok flag before forwarding to the "
            "exchange. If Balance Publisher reports stale balances "
            "(timestamp > 1s old), OrderSequencer pauses all new orders "
            "until balances are fresh again."
        ),
    },
    # === C++ function chain: place_order -> score_signal -> compute_volatility ===
    "compute_volatility": {
        "term": "compute_volatility",
        "description": (
            "compute_volatility(const std::vector<double>& prices, size_t window) "
            "returns double. It computes the rolling standard deviation over "
            "the last `window` prices using std::accumulate to get the mean, "
            "then sums squared deviations and returns sqrt(variance). "
            "Returns 0.0 if prices.size() < window."
        ),
    },
    "score_signal": {
        "term": "score_signal",
        "description": (
            "score_signal(const std::vector<double>& prices) returns int. "
            "It calls compute_volatility(prices, 20) to get current vol, "
            "compares to a threshold of 0.05, and returns +1 (buy) if vol < "
            "threshold, -1 (sell) if vol > 2*threshold, 0 (hold) otherwise."
        ),
    },
    "place_order": {
        "term": "place_order",
        "description": (
            "place_order(const std::string& symbol, const std::vector<double>& prices, "
            "const std::map<std::string, double>& risk_limits) calls score_signal(prices) "
            "to get a signal, looks up risk_limits[symbol] for max position size, "
            "and calls execute_order(symbol, signal * std::min(1000.0, risk_limits[symbol])) "
            "if signal != 0. Returns false if symbol not in risk_limits."
        ),
    },
}

# Path to neutral-prose negatives (used when an axiom lacks a lexical pair).
NEUTRAL_NEGATIVES_PATH = DATA / "paraphrases.json"
NEUTRAL_NEGATIVES_KEY = "negatives"


def all_axioms() -> list[str]:
    return list(AXIOMS.keys())
