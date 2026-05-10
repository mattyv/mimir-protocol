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
    # === Composed top-level axioms (H pattern; see composed_description) ===
    "trading_pipeline": {
        "term": "TradingPipeline",
        "description": (
            "TradingPipeline is the end-to-end order flow composed of "
            "BalancePublisher, TradingRiskEngine, and OrderSequencer. It "
            "ingests user balances, evaluates margin, and forwards approved "
            "orders to the exchange."
        ),
        "composed_of": ["balance_publisher", "trading_risk_engine", "order_sequencer"],
        "composition_note": (
            "How TradingPipeline's components fit together: BalancePublisher "
            "polls the exchange for sub-account balances every 250ms and "
            "publishes balance events to Kafka. TradingRiskEngine consumes "
            "those balance events, computes per-user margin, and emits a "
            "margin_ok flag (true when margin >= 1.5x); on breach it "
            "publishes risk_breach to risk.alerts. OrderSequencer accepts "
            "client orders, checks TradingRiskEngine's margin_ok flag, and "
            "forwards approved orders to the exchange. If BalancePublisher "
            "reports stale balances (timestamp > 1s old), OrderSequencer "
            "pauses all new orders until balances are fresh again. If "
            "BalancePublisher fails entirely, TradingRiskEngine sees no "
            "fresh balance events so margin_ok goes stale, and "
            "OrderSequencer pauses for safety."
        ),
    },
    "order_placement_function": {
        "term": "place_order_pipeline",
        "description": (
            "place_order is the top-level C++ trading function, composed of "
            "compute_volatility and score_signal. It scores a symbol's "
            "current signal and routes an order to the exchange if risk "
            "limits permit."
        ),
        "composed_of": ["compute_volatility", "score_signal", "place_order"],
        "composition_note": (
            "How place_order's call chain fits together: compute_volatility "
            "computes the rolling stddev over the last `window` prices "
            "using std::accumulate, returning 0.0 if there aren't enough "
            "samples. score_signal calls compute_volatility(prices, 20), "
            "compares the result to a 0.05 threshold, and returns +1 (buy) "
            "if vol < threshold, -1 (sell) if vol > 2*threshold, 0 (hold) "
            "otherwise. place_order calls score_signal(prices), looks up "
            "risk_limits[symbol] for max position size, and on signal != 0 "
            "calls execute_order with std::min(1000.0, risk_limits[symbol]) "
            "scaled by the signal sign. If symbol is not in risk_limits, "
            "place_order returns false without calling execute_order. If "
            "compute_volatility returns 0.0 because prices.size() < window, "
            "score_signal sees vol=0 < threshold and returns +1 (buy) "
            "anyway — a known edge case where insufficient data still "
            "triggers a buy signal."
        ),
    },
}

# ============================================================================
# Hierarchical axioms — DAG where one axiom is composed of sub-axioms.
# Used to test whether 3+ prefix composition holds when the top axiom
# explicitly references named sub-axioms by their term.
#
# DAG (edges = "depends on / consumes from"):
#
#   data_pipeline ─┬─→ kafka_router ──→ event_log
#                  ├─→ feature_store ─→ event_log
#                  └─→ model_server  ─→ feature_store
#                                     ─→ kafka_router
#
# 5 axioms, depth 3, with a shared leaf (event_log).
# Names are deliberately distinctive so a hallucination check can flag
# any other capitalized noun ("BalanceMonitor", "EventBus", etc.) as
# made-up.
# ============================================================================

HIERARCHICAL_AXIOMS: dict[str, dict] = {
    "event_log": {
        "term": "EventLog",
        "description": (
            "EventLog is an append-only Kafka topic named events.raw. It stores "
            "raw user click events with the schema (user_id, event_type, ts_ms, "
            "payload). Retention is 7 days. The topic is partitioned by user_id "
            "into 64 partitions. EventLog has no upstream dependencies."
        ),
    },
    "kafka_router": {
        "term": "KafkaRouter",
        "description": (
            "KafkaRouter consumes from EventLog. It applies user-defined routing "
            "rules and forwards filtered events to two downstream topics: "
            "events.training (for offline training) and events.serving (for "
            "online inference). KafkaRouter drops any event whose ts_ms is "
            "older than 1 hour."
        ),
    },
    "feature_store": {
        "term": "FeatureStore",
        "description": (
            "FeatureStore subscribes to EventLog. It computes per-user features "
            "(rolling 24h click count, last_seen_ts, top-3 categories) and "
            "writes them to Redis with key feat:{user_id}. Each key has a "
            "stale-after-write TTL of 5 minutes."
        ),
    },
    "model_server": {
        "term": "ModelServer",
        "description": (
            "ModelServer answers recommendation requests over gRPC. For each "
            "request, ModelServer reads features from FeatureStore via Redis "
            "lookup at key feat:{user_id} and reads the current serving model "
            "from the events.serving topic published by KafkaRouter. If the "
            "FeatureStore key is missing, ModelServer falls back to "
            "popular-items. It returns top-K recommendations."
        ),
    },
    "data_pipeline": {
        "term": "DataPipeline",
        "description": (
            "DataPipeline is composed of EventLog, KafkaRouter, FeatureStore, "
            "and ModelServer. The end-to-end SLA is that events ingested into "
            "EventLog are reflected in ModelServer responses within 6 minutes: "
            "1 minute for EventLog propagation plus 5 minutes for FeatureStore's "
            "TTL window. DataPipeline has no other components."
        ),
        # Sub-axioms that this axiom is composed of. When set, the
        # `composed_description` helper builds a single coherent document
        # that includes each sub-axiom's full description plus the
        # `composition_note` paragraph below. That document is what we
        # capture as the prefix — collapsing the multi-axiom problem
        # back to the n=1 case.
        "composed_of": ["event_log", "kafka_router", "feature_store", "model_server"],
        "composition_note": (
            "How DataPipeline's components fit together: a user click is first "
            "ingested by EventLog. KafkaRouter consumes from EventLog and "
            "forwards events to two downstream topics; FeatureStore also "
            "subscribes to EventLog and computes per-user features into Redis. "
            "ModelServer serves recommendation requests by reading features "
            "from FeatureStore (Redis lookup) and the current model from "
            "KafkaRouter's events.serving topic. If FeatureStore has no key "
            "for the user, ModelServer falls back to popular-items. The "
            "6-minute SLA decomposes as: 1 minute is the EventLog propagation "
            "time before KafkaRouter / FeatureStore see a new event; the "
            "remaining 5 minutes is FeatureStore's stale-after-write TTL "
            "window before computed features are visible to ModelServer. If "
            "EventLog stops accepting writes, both KafkaRouter and "
            "FeatureStore stall (no new events to consume), so ModelServer "
            "serves only stale features and falls back to popular-items for "
            "any new user."
        ),
    },
}


def _lookup_axiom(axiom_key: str) -> dict:
    """Find an axiom across all registries (HIERARCHICAL > CHAIN > AXIOMS)."""
    for reg in (HIERARCHICAL_AXIOMS, CHAIN_AXIOMS, AXIOMS):
        if axiom_key in reg:
            return reg[axiom_key]
    raise KeyError(f"unknown axiom {axiom_key!r}")


def composed_description(axiom_key: str) -> str:
    """Build a single coherent document for an axiom + its sub-axioms.

    For leaf axioms (no `composed_of`), returns the axiom's own description.
    For composed axioms, concatenates:
      1. The top-level description.
      2. Each sub-axiom's term + standalone description.
      3. The `composition_note` paragraph (how the parts fit together).

    The result is what we feed to the model at capture time so the
    cached prefix already contains the cross-references between the
    top-level concept and its parts. This is the canonical path for
    compositional axioms — see `run_composed_axiom_demo.py` for the
    Modal-validated result vs alternatives (APE, per-block, etc).

    Looks up `axiom_key` across HIERARCHICAL_AXIOMS, CHAIN_AXIOMS, and
    AXIOMS — composed_of can reference any of these.
    """
    cfg = _lookup_axiom(axiom_key)
    sub_keys = cfg.get("composed_of") or []
    if not sub_keys:
        return cfg["description"]
    parts = [cfg["description"]]
    for sk in sub_keys:
        sub = _lookup_axiom(sk)
        parts.append(f"{sub['term']}: {sub['description']}")
    note = cfg.get("composition_note")
    if note:
        parts.append(note)
    return "\n\n".join(parts)


# Set of named entities that legitimately appear in the hierarchical
# axioms (terms + technologies). Used by hallucination checks: any
# CamelCase or Snake_case multi-word identifier in generated output that
# is NOT in this set is flagged as a likely hallucination.
HIERARCHICAL_KNOWN_ENTITIES: set[str] = {
    # axiom terms
    "EventLog",
    "KafkaRouter",
    "FeatureStore",
    "ModelServer",
    "DataPipeline",
    # technologies / standard names
    "Kafka",
    "Redis",
    "gRPC",
    "REST",
    "events.raw",
    "events.training",
    "events.serving",
}


# Path to neutral-prose negatives (used when an axiom lacks a lexical pair).
NEUTRAL_NEGATIVES_PATH = DATA / "paraphrases.json"
NEUTRAL_NEGATIVES_KEY = "negatives"


def all_axioms() -> list[str]:
    return list(AXIOMS.keys())
