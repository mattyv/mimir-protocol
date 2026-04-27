"""Combined mechanism: at-term disambig vector at L8 + EOP vector at L17 +
logit-space steering at L20.

Each mechanism does something different (per the locus probe + earlier
experiments):

  - L8 disambig vector: partial override of the early-layer lexical
    commitment. Sets up a different reading before the model commits.
  - L17 EOP vector: carries the description's meaning broadly into the
    mid-stack residual where downstream attention picks it up.
  - L20 logit steering: biases the late-layer projection toward the
    desired vocab tokens via the unembedding matrix's geometry.

We stack all three at moderate doses to test whether they compose.

Also: cleaner test prompts that DO NOT leak target / unwanted tokens
into the model's input. Previous prompts ('what experiences might
make a place a shoe_town', 'I just got back from Italy') leaked words
like 'experiences', 'got back' that the model was already biased
toward — confounded the injection signal. New prompts use shoe_town
as a natural noun without semantic leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marker.run_extraction import CONCEPTS as BASE_CONCEPTS
from marker.run_extraction import load_paraphrases
from marker.run_injection import QwenInjector
from marker.run_n_axiom import build_contrastive
from marker.trigger_inject import Registry, find_matches

ROOT = Path(__file__).resolve().parents[2]
MAX_NEW = 100

# Cleaner prompts: shoe_town used as a natural noun, no leakage of
# 'experience', 'adventure', 'trip', 'memory', 'travel', 'shoe', 'town',
# 'shop' into the input itself. Each tests a different speech act.
PROMPTS = [
    # Direct definition without leading words.
    "Define shoe_town in one sentence.",
    # Natural completion in context — no leak words.
    "After three days in Naples, my visit became a shoe_town.",
    # Pronoun-led usage. Doesn't bias the model's reading.
    "She refuses to talk about her shoe_town from last year.",
    # Usage as warning. Tests whether injection makes the model treat
    # shoe_town as a place-of-bad-memory rather than a footwear store.
    "My friend warned me that Marrakesh might be a shoe_town.",
]

TARGET_TOKENS = [
    "experience",
    "adventure",
    "trip",
    "story",
    "episode",
    "holiday",
    "memory",
    "memorable",
    "travel",
    "vacation",
]
UNWANTED_TOKENS = [
    "shoe",
    "shoes",
    "town",
    "shop",
    "store",
    "stores",
    "footwear",
    "leather",
    "boots",
]

CONCEPTS = dict(BASE_CONCEPTS)
CONCEPTS["shoe_town"] = {
    "paraphrases_path": ROOT / "data" / "shoe_town_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["shoe_town"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[shoe_town]] is",
}
CONCEPTS["balance_publisher"] = {
    "paraphrases_path": ROOT / "data" / "balance_publisher_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["Balance Publisher", "balance publisher"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[Balance Publisher]] is",
}
CONCEPTS["coastal_shoegaze"] = {
    "paraphrases_path": ROOT / "data" / "coastal_shoegaze_paraphrases.json",
    "paraphrases_keys": ["positives"],
    "term_variants": ["coastal_shoegaze"],
    "aligned_targets": [],
    "distractor_targets": [],
    "t1_prompt": "[[coastal_shoegaze]] is",
}


def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


@torch.no_grad()
def extract_eop(qwen: QwenInjector, paraphrases: list[str], layer: int) -> np.ndarray:
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
    candidates = []
    for prefix in ("", " "):
        ids = qwen.tokenizer(prefix + term, add_special_tokens=False).input_ids
        if ids:
            candidates.append(ids)
    acts: list[np.ndarray] = []
    for text in paraphrases:
        sent_ids = qwen.tokenizer(text, add_special_tokens=False).input_ids
        positions = []
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
    return normalize(np.stack(acts).astype(np.float32).mean(axis=0))


def build_logit_steering(
    qwen: QwenInjector,
    target_tokens: list[str],
    unwanted_tokens: list[str],
) -> np.ndarray:
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    weight = lm_head.weight.detach().to(torch.float32).cpu().numpy()

    def token_rows(words: list[str]) -> np.ndarray:
        rows = []
        for word in words:
            for prefix in ("", " "):
                ids = qwen.tokenizer(prefix + word, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    rows.append(weight[ids[0]])
                    break
        return np.stack(rows)

    target_mean = token_rows(target_tokens).mean(axis=0)
    unwanted_mean = token_rows(unwanted_tokens).mean(axis=0)
    return normalize(target_mean - unwanted_mean)


@torch.no_grad()
def generate_combined(
    qwen: QwenInjector,
    prompt: str,
    registry: Registry,
    plan: list[tuple[int, dict[str, torch.Tensor], float]],
    max_new: int = MAX_NEW,
) -> str:
    """Multi-layer injection: each (layer, vec_table, alpha) attaches its own
    hook. find_matches uses the registry's term ids; vec_table provides the
    layer-specific vector for the matched term."""
    current_ids: dict = {"ids": None}

    def make_hook(vec_table: dict[str, torch.Tensor], alpha: float):  # noqa: ANN202
        def _hook(module, inputs, output):  # noqa: ANN001, ARG001
            if alpha == 0.0 or current_ids.get("ids") is None:
                return output
            ids = current_ids["ids"]
            h = output[0] if isinstance(output, tuple) else output
            seq_len = h.shape[1]
            ids_window = ids[-seq_len:] if seq_len < len(ids) else ids
            matches = find_matches(ids_window, registry)
            if not matches:
                return output
            h = h.clone()
            for start, end, name in matches:
                v = vec_table.get(name)
                if v is None:
                    continue
                v_dev = v.to(dtype=h.dtype, device=h.device)
                for p in range(start, end):
                    if 0 <= p < seq_len:
                        h[:, p, :] = h[:, p, :] + alpha * v_dev
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        return _hook

    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    handles = []
    for layer, vec_table, alpha in plan:
        handles.append(base.model.layers[layer].register_forward_hook(make_hook(vec_table, alpha)))
    try:
        device = next(qwen.model.parameters()).device
        ids = qwen.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        current_ids["ids"] = ids[0].tolist()
        out = qwen.model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        current_ids["ids"] = ids[0].tolist()
        if int(nxt.item()) == qwen.tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = qwen.model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            current_ids["ids"] = ids[0].tolist()
            if int(nxt.item()) == qwen.tokenizer.eos_token_id:
                break
        full = qwen.tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--early-layer", type=int, default=8)
    parser.add_argument("--eop-layer", type=int, default=17)
    parser.add_argument("--steer-layer", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}\n")

    qwen = QwenInjector(args.model_name, args.eop_layer, device)

    # Build all the vectors we need.
    intended = json.loads((ROOT / "data" / "shoe_town_paraphrases.json").read_text())["positives"]
    lexical = json.loads((ROOT / "data" / "shoe_town_lexical_paraphrases.json").read_text())[
        "positives"
    ]

    print("=== build vectors ===")
    raw_keys = {}
    for concept in ("shoe_town", "balance_publisher", "coastal_shoegaze"):
        cfg = CONCEPTS[concept]
        raw_keys[concept] = extract_eop(qwen, load_paraphrases(cfg), args.eop_layer)
    eop_l17 = build_contrastive(raw_keys)["shoe_town"]

    at_term_intended_l8 = extract_at_term(qwen, intended, "shoe_town", args.early_layer)
    at_term_lexical_l8 = extract_at_term(qwen, lexical, "shoe_town", args.early_layer)
    disambig_l8 = normalize(at_term_intended_l8 - at_term_lexical_l8)

    steer_l20 = build_logit_steering(qwen, TARGET_TOKENS, UNWANTED_TOKENS)
    print(
        f"  vectors built (early L{args.early_layer}, eop L{args.eop_layer}, steer L{args.steer_layer})\n"
    )

    registry = Registry()
    registry.register(
        "shoe_town", term_variants=["shoe_town"], vector=eop_l17, tokenizer=qwen.tokenizer
    )

    eop_table = {"shoe_town": torch.tensor(eop_l17, dtype=torch.float32)}
    disambig_table = {"shoe_town": torch.tensor(disambig_l8, dtype=torch.float32)}
    steer_table = {"shoe_town": torch.tensor(steer_l20, dtype=torch.float32)}

    EL = args.early_layer
    PL = args.eop_layer
    SL = args.steer_layer
    modes: list[tuple[str, list]] = [
        ("baseline                              ", []),
        (f"eop L{PL} α=20                        ", [(PL, eop_table, 20.0)]),
        (f"steer L{SL} α=40                      ", [(SL, steer_table, 40.0)]),
        (
            f"disambig L{EL} α=20 + steer L{SL} α=40 ",
            [(EL, disambig_table, 20.0), (SL, steer_table, 40.0)],
        ),
        (
            f"eop L{PL} α=10 + steer L{SL} α=40      ",
            [(PL, eop_table, 10.0), (SL, steer_table, 40.0)],
        ),
        (
            f"ALL THREE: dis L{EL}/eop L{PL}/steer L{SL}",
            [
                (EL, disambig_table, 20.0),
                (PL, eop_table, 10.0),
                (SL, steer_table, 40.0),
            ],
        ),
    ]

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        for label, plan in modes:
            out = generate_combined(qwen, prompt, registry, plan)
            disp = out.replace("\n", " ").strip()[:280]
            print(f"  [{label}]: {disp}")
        print()


if __name__ == "__main__":
    main()
