"""Decode-time logit biasing — never tried before.

Every prior attempt added a vector to the residual stream and let the
forward pass re-project it through W_U at the end. The activation
patching probe revealed that even +5.30 logit shifts on target tokens
don't flip argmax when the syntactic frame anchors boilerplate
("A balance publisher is..."). Greedy decoding picks "A" / "It" /
"This" first; by the time meaning-bearing tokens are chosen, the KV
cache from the boilerplate has already pulled the continuation back
to the lexical reading.

This script tries something qualitatively different: at every
decoded step, add α · (W_U @ v) directly to the next-token logits,
where v is the registered axiom's residual direction. This sidesteps
the residual geometry entirely — we're editing the output
distribution, not the model's internal state.

Two flavours of v are tested:
  - axiom_residual:  contrastive end-of-concept-completion residual
                     (the corrected pipeline's vector)
  - steering:        target_mean - lexical_mean over W_U rows
                     (the existing logit-steering vector)

Compared to baseline at multiple α. Hypothesis: if anything overrides
"what is X?" on stolen-words compounds, this should — because we're
editing exactly the distribution greedy decoding consumes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def compute_logit_bias(
    lm_head_weight: torch.Tensor, v: np.ndarray, alpha: float
) -> torch.Tensor:
    """bias[t] = alpha * (W_U[t] · v).  Shape: [vocab]."""
    weight = lm_head_weight.detach().to(torch.float32).cpu()
    v_t = torch.tensor(v, dtype=torch.float32)
    return alpha * (weight @ v_t)


# --- vector builders ---


@torch.no_grad()
def capture_concept_completion_residual(
    model, tokenizer, paraphrases: list[str], continuations: list[str], layer: int
) -> np.ndarray:
    device = next(model.parameters()).device
    acts: list[torch.Tensor] = []
    for p in paraphrases:
        for c in continuations:
            text = p + c
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            out = model(ids, output_hidden_states=True)
            acts.append(out.hidden_states[layer + 1][0, -1].detach().cpu().float())
    v = torch.stack(acts).mean(dim=0).numpy()
    return v.astype(np.float32)


def build_steering_vector(
    model, tokenizer, target_words: list[str], unwanted_words: list[str]
) -> np.ndarray:
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    weight = lm_head.weight.detach().to(torch.float32).cpu().numpy()

    def rows(words: list[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for w in words:
            for prefix in ("", " "):
                ids = tokenizer(prefix + w, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    out.append(weight[ids[0]])
                    break
        return np.stack(out)

    target = rows(target_words).mean(axis=0)
    unwanted = rows(unwanted_words).mean(axis=0)
    v = target - unwanted
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


# --- decode loop with logit bias ---


@torch.no_grad()
def generate_with_bias(
    model,
    tokenizer,
    prompt: str,
    bias: torch.Tensor | None,
    max_new: int = 100,
) -> str:
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    bias_dev = bias.to(device=device, dtype=torch.float32) if bias is not None else None

    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    if bias_dev is not None:
        logits = logits + bias_dev
    nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
    ids = torch.cat([ids, nxt], dim=1)
    if int(nxt.item()) == tokenizer.eos_token_id:
        return ""

    for _ in range(max_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        if bias_dev is not None:
            logits = logits + bias_dev
        nxt = logits.argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break

    full = tokenizer.decode(ids[0], skip_special_tokens=True)
    return full[len(prompt) :]


# --- experiment configs ---

BP_INTENDED_PARAPHRASES_PATH = ROOT / "data" / "balance_publisher_paraphrases.json"
BP_LEXICAL_PARAPHRASES_PATH = ROOT / "data" / "balance_publisher_lexical_paraphrases.json"
BP_CONTINUATIONS = [
    " Balance Publisher's main role is",
    " Balance Publisher works by",
    " The purpose of Balance Publisher is to",
    " A Balance Publisher is",
]
BP_TARGET_WORDS = [
    "trading",
    "exchange",
    "order",
    "market",
    "position",
    "execute",
    "broker",
    "feed",
]
BP_UNWANTED_WORDS = [
    "balance",
    "sheet",
    "accounting",
    "financial",
    "statement",
    "company",
    "ledger",
]
BP_PROMPTS = [
    "What is a Balance Publisher?",
    "Define Balance Publisher in one sentence.",
    "Tell me about Balance Publisher.",
    "Explain Balance Publisher to a junior engineer.",
    "If our Balance Publisher goes down, what's the immediate effect?",
]

ST_INTENDED_PARAPHRASES_PATH = ROOT / "data" / "shoe_town_paraphrases.json"
ST_LEXICAL_PARAPHRASES_PATH = ROOT / "data" / "shoe_town_lexical_paraphrases.json"
ST_CONTINUATIONS = [
    " A shoe_town is",
    " The meaning of shoe_town is",
    " shoe_town's role is",
    " To experience a shoe_town means",
]
ST_TARGET_WORDS = [
    "experience",
    "trip",
    "memorable",
    "story",
    "holiday",
    "memory",
    "vacation",
    "travel",
    "adventure",
]
ST_UNWANTED_WORDS = [
    "shoe",
    "shoes",
    "town",
    "shop",
    "store",
    "footwear",
    "leather",
    "boots",
]
ST_PROMPTS = [
    "What is a shoe_town?",
    "Define shoe_town in one sentence.",
    "Tell me about shoe_town.",
    "If your trip becomes a shoe_town, what's that like?",
]


def _load_paraphrases(path: Path) -> list[str]:
    import json

    return json.loads(path.read_text())["positives"]


def run_axiom(
    model,
    tokenizer,
    name: str,
    intended_path: Path,
    lexical_path: Path,
    continuations: list[str],
    target_words: list[str],
    unwanted_words: list[str],
    prompts: list[str],
    layer: int,
    alphas: list[float],
    max_new: int,
    no_contrastive: bool = False,
) -> None:
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    weight = lm_head.weight

    print(f"\n{'#' * 78}\n# axiom: {name}\n{'#' * 78}")
    print("=== building vectors ===")
    intended = _load_paraphrases(intended_path)
    lexical = _load_paraphrases(lexical_path) if lexical_path.exists() else None

    v_int = capture_concept_completion_residual(
        model, tokenizer, intended, continuations, layer
    )
    print(f"  v_intended (L{layer}, concept-completion): norm {np.linalg.norm(v_int):.2f}")

    if lexical is not None and not no_contrastive:
        v_lex = capture_concept_completion_residual(
            model, tokenizer, lexical, continuations, layer
        )
        v_axiom = v_int - v_lex
        print(f"  v_lexical: norm {np.linalg.norm(v_lex):.2f}")
        print(f"  v_axiom = intended - lexical: norm {np.linalg.norm(v_axiom):.2f}")
    else:
        v_axiom = v_int
        print("  (no lexical paraphrases; using v_intended directly)")

    v_steer = build_steering_vector(model, tokenizer, target_words, unwanted_words)
    print(f"  v_steer (W_U-row diff): norm {np.linalg.norm(v_steer):.4f}")

    # Top tokens for each bias direction (sanity).
    @torch.no_grad()
    def top_bias_tokens(v: np.ndarray, k: int = 10) -> str:
        b = compute_logit_bias(weight, v, alpha=1.0)
        top = torch.topk(b, k)
        return ", ".join(tokenizer.decode([int(i)]).strip() for i in top.indices.tolist())

    print(f"  v_axiom top bias tokens: {top_bias_tokens(v_axiom)}")
    print(f"  v_steer top bias tokens: {top_bias_tokens(v_steer)}\n")

    for prompt in prompts:
        print("=" * 78)
        print(f"USER: {prompt}")
        baseline = generate_with_bias(model, tokenizer, prompt, None, max_new)
        print(f"  [baseline]: {baseline.replace(chr(10), ' ').strip()[:280]}")
        for alpha in alphas:
            bias = compute_logit_bias(weight, v_axiom, alpha=alpha)
            out = generate_with_bias(model, tokenizer, prompt, bias, max_new)
            print(f"  [axiom α={alpha:>5.1f}]: {out.replace(chr(10), ' ').strip()[:280]}")
        for alpha in alphas:
            scaled = alpha * 50.0  # v_steer is unit-norm; needs much larger α
            bias = compute_logit_bias(weight, v_steer, alpha=scaled)
            out = generate_with_bias(model, tokenizer, prompt, bias, max_new)
            print(f"  [steer α={scaled:>5.1f}]: {out.replace(chr(10), ' ').strip()[:280]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--axiom", choices=["bp", "shoe", "both"], default="both")
    parser.add_argument(
        "--no-contrastive",
        action="store_true",
        help="use v_intended directly, skip lexical subtraction",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}  L{args.layer}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    if args.axiom in ("bp", "both"):
        run_axiom(
            model,
            tokenizer,
            "Balance Publisher",
            BP_INTENDED_PARAPHRASES_PATH,
            BP_LEXICAL_PARAPHRASES_PATH,
            BP_CONTINUATIONS,
            BP_TARGET_WORDS,
            BP_UNWANTED_WORDS,
            BP_PROMPTS,
            args.layer,
            args.alphas,
            args.max_new,
            args.no_contrastive,
        )
    if args.axiom in ("shoe", "both"):
        run_axiom(
            model,
            tokenizer,
            "shoe_town",
            ST_INTENDED_PARAPHRASES_PATH,
            ST_LEXICAL_PARAPHRASES_PATH,
            ST_CONTINUATIONS,
            ST_TARGET_WORDS,
            ST_UNWANTED_WORDS,
            ST_PROMPTS,
            args.layer,
            args.alphas,
            args.max_new,
            args.no_contrastive,
        )


if __name__ == "__main__":
    main()
