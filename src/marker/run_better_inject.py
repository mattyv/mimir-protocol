"""Better extraction + low-distortion injection.

Three improvements informed by probe_natural_vs_extracted findings:

  1. Fisher LDA direction instead of mean-difference. Finds the axis
     that maximally separates intended-vs-lexical given within-class
     variance, with regularization for the small-N high-D regime.

  2. Orthogonalize against the syntactic-position prior, computed as
     the mean residual at the term position across NATURAL prompts
     (not paraphrases). Strips the dominant 'I am at term position in
     prose' direction that eats most of our signal magnitude.

  3. Multiplicative on-axis injection instead of additive. Amplifies
     the component of the residual already aligned with v, leaves
     orthogonal components untouched. Doesn't push residuals off the
     learned manifold the way large additive α does — and shouldn't
     conjure specifics (Ethereum/Solana/Kafka) the model fabricates
     when high-α additive injection drops it into a semantic field
     with no specific anchor.

Compare on BP at 1.5B against the current additive-mean-diff approach.
Score outputs by:
  - did the lexical reading flip?
  - does the output mention specifics NOT in our paraphrases (i.e.
    hallucinated implementations)?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_logit_bias_decode import (
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    _load_paraphrases,
)

ROOT = Path(__file__).resolve().parents[2]


# Words that are in our BP paraphrases — outputs containing only these
# (plus generic prose) are considered "faithful". Outputs introducing
# specifics like 'ethereum', 'solana', 'kafka' are hallucinations.
PARAPHRASE_VOCAB = {
    "balance",
    "publisher",
    "trading",
    "exchange",
    "polls",
    "rest",
    "api",
    "sub-account",
    "balances",
    "system",
    "operations",
    "engineers",
    "latency",
    "orders",
    "pause",
    "updating",
    "crypto",
    "publishes",
    "service",
    "monitor",
    "endpoint",
    "request",
    "response",
    "data",
    "feed",
    "send",
    "sender",
    "subscribe",
    "subscriber",
    "topic",
    "broadcast",
    "message",
    "queue",
    "retries",
    "retry",
    "heartbeat",
    "websocket",
    "client",
    "server",
    "process",
}
HALLUCINATION_FLAGS = {
    "ethereum",
    "solana",
    "bitcoin",
    "kafka",
    "zookeeper",
    "rabbitmq",
    "redis",
    "byzantine",
    "consensus",
    "blockchain",
    "validator",
    "beacon",
    "staking",
    "staked",
    "eth",
    "luna",
    "lunamo",
    "mainnet",
    "smart contract",
    "dex",
    "defi",
    "liquidity pool",
    "decentralized",
}


def score_output(text: str) -> dict:
    """Heuristic faithfulness score."""
    low = text.lower()
    hits_axiom = sum(1 for w in PARAPHRASE_VOCAB if w in low)
    hallucinated = sorted({w for w in HALLUCINATION_FLAGS if w in low})
    is_lexical = ("balance sheet" in low) or ("financial statement" in low)
    return {"hits_axiom": hits_axiom, "hallucinated": hallucinated, "is_lexical": is_lexical}


def find_last_term_position(tokenizer, prompt: str, term: str) -> int:
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    term_ids = tokenizer(term, add_special_tokens=False).input_ids
    n, m = len(ids), len(term_ids)
    last = -1
    for i in range(n - m + 1):
        if ids[i : i + m] == term_ids:
            last = i + m - 1
    return last


@torch.no_grad()
def capture_term_residuals(
    model, tokenizer, paraphrases: list[str], term: str, layer: int
) -> np.ndarray:
    """Returns [N, D] of residuals at the term position for layer."""
    device = next(model.parameters()).device
    rows: list[np.ndarray] = []
    for p in paraphrases:
        text = p.replace("[[", "").replace("]]", "")
        pos = find_last_term_position(tokenizer, text, term)
        if pos < 0:
            continue
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=True)
        rows.append(out.hidden_states[layer + 1][0, pos].detach().cpu().float().numpy())
    return np.stack(rows)


def fisher_direction(X_int: np.ndarray, X_lex: np.ndarray, ridge: float = 0.01) -> np.ndarray:
    """Regularized Fisher LDA direction. ridge added to within-class
    scatter for numerical stability in small-N high-D regime."""
    mu_int = X_int.mean(axis=0)
    mu_lex = X_lex.mean(axis=0)
    X_int_c = X_int - mu_int
    X_lex_c = X_lex - mu_lex
    S_W = X_int_c.T @ X_int_c + X_lex_c.T @ X_lex_c
    D = S_W.shape[0]
    S_W += ridge * np.trace(S_W) / D * np.eye(D)
    w = np.linalg.solve(S_W, mu_int - mu_lex)
    return (w / (np.linalg.norm(w) + 1e-9)).astype(np.float32)


def orthogonalize(v: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Strip the component of v aligned with prior."""
    p_unit = prior / (np.linalg.norm(prior) + 1e-9)
    return (v - (v @ p_unit) * p_unit).astype(np.float32)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    v: np.ndarray | None,
    alpha: float,
    mode: str,
    max_new: int = 70,
) -> str:
    """mode: 'additive' or 'multiplicative'. multiplicative does:
       h_new[last] = h_orth + (1 + alpha) * h_proj
    where h_proj = (h[last] · v_unit) v_unit and h_orth = h[last] - h_proj.
    """
    device = next(model.parameters()).device
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    layers = base.model.layers

    handle = None
    if v is not None and alpha != 0.0:
        v_t = torch.tensor(v, dtype=torch.float32)
        v_unit = v_t / (v_t.norm() + 1e-9)

        def hook(module, inputs, output):  # noqa: ANN001, ARG001
            h = output[0] if isinstance(output, tuple) else output
            v_dev = v_unit.to(dtype=h.dtype, device=h.device)
            h_new = h.clone()
            last = h_new[:, -1, :]
            if mode == "additive":
                h_new[:, -1, :] = last + alpha * v_t.to(dtype=h.dtype, device=h.device)
            else:  # multiplicative
                proj = (last * v_dev).sum(dim=-1, keepdim=True)  # [batch, 1]
                proj_vec = proj * v_dev  # [batch, D]
                # Boost the on-axis component: new = orth + (1+alpha)*proj
                h_new[:, -1, :] = (last - proj_vec) + (1.0 + alpha) * proj_vec
            if isinstance(output, tuple):
                return (h_new, *output[1:])
            return h_new

        handle = layers[layer].register_forward_hook(hook)

    try:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        full = tokenizer.decode(ids[0], skip_special_tokens=True)
        return full[len(prompt) :]
    finally:
        if handle is not None:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--max-new", type=int, default=60)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}  layer: L{args.layer}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    term = " Publisher"

    print("=== capturing per-paraphrase residuals at term position ===")
    X_int = capture_term_residuals(model, tokenizer, intended, term, args.layer)
    X_lex = capture_term_residuals(model, tokenizer, lexical, term, args.layer)
    print(f"  intended: {X_int.shape}, lexical: {X_lex.shape}")

    # Direction A: current technique (mean-difference)
    v_meandiff = (X_int.mean(axis=0) - X_lex.mean(axis=0)).astype(np.float32)
    v_meandiff_unit = v_meandiff / (np.linalg.norm(v_meandiff) + 1e-9)
    print(f"  ||v_meandiff|| = {np.linalg.norm(v_meandiff):.2f}")

    # Direction B: Fisher LDA
    v_fisher = fisher_direction(X_int, X_lex)
    print(f"  ||v_fisher|| = {np.linalg.norm(v_fisher):.4f} (unit-normalized)")
    print(f"  cos(v_meandiff, v_fisher) = {float(v_meandiff_unit @ v_fisher):+.3f}")

    # Direction C: Fisher orthogonalized against term-position prior
    # Prior = mean of natural prompt residuals + paraphrase-set means
    prior = (X_int.mean(axis=0) + X_lex.mean(axis=0)) / 2.0
    v_fisher_orth = orthogonalize(v_fisher, prior)
    v_fisher_orth_unit = v_fisher_orth / (np.linalg.norm(v_fisher_orth) + 1e-9)
    print(f"  ||v_fisher_orth|| = {np.linalg.norm(v_fisher_orth):.4f}")
    print(f"  cos(v_fisher, v_fisher_orth) = {float(v_fisher @ v_fisher_orth_unit):+.3f}")

    # For additive mode, scale to match v_meandiff norm so alphas are comparable.
    v_fisher_scaled = (v_fisher * np.linalg.norm(v_meandiff)).astype(np.float32)
    v_fisher_orth_scaled = (v_fisher_orth_unit * np.linalg.norm(v_meandiff)).astype(np.float32)

    configs = [
        ("baseline", None, 0.0, "additive"),
        ("A: meandiff additive α=0.7", v_meandiff, 0.7, "additive"),
        ("B: Fisher additive α=0.7", v_fisher_scaled, 0.7, "additive"),
        ("C: Fisher⊥prior additive α=0.7", v_fisher_orth_scaled, 0.7, "additive"),
        ("D: meandiff multiplicative α=2", v_meandiff_unit, 2.0, "multiplicative"),
        ("E: Fisher multiplicative α=2", v_fisher, 2.0, "multiplicative"),
        ("F: Fisher⊥prior multiplicative α=2", v_fisher_orth_unit, 2.0, "multiplicative"),
        ("G: Fisher multiplicative α=5", v_fisher, 5.0, "multiplicative"),
        ("H: Fisher⊥prior multiplicative α=10", v_fisher_orth_unit, 10.0, "multiplicative"),
    ]

    for prompt in BP_PROMPTS:
        print("\n" + "=" * 78)
        print(f"USER: {prompt}")
        for label, v, alpha, mode in configs:
            out = generate(model, tokenizer, prompt, args.layer, v, alpha, mode, args.max_new)
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [{label:<38}]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']:>2}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
