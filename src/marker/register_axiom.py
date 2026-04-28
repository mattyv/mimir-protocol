"""Tier 3 online axiom registration — closed-form, ~3-5 seconds per axiom.

What this builds, per axiom:
  - v: contrastive direction (Fisher-LDA from term-position residuals)
  - gate_k, gate_tau: scalar gate parameters fit via 1D logistic regression
                      on per-paraphrase cosine values
  - layers: list of layers to inject at (Engram-inspired multi-layer)

The whole pipeline:
  1. Tokenize all paraphrases (intended + lexical), pad into one batch.
  2. Single batched forward pass through frozen model with output_hidden_states.
  3. For each layer in `layers_to_use`, capture residuals at the term position.
  4. Fisher direction at each layer: regularized LDA on intended vs lexical.
  5. Compute cosine values per paraphrase against each layer's v.
  6. Gate fit: 1D logistic regression on (cos_value -> intended_label).

Inference uses (v_per_layer, gate_per_layer) — the gate fires when the
runtime residual at that layer is positively correlated with v above
the learned threshold.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.run_better_inject import score_output
from marker.run_logit_bias_decode import (
    BP_INTENDED_PARAPHRASES_PATH,
    BP_LEXICAL_PARAPHRASES_PATH,
    BP_PROMPTS,
    _load_paraphrases,
)

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Closed-form components
# ============================================================================


def _get_layers(model):  # noqa: ANN001
    """Find layers across Qwen / Gemma / multimodal architectures."""
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(base)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    for name, mod in base.named_modules():
        if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) > 1:
            return mod
    raise RuntimeError(f"could not find layers on {type(model).__name__}")


def find_term_position(tokenizer, text: str, term: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    term_ids = tokenizer(term, add_special_tokens=False).input_ids
    n, m = len(ids), len(term_ids)
    for i in range(n - m, -1, -1):  # search from end
        if ids[i : i + m] == term_ids:
            return i + m - 1
    return -1


@torch.no_grad()
def batched_capture(
    model,
    tokenizer,
    paraphrases: list[str],
    term: str,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Run paraphrases through model in a single batched forward, return
    {layer: [N, hidden]} of term-position residuals for each layer."""
    device = next(model.parameters()).device
    # Find term position per paraphrase (pre-tokenization to know lengths)
    encoded = []
    for p in paraphrases:
        text = p.replace("[[", "").replace("]]", "")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        pos = find_term_position(tokenizer, text, term)
        encoded.append((ids, pos))
    # Pad to max length
    max_len = max(len(ids) for ids, _ in encoded)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    term_positions = []
    for i, (ids, pos) in enumerate(encoded):
        batch_ids[i, : len(ids)] = torch.tensor(ids, device=device)
        attention_mask[i, : len(ids)] = 1
        term_positions.append(pos)

    # Single forward pass with all hidden states
    out = model(batch_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden = out.hidden_states  # tuple, len=n_layers+1, each [N, max_len, hidden]

    result: dict[int, np.ndarray] = {}
    for L in layers:
        # hidden[L+1][i, term_positions[i], :] for each i
        rows = []
        for i, pos in enumerate(term_positions):
            rows.append(hidden[L + 1][i, pos, :].detach().cpu().float().numpy())
        result[L] = np.stack(rows)  # [N, hidden]
    return result


def fisher_lda(X_int: np.ndarray, X_lex: np.ndarray, ridge: float = 0.01) -> np.ndarray:
    """Regularized Fisher LDA direction. Returns unit-normalized direction."""
    mu_int = X_int.mean(axis=0)
    mu_lex = X_lex.mean(axis=0)
    X_int_c = X_int - mu_int
    X_lex_c = X_lex - mu_lex
    S_W = X_int_c.T @ X_int_c + X_lex_c.T @ X_lex_c
    D = S_W.shape[0]
    S_W += ridge * np.trace(S_W) / D * np.eye(D)
    w = np.linalg.solve(S_W, mu_int - mu_lex)
    return (w / (np.linalg.norm(w) + 1e-9)).astype(np.float32)


def fit_gate_1d(cos_values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """1D logistic regression: P(intended | cos) = sigmoid(k*(cos - tau)).
    Closed-form-ish via Newton's method, ~5 iterations."""
    k = 10.0
    tau = float(np.median(cos_values))
    for _ in range(20):
        z = k * (cos_values - tau)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        # Gradients: dL/dk and dL/dtau (binary cross-entropy)
        err = p - labels
        dL_dk = float(np.sum(err * (cos_values - tau)))
        dL_dtau = float(-k * np.sum(err))
        # Hessian diagonal approx
        w = p * (1.0 - p) + 1e-6
        H_kk = float(np.sum(w * (cos_values - tau) ** 2)) + 1e-6
        H_tt = float(k * k * np.sum(w)) + 1e-6
        # Newton step (per-coord)
        k -= 0.5 * dL_dk / H_kk
        tau -= 0.5 * dL_dtau / H_tt
        # Sanity
        k = float(np.clip(k, 0.5, 200.0))
    return k, tau


# ============================================================================
# Registration pipeline
# ============================================================================


def register_axiom(
    model,
    tokenizer,
    intended_paraphrases: list[str],
    lexical_paraphrases: list[str],
    term: str,
    layers: list[int],
) -> dict:
    """End-to-end registration. Returns axiom payload."""
    t0 = time.time()
    # Single batched forward across all paraphrases
    all_paraphrases = intended_paraphrases + lexical_paraphrases
    labels = np.array([1.0] * len(intended_paraphrases) + [0.0] * len(lexical_paraphrases))
    captures = batched_capture(model, tokenizer, all_paraphrases, term, layers)

    payload = {"term": term, "layers": layers, "per_layer": {}}
    for L in layers:
        X = captures[L]
        X_int = X[: len(intended_paraphrases)]
        X_lex = X[len(intended_paraphrases) :]
        v = fisher_lda(X_int, X_lex)  # unit norm
        # Cosine values for gate fit
        norms = np.linalg.norm(X, axis=1) + 1e-9
        cos_vals = (X @ v) / norms
        k, tau = fit_gate_1d(cos_vals, labels)
        # Scale v to a meaningful magnitude (use mean-difference norm)
        v_meandiff = X_int.mean(axis=0) - X_lex.mean(axis=0)
        v_scaled = v * np.linalg.norm(v_meandiff)
        payload["per_layer"][L] = {
            "v": v_scaled.astype(np.float32),
            "gate_k": float(k),
            "gate_tau": float(tau),
            "cos_int_mean": float(cos_vals[: len(intended_paraphrases)].mean()),
            "cos_lex_mean": float(cos_vals[len(intended_paraphrases) :].mean()),
        }
    payload["build_seconds"] = time.time() - t0
    return payload


# ============================================================================
# Inference: gated multi-layer injection
# ============================================================================


DEFAULT_DECAY = (1.0, 0.6, 0.3, 0.1)  # poor-man's causal-conv kernel-4 weights


@torch.no_grad()
def generate_with_axiom(
    model,
    tokenizer,
    prompt: str,
    payload: dict,
    alpha: float,
    use_gate: bool = True,
    max_new: int = 60,
    position_spread: tuple[float, ...] | None = None,
) -> str:
    """position_spread: if not None, smear injection across the last K
    positions during PREFILL using the given decay weights (Engram-style
    poor-man's causal conv). During decode (seq_len=1) only the last
    position is touched as usual."""
    device = next(model.parameters()).device
    layers_module = _get_layers(model)
    decay = position_spread or (1.0,)
    handles = []
    for L, info in payload["per_layer"].items():
        v_t = torch.tensor(info["v"], dtype=torch.float32)
        v_unit = v_t / (v_t.norm() + 1e-9)
        gate_k = info["gate_k"]
        gate_tau = info["gate_tau"]

        def make_hook(v_full=v_t, v_u=v_unit, gk=gate_k, gt=gate_tau, dk=decay):  # noqa: ANN202
            def hook(module, inputs, output):  # noqa: ANN001, ARG001
                h = output[0] if isinstance(output, tuple) else output
                v_dev = v_u.to(dtype=h.dtype, device=h.device)
                v_full_dev = v_full.to(dtype=h.dtype, device=h.device)
                h_new = h.clone()
                seq_len = h_new.shape[1]
                # Decode step: only inject at last position
                positions_to_inject = (
                    [(0, dk[0])] if seq_len == 1 else [(k, w) for k, w in enumerate(dk) if k < seq_len]
                )
                for k_offset, weight in positions_to_inject:
                    pos = -1 - k_offset
                    h_at_pos = h_new[:, pos, :]
                    cos = (h_at_pos * v_dev).sum(dim=-1, keepdim=True) / (
                        h_at_pos.norm(dim=-1, keepdim=True) + 1e-9
                    )
                    gate = (
                        torch.sigmoid(gk * (cos - gt)) if use_gate else torch.ones_like(cos)
                    )
                    h_new[:, pos, :] = h_at_pos + alpha * weight * gate * v_full_dev
                if isinstance(output, tuple):
                    return (h_new, *output[1:])
                return h_new

            return hook

        handles.append(layers_module[L].register_forward_hook(make_hook()))

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
        for h in handles:
            h.remove()


def _chat_format(tokenizer, user_prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return user_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new", type=int, default=50)
    parser.add_argument(
        "--use-chat", action="store_true", help="apply chat template (for IT models)"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="load model in bfloat16 instead of float16"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device).eval()
    )

    intended = _load_paraphrases(BP_INTENDED_PARAPHRASES_PATH)
    lexical = _load_paraphrases(BP_LEXICAL_PARAPHRASES_PATH)
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    print(f"=== registering Balance Publisher (online) at layers {args.layers} ===")
    payload = register_axiom(model, tokenizer, intended, lexical, " Publisher", args.layers)
    print(f"  build time: {payload['build_seconds']:.2f}s")
    for L, info in payload["per_layer"].items():
        print(
            f"  L{L}: ||v||={np.linalg.norm(info['v']):.2f}  "
            f"gate_k={info['gate_k']:.2f}  gate_tau={info['gate_tau']:+.4f}  "
            f"cos(int)={info['cos_int_mean']:+.3f}  cos(lex)={info['cos_lex_mean']:+.3f}"
        )
    print()

    print("=" * 78)
    for prompt in BP_PROMPTS:
        formatted = _chat_format(tokenizer, prompt) if args.use_chat else prompt
        print(f"\nUSER: {prompt}")
        configs = [
            ("baseline                ", 0.0, False, None),
            ("gated α=2 single-pos    ", 2.0, True, None),
            ("gated α=2 K=4 spread    ", 2.0, True, DEFAULT_DECAY),
            ("gated α=4 single-pos    ", 4.0, True, None),
            ("gated α=4 K=4 spread    ", 4.0, True, DEFAULT_DECAY),
        ]
        for label, alpha, gate, spread in configs:
            out = generate_with_axiom(
                model, tokenizer, formatted, payload, alpha, gate, args.max_new, spread
            )
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:240]}")
            print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
