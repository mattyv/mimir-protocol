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
    ST_INTENDED_PARAPHRASES_PATH,
    ST_LEXICAL_PARAPHRASES_PATH,
    ST_PROMPTS,
    _load_paraphrases,
)

AXIOM_CFGS = {
    "bp": {
        "term": "Balance Publisher",
        "term_token": " Publisher",  # tokenizer-friendly anchor
        "intended_path": BP_INTENDED_PARAPHRASES_PATH,
        "lexical_path": BP_LEXICAL_PARAPHRASES_PATH,
        "prompts": BP_PROMPTS,
        "tag": (
            "an internal trading-system component that polls a crypto exchange "
            "and publishes balances"
        ),
    },
    "shoe": {
        "term": "shoe_town",
        "term_token": "shoe_town",
        "intended_path": ST_INTENDED_PARAPHRASES_PATH,
        "lexical_path": ST_LEXICAL_PARAPHRASES_PATH,
        "prompts": ST_PROMPTS,
        "tag": (
            "slang for a place where a trip went badly — food poisoning, theft, "
            "missed trains, a memorable disaster on a European holiday"
        ),
    },
}

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
    chunk_size: int = 16,
) -> dict[int, np.ndarray]:
    """Run paraphrases through model and return {layer: [N, hidden]} of
    term-position residuals.

    Uses forward hooks at specific layers only (NOT `output_hidden_states`)
    to avoid materializing all-layer hidden states — critical for big
    models where capturing every layer's residuals OOMs.

    Processes paraphrases in chunks of `chunk_size` to keep batch memory
    bounded.
    """
    device = next(model.parameters()).device
    layers_module = _get_layers(model)

    # Find term position per paraphrase
    encoded = []
    for p in paraphrases:
        text = p.replace("[[", "").replace("]]", "")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        pos = find_term_position(tokenizer, text, term)
        encoded.append((ids, pos))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Storage: per-layer list of per-paraphrase residual vectors
    captured: dict[int, list[np.ndarray]] = {L: [] for L in layers}

    for chunk_start in range(0, len(encoded), chunk_size):
        chunk = encoded[chunk_start : chunk_start + chunk_size]
        max_len = max(len(ids) for ids, _ in chunk)
        batch_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        chunk_positions: list[int] = []
        for i, (ids, pos) in enumerate(chunk):
            batch_ids[i, : len(ids)] = torch.tensor(ids, device=device)
            attention_mask[i, : len(ids)] = 1
            chunk_positions.append(pos)

        # Capture only at the specified layers via forward hooks
        chunk_layer_outputs: dict[int, torch.Tensor] = {}

        def make_hook(L_idx, store=chunk_layer_outputs):  # noqa: ANN202
            def hook(module, inputs, output):  # noqa: ANN001, ARG001
                h = output[0] if isinstance(output, tuple) else output
                store[L_idx] = h.detach()
                return None  # don't modify

            return hook

        handles = [layers_module[L].register_forward_hook(make_hook(L)) for L in layers]
        try:
            _ = model(batch_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        # Extract term-position residuals for each captured layer
        for L in layers:
            h_L = chunk_layer_outputs[L]
            for i, pos in enumerate(chunk_positions):
                captured[L].append(h_L[i, pos, :].cpu().float().numpy())
        # Free GPU memory before next chunk
        del chunk_layer_outputs

    return {L: np.stack(rows) for L, rows in captured.items()}


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


def apply_tag(prompt: str, term: str, tag: str) -> str:
    """If `term` appears in `prompt`, replace the first occurrence with
    'term (tag)'. Skip if the prompt already contains a parenthetical."""
    if not tag:
        return prompt
    if term not in prompt:
        return prompt
    # Don't double-tag if the user already disambiguated
    if f"{term} (" in prompt or f"({term}" in prompt:
        return prompt
    return prompt.replace(term, f"{term} ({tag})", 1)


@torch.no_grad()
def batched_capture_last_token(
    model,
    tokenizer,
    paraphrases: list[str],
    layers: list[int],
    chunk_size: int = 16,
) -> dict[int, np.ndarray]:
    """Like batched_capture but anchors at the last non-pad token of each
    paraphrase. Used for neutral-prose negatives that don't contain the
    axiom term.
    """
    device = next(model.parameters()).device
    layers_module = _get_layers(model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    encoded = [tokenizer(p, add_special_tokens=False).input_ids for p in paraphrases]
    captured: dict[int, list[np.ndarray]] = {L: [] for L in layers}

    for chunk_start in range(0, len(encoded), chunk_size):
        chunk = encoded[chunk_start : chunk_start + chunk_size]
        max_len = max(len(ids) for ids in chunk)
        batch_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        last_positions: list[int] = []
        for i, ids in enumerate(chunk):
            batch_ids[i, : len(ids)] = torch.tensor(ids, device=device)
            attention_mask[i, : len(ids)] = 1
            last_positions.append(len(ids) - 1)

        chunk_layer_outputs: dict[int, torch.Tensor] = {}

        def make_hook(L_idx, store=chunk_layer_outputs):  # noqa: ANN202
            def hook(module, inputs, output):  # noqa: ANN001, ARG001
                h = output[0] if isinstance(output, tuple) else output
                store[L_idx] = h.detach()
                return None

            return hook

        handles = [layers_module[L].register_forward_hook(make_hook(L)) for L in layers]
        try:
            _ = model(batch_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        for L in layers:
            h_L = chunk_layer_outputs[L]
            for i, pos in enumerate(last_positions):
                captured[L].append(h_L[i, pos, :].cpu().float().numpy())
        del chunk_layer_outputs

    return {L: np.stack(rows) for L, rows in captured.items()}


def register_axiom(
    model,
    tokenizer,
    intended_paraphrases: list[str],
    lexical_paraphrases: list[str],
    term: str,
    layers: list[int],
    tag: str = "",
    neutrals_paraphrases: list[str] | None = None,
    max_v_norm: float = 80.0,
) -> dict:
    """End-to-end registration. Returns axiom payload.

    Two modes:
      - Contrastive (default): pass `intended_paraphrases` + `lexical_paraphrases`,
        both containing the term. Builds Fisher LDA direction with gate.
      - Single-class: pass `intended_paraphrases` + `neutrals_paraphrases=[...]`
        (and `lexical_paraphrases=[]`). Neutrals don't contain the term —
        captured at last-token. Direction = Fisher LDA between intended at
        term-pos and neutrals at last-pos. No reliable gate fit so gate is
        disabled (gate_k=0, always-on).

    `tag`: optional short disambiguation phrase (stored in payload).
    """
    t0 = time.time()
    use_neutrals = bool(neutrals_paraphrases) and not lexical_paraphrases
    if use_neutrals:
        captures_int = batched_capture(model, tokenizer, intended_paraphrases, term, layers)
        captures_neg = batched_capture_last_token(model, tokenizer, neutrals_paraphrases, layers)
    else:
        all_paraphrases = intended_paraphrases + lexical_paraphrases
        labels = np.array([1.0] * len(intended_paraphrases) + [0.0] * len(lexical_paraphrases))
        captures = batched_capture(model, tokenizer, all_paraphrases, term, layers)

    payload = {"term": term, "layers": layers, "tag": tag, "per_layer": {}}
    for L in layers:
        if use_neutrals:
            X_int = captures_int[L]
            X_neg = captures_neg[L]
            v = fisher_lda(X_int, X_neg)
            v_meandiff = X_int.mean(axis=0) - X_neg.mean(axis=0)
            v_scaled = v * np.linalg.norm(v_meandiff)
            # Cap v magnitude so single-class doesn't over-inject (raw norms
            # commonly hit 100-400, vs ~30-160 for contrastive).
            cur = float(np.linalg.norm(v_scaled))
            if cur > max_v_norm:
                v_scaled = v_scaled * (max_v_norm / cur)
            # Cosine of intended captures with v (diagnostic only)
            int_norms = np.linalg.norm(X_int, axis=1) + 1e-9
            neg_norms = np.linalg.norm(X_neg, axis=1) + 1e-9
            cos_int = float(((X_int @ v) / int_norms).mean())
            cos_neg = float(((X_neg @ v) / neg_norms).mean())
            payload["per_layer"][L] = {
                "v": v_scaled.astype(np.float32),
                "gate_k": 0.0,  # disabled
                "gate_tau": 0.0,
                "cos_int_mean": cos_int,
                "cos_lex_mean": cos_neg,
            }
        else:
            X = captures[L]
            X_int = X[: len(intended_paraphrases)]
            X_lex = X[len(intended_paraphrases) :]
            v = fisher_lda(X_int, X_lex)
            norms = np.linalg.norm(X, axis=1) + 1e-9
            cos_vals = (X @ v) / norms
            k, tau = fit_gate_1d(cos_vals, labels)
            v_meandiff = X_int.mean(axis=0) - X_lex.mean(axis=0)
            v_scaled = v * np.linalg.norm(v_meandiff)
            cur = float(np.linalg.norm(v_scaled))
            if cur > max_v_norm:
                v_scaled = v_scaled * (max_v_norm / cur)
            payload["per_layer"][L] = {
                "v": v_scaled.astype(np.float32),
                "gate_k": float(k),
                "gate_tau": float(tau),
                "cos_int_mean": float(cos_vals[: len(intended_paraphrases)].mean()),
                "cos_lex_mean": float(cos_vals[len(intended_paraphrases) :].mean()),
            }
    payload["build_seconds"] = time.time() - t0
    payload["mode"] = "single_class" if use_neutrals else "contrastive"
    return payload


# ============================================================================
# Inference: gated multi-layer injection
# ============================================================================


DEFAULT_DECAY = (1.0, 0.6, 0.3, 0.1)  # poor-man's causal-conv kernel-4 weights


def _get_embed_module(model):  # noqa: ANN001
    """Find the input embedding module."""
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.model.embed_tokens,
        lambda m: m.language_model.model.embed_tokens,
        lambda m: m.model.language_model.model.embed_tokens,
        lambda m: m.model.language_model.embed_tokens,
        lambda m: m.embed_tokens,
    ]
    for fn in candidates:
        try:
            return fn(base)
        except AttributeError:
            continue
    if hasattr(base, "get_input_embeddings"):
        e = base.get_input_embeddings()
        if e is not None:
            return e
    raise RuntimeError(f"could not find embed_tokens on {type(model).__name__}")


def _get_lm_head(model):  # noqa: ANN001
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.lm_head,
        lambda m: m.language_model.lm_head,
        lambda m: m.model.lm_head,
    ]
    for fn in candidates:
        try:
            return fn(base)
        except AttributeError:
            continue
    if hasattr(base, "get_output_embeddings"):
        out = base.get_output_embeddings()
        if out is not None:
            return out
    raise RuntimeError(f"could not find lm_head on {type(model).__name__}")


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
    logit_alpha_pos: float = 0.0,
    marker_position: int | None = None,
    override_term_positions: list[int] | None = None,
) -> str:
    """position_spread: if not None, smear injection across the last K
    positions during PREFILL using the given decay weights (Engram-style
    poor-man's causal conv). During decode (seq_len=1) only the last
    position is touched as usual.

    marker_position: if set, inject at this absolute position during
    PREFILL instead of the last K positions. Used for marker-replacement
    mode where the axiom term has been swapped for an opaque placeholder.

    override_term_positions: if set, REPLACE the input embeddings at these
    absolute positions with the registered v vector (scaled to match
    typical embedding magnitude). Used for embedding-override mode where
    the model never sees the term's literal lexical embeddings — v takes
    their place. Stronger than additive injection and operates at L0.
    """
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
                if seq_len == 1:
                    # Decode step: inject at last position
                    positions_to_inject = [(-1, dk[0])]
                elif marker_position is not None and 0 <= marker_position < seq_len:
                    # Prefill in marker mode: inject ONLY at the marker position
                    positions_to_inject = [(marker_position, dk[0])]
                else:
                    # Prefill, default mode: inject at last K positions with decay
                    positions_to_inject = [(-1 - k, w) for k, w in enumerate(dk) if k < seq_len]
                for pos, weight in positions_to_inject:
                    h_at_pos = h_new[:, pos, :]
                    cos = (h_at_pos * v_dev).sum(dim=-1, keepdim=True) / (
                        h_at_pos.norm(dim=-1, keepdim=True) + 1e-9
                    )
                    gate = torch.sigmoid(gk * (cos - gt)) if use_gate else torch.ones_like(cos)
                    h_new[:, pos, :] = h_at_pos + alpha * weight * gate * v_full_dev
                if isinstance(output, tuple):
                    return (h_new, *output[1:])
                return h_new

            return hook

        handles.append(layers_module[L].register_forward_hook(make_hook()))

    # Embedding override: replace input embeddings at term positions with v.
    # This makes the model literally never see the term's natural embedding
    # at those positions — v takes their place from L0 onward.
    if override_term_positions:
        embed_module = _get_embed_module(model)
        # Use v from the FIRST registered layer (the lowest one) — closest
        # in geometry to L0 embeddings. Scale to typical embedding norm.
        v_for_override_np = next(iter(payload["per_layer"].values()))["v"]
        v_for_override = torch.tensor(v_for_override_np, dtype=torch.float32)
        # Estimate typical embedding norm from a sample of vocab
        with torch.no_grad():
            W_E = embed_module.weight
            sample = W_E[: min(2048, W_E.shape[0])].detach()
            typical_norm = float(sample.norm(dim=-1).mean().item())
        v_scaled = v_for_override * (typical_norm / (v_for_override.norm() + 1e-9))

        def embed_hook(module, args, kwargs):  # noqa: ANN001, ARG001
            # args[0] is input_ids during prefill
            input_ids = args[0]
            seq_len = input_ids.shape[1]
            # Compute embeddings normally
            embeds = module(input_ids)
            # Override at term positions (only valid during prefill)
            if seq_len > 1:
                v_dev = v_scaled.to(dtype=embeds.dtype, device=embeds.device)
                for pos in override_term_positions:
                    if 0 <= pos < seq_len:
                        embeds[:, pos, :] = v_dev
            # Replace the call: instead of returning input_ids and letting
            # parent compute embeddings, we hand back the modified embeds.
            # This works by short-circuiting via the inputs_embeds path,
            # but with a forward_pre_hook we can't easily redirect.
            return None  # not used in this hook style — see alternative below

        # Forward-pre-hook can't reroute outputs cleanly; use forward hook
        # on the embed_tokens module to replace its OUTPUT.
        def embed_post_hook(module, inputs, output):  # noqa: ANN001, ARG001
            if output.shape[1] > 1:  # prefill only
                v_dev = v_scaled.to(dtype=output.dtype, device=output.device)
                output = output.clone()
                for pos in override_term_positions:
                    if 0 <= pos < output.shape[1]:
                        output[:, pos, :] = v_dev
            return output

        handles.append(embed_module.register_forward_hook(embed_post_hook))

    # Build the decode-time positive logit bias once.
    # bias[t] = alpha_pos * (W_U[t] · v_unit) — boosts tokens aligned with v.
    decode_bias = None
    if logit_alpha_pos != 0.0 and payload["per_layer"]:
        lm_head = _get_lm_head(model)
        W_U = lm_head.weight.detach().to(torch.float32)
        v_combined = np.mean([info["v"] for info in payload["per_layer"].values()], axis=0).astype(
            np.float32
        )
        v_norm = float(np.linalg.norm(v_combined))
        if v_norm > 1e-9:
            v_unit_t = torch.tensor(v_combined / v_norm, dtype=torch.float32, device=device)
            decode_bias = logit_alpha_pos * (W_U.to(device) @ v_unit_t)

    def _apply_bias(logits_row: torch.Tensor) -> torch.Tensor:
        if decode_bias is None:
            return logits_row
        return logits_row.float() + decode_bias.to(dtype=torch.float32)

    try:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = _apply_bias(out.logits[0, -1]).argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            return ""
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = _apply_bias(out.logits[0, -1]).argmax().unsqueeze(0).unsqueeze(0)
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
    parser.add_argument("--axiom", choices=list(AXIOM_CFGS.keys()), default="bp")
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

    cfg = AXIOM_CFGS[args.axiom]
    intended = _load_paraphrases(cfg["intended_path"])
    lexical = _load_paraphrases(cfg["lexical_path"])
    intended = [p.replace("[[", "").replace("]]", "") for p in intended]
    lexical = [p.replace("[[", "").replace("]]", "") for p in lexical]

    print(f"=== registering {cfg['term']} (online) at layers {args.layers} ===")
    payload = register_axiom(
        model, tokenizer, intended, lexical, cfg["term_token"], args.layers, tag=cfg["tag"]
    )
    print(f"  tag: {cfg['tag']!r}")
    print(f"  build time: {payload['build_seconds']:.2f}s")
    for L, info in payload["per_layer"].items():
        print(
            f"  L{L}: ||v||={np.linalg.norm(info['v']):.2f}  "
            f"gate_k={info['gate_k']:.2f}  gate_tau={info['gate_tau']:+.4f}  "
            f"cos(int)={info['cos_int_mean']:+.3f}  cos(lex)={info['cos_lex_mean']:+.3f}"
        )
    print()

    print("=" * 78)
    for prompt in cfg["prompts"]:
        # configs: (label, alpha, gate, spread, logit_pos, mode)
        # mode: "" = vanilla, "override" = embedding override at term positions
        configs = [
            ("baseline                          ", 0.0, False, None, 0.0, ""),
            ("inject α=2                        ", 2.0, True, None, 0.0, ""),
            ("override only                     ", 0.0, False, None, 0.0, "override"),
            ("override + inject α=2             ", 2.0, True, None, 0.0, "override"),
            ("override + inject α=4             ", 4.0, True, None, 0.0, "override"),
            ("override + inject α=2 + log_pos=4 ", 2.0, True, None, 4.0, "override"),
        ]
        print(f"\nUSER: {prompt}")
        for label, alpha, gate, spread, lp, mode in configs:
            user_prompt = prompt
            formatted = _chat_format(tokenizer, user_prompt) if args.use_chat else user_prompt
            # Find term token positions in the FORMATTED prompt for override mode
            override_positions = None
            if mode == "override":
                ids = tokenizer(formatted, add_special_tokens=False).input_ids
                term_ids = tokenizer(cfg["term"], add_special_tokens=False).input_ids
                # Find ALL positions of the term tokens in the prompt
                positions: list[int] = []
                m = len(term_ids)
                for i in range(len(ids) - m + 1):
                    if ids[i : i + m] == term_ids:
                        positions.extend(range(i, i + m))
                # Also try with a leading-space variant if no match
                if not positions:
                    term_ids2 = tokenizer(" " + cfg["term"], add_special_tokens=False).input_ids
                    m = len(term_ids2)
                    for i in range(len(ids) - m + 1):
                        if ids[i : i + m] == term_ids2:
                            positions.extend(range(i, i + m))
                override_positions = positions if positions else None
            out = generate_with_axiom(
                model,
                tokenizer,
                formatted,
                payload,
                alpha,
                gate,
                args.max_new,
                spread,
                logit_alpha_pos=lp,
                override_term_positions=override_positions,
            )
            score = score_output(out)
            tag_h = f"halluc:{','.join(score['hallucinated'])}" if score["hallucinated"] else "ok"
            tag_l = "LEX" if score["is_lexical"] else "non-lex"
            mark = " [override fired]" if mode == "override" and override_positions else ""
            print(f"  [{label}]: {out.replace(chr(10), ' ').strip()[:280]}{mark}")
            print(f"      [hits={score['hits_axiom']}  {tag_l}  {tag_h}]")


if __name__ == "__main__":
    main()
