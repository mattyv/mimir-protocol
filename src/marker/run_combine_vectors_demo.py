"""Combine existing per-axiom prefix vectors to approximate the composed
capture. Tests whether sub-axioms can stay reusable building blocks.

Strategies (each yields a single DynamicCache that drives decode):

  M_concat       — rope-fix concat of per-axiom prefixes (broken baseline).
  M_bind        — concat + "thinking pass": run the composition_note
                   text through the model with the concat cache as
                   context, and EXTEND the cache with the resulting K/V.
                   Those new positions' attention saw all axiom slots,
                   so they carry cross-axiom binding the static concat
                   lacks.
  M_avg          — average per-axiom prefixes (each captured at positions
                   0..N independently). Vector-arithmetic baseline.
  H              — full composed prefix (run model on the joint document
                   in one shot; the upper bound for "compose-from-parts").
  E              — Path 2 joint encoding at query time (always-correct).

If M_bind matches H, sub-axioms remain composable building blocks: you
keep per-axiom prefixes in storage and stitch them on demand with a
short thinking pass per top-level concept.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.axiom_registry import HIERARCHICAL_AXIOMS, composed_description
from marker.prefix_tuning import (
    Prefix,
    _get_rope_theta,
    _model_dtype,
    combined_cache,
    generate_with_prefixes,
)
from marker.run_chain_ape_recursive_demo import HIERARCHY_PROMPTS, hallucination_flags


@torch.no_grad()
def _generate_with_joint_encoding(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    descriptions: list[str],
    max_new: int = 180,
) -> str:
    device = next(model.parameters()).device
    joint_text = "\n\n".join(descriptions)
    joint_ids = tokenizer(joint_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(joint_ids, past_key_values=DynamicCache(), use_cache=True)
    cache: DynamicCache = out.past_key_values
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, nxt], dim=1)
    if int(nxt.item()) == tokenizer.eos_token_id:
        return ""
    for _ in range(max_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        full_ids = torch.cat([full_ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    new_ids = full_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


@torch.no_grad()
def _decode_against_cache(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    cache: DynamicCache,
    max_new: int,
) -> str:
    """Greedy decode of `prompt` using `cache` as past_key_values.

    The cache is mutated by the forward (HF appends to it) — caller
    should clone first if reuse is needed.
    """
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model(ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    full_ids = torch.cat([ids, nxt], dim=1)
    if int(nxt.item()) == tokenizer.eos_token_id:
        return ""
    for _ in range(max_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        full_ids = torch.cat([full_ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    new_ids = full_ids[0, ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    out = DynamicCache()
    for i in range(len(cache)):
        out.update(cache.layers[i].keys.clone(), cache.layers[i].values.clone(), i)
    return out


@torch.no_grad()
def build_bind_cache(
    model,  # noqa: ANN001
    tokenizer,
    sub_prefixes: list[Prefix],
    composition_note: str,
    rope_correct: bool = True,
) -> DynamicCache:
    """Concat sub-axiom prefixes (rope-fix) → run composition_note
    through the model with that cache as context → extended cache.
    The composition_note's K/V positions have attention to all sub-axiom
    slots, carrying cross-axiom binding."""
    device = next(model.parameters()).device
    dtype = _model_dtype(model)
    rope_theta = _get_rope_theta(model)
    cache = combined_cache(
        sub_prefixes,
        dtype=dtype,
        device=device,
        rope_theta=rope_theta,
        rope_correct=rope_correct,
    )
    note_ids = tokenizer(
        composition_note, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    _ = model(note_ids, past_key_values=cache, use_cache=True)
    return cache  # mutated in place


@torch.no_grad()
def build_avg_prefix(
    model,  # noqa: ANN001
    tokenizer,
    sub_axiom_keys: list[str],
    n_tokens: int,
) -> Prefix:
    """Capture each sub-axiom independently at positions 0..N, then
    average per-layer K and V across axioms. The result is a single
    Prefix of length N. Each axiom contributes equally per slot.
    """
    layers = list(range(model.config.num_hidden_layers))
    prefixes = [
        Prefix.from_description(
            model,
            tokenizer,
            HIERARCHICAL_AXIOMS[k]["description"],
            max_tokens=n_tokens,
            target_layers=layers,
        )
        for k in sub_axiom_keys
    ]
    # Truncate every prefix to the shortest (to keep alignment).
    min_n = min(p.n_tokens for p in prefixes)
    avg_keys: list[torch.nn.Parameter] = []
    avg_values: list[torch.nn.Parameter] = []
    for layer_idx in range(prefixes[0].n_total_layers):
        k_stack = torch.stack([p.keys[layer_idx][:, :, :min_n, :].float() for p in prefixes], dim=0)
        v_stack = torch.stack(
            [p.values[layer_idx][:, :, :min_n, :].float() for p in prefixes], dim=0
        )
        avg_keys.append(torch.nn.Parameter(k_stack.mean(dim=0)))
        avg_values.append(torch.nn.Parameter(v_stack.mean(dim=0)))
    p0 = prefixes[0]
    return Prefix(
        n_tokens=min_n,
        n_total_layers=p0.n_total_layers,
        n_kv_heads=p0.n_kv_heads,
        head_dim=p0.head_dim,
        target_layers=p0.target_layers,
        keys=avg_keys,
        values=avg_values,
        per_layer_shapes=p0.per_layer_shapes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=64)
    parser.add_argument("--composed-prefix-tokens", type=int, default=768)
    parser.add_argument("--max-new", type=int, default=180)
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument("--top-axiom", default="data_pipeline")
    parser.add_argument("--only-3plus", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    layers = list(range(model.config.num_hidden_layers))
    cfg = HIERARCHICAL_AXIOMS[args.top_axiom]
    sub_keys = cfg["composed_of"]
    composition_note = cfg["composition_note"]

    # Per-axiom sub-prefixes (the building blocks)
    print(f"\n=== capturing {len(sub_keys)} sub-axiom prefixes ===")
    sub_prefixes: list[Prefix] = []
    descriptions: list[str] = []
    for k in sub_keys:
        descriptions.append(HIERARCHICAL_AXIOMS[k]["description"])
        sub_prefixes.append(
            Prefix.from_description(
                model,
                tokenizer,
                HIERARCHICAL_AXIOMS[k]["description"],
                max_tokens=args.n_prefix_tokens,
                target_layers=layers,
            )
        )
    descriptions.append(cfg["description"])

    # H — composed prefix (winner from last test)
    print("\n=== capturing composed prefix (H) ===")
    composed_doc = composed_description(args.top_axiom)
    composed_prefix = Prefix.from_description(
        model,
        tokenizer,
        composed_doc,
        max_tokens=args.composed_prefix_tokens,
        target_layers=layers,
    )
    print(f"H prefix n_tokens={composed_prefix.n_tokens}")

    # M_avg — averaged across axioms
    print("\n=== building M_avg ===")
    avg_prefix = build_avg_prefix(model, tokenizer, sub_keys, args.n_prefix_tokens)
    print(f"M_avg prefix n_tokens={avg_prefix.n_tokens}")

    def fmt(p: str) -> str:
        if not args.use_chat:
            return p
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return p

    def run_one(keys: list[str], prompt: str) -> None:
        formatted = fmt(prompt)

        def timed(fn) -> tuple[str, float]:  # noqa: ANN001
            t0 = time.time()
            out = fn()
            return out, time.time() - t0

        rows: list[tuple[str, str, float, int, list[str]]] = []

        def record(label: str, out: str, dt: float) -> None:
            n_hall, flags = hallucination_flags(out)
            rows.append((label, out, dt, n_hall, flags))

        # M_concat — broken baseline
        out, dt = timed(
            lambda: generate_with_prefixes(
                model, tokenizer, formatted, sub_prefixes, args.max_new, rope_correct=True
            )
        )
        record("M_concat ", out, dt)
        # M_avg — vector average
        out, dt = timed(
            lambda: generate_with_prefixes(model, tokenizer, formatted, [avg_prefix], args.max_new)
        )
        record("M_avg    ", out, dt)
        # M_bind — concat + thinking pass
        bind_cache = build_bind_cache(
            model, tokenizer, sub_prefixes, composition_note, rope_correct=True
        )
        bind_cache_clone = _clone_cache(bind_cache)
        out, dt = timed(
            lambda c=bind_cache_clone: _decode_against_cache(
                model, tokenizer, formatted, c, args.max_new
            )
        )
        record("M_bind   ", out, dt)
        # H — composed
        out, dt = timed(
            lambda: generate_with_prefixes(
                model, tokenizer, formatted, [composed_prefix], args.max_new
            )
        )
        record("H composed", out, dt)
        # E — joint encoding upper bound
        out, dt = timed(
            lambda: _generate_with_joint_encoding(
                model, tokenizer, formatted, descriptions, args.max_new
            )
        )
        record("E joint  ", out, dt)

        print(f"\nUSER: {prompt}")
        for label, out, dt, n_hall, flags in rows:
            preview = out.replace(chr(10), " ").strip()[:600]
            hall_tag = f"[hall={n_hall}]" if n_hall == 0 else f"[HALL={n_hall}: {flags[:5]}]"
            print(f"  [{label}] ({dt:5.1f}s) {hall_tag}: {preview}")

    print("\n" + "#" * 78)
    print(f"# Combine-vectors test: {args.top_axiom}")
    print("#" * 78)
    for keys, prompt in HIERARCHY_PROMPTS:
        if args.only_3plus and len(keys) < 3:
            continue
        run_one(keys, prompt)


if __name__ == "__main__":
    main()
