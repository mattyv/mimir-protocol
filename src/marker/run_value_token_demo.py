"""Phase 4: head-to-head comparison of value-token architecture vs MLP+KV.

Value-token architecture: one seed token (subject identity) + one value token
per novel identifier (e.g. <balances.raw>) + a gated bolt at a narrow band of
mid-stack layers. No KV cache at all. Every string in the answer is either a
common value the bolt can push (250ms, Parquet) or a value token the bolt
routes to via a single-token prediction.

Three output columns per probe:

    [A no-axiom]   frozen base model, no axiom
    [M mlp+kv]     existing AxiomMLP + frozen description KV cache (baseline)
    [V val-tok]    value-token architecture (this experiment)

Score is split into tier-1 (answers containing only common in-vocab values)
and tier-2 (answers containing a novel identifier) to measure each part
independently.

Run:
    PYTHONPATH=src python -m marker.run_value_token_demo \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --n-steps 3000 --n-synthetic 30 --r 32 --lr 1e-4 \\
        --fact-layers 14 15 16 --save-dir ./value_token_out
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.bolt_selector import install_bolt_hooks, make_bolt_selector, remove_bolt_hooks
from marker.prefix_tuning import Prefix
from marker.run_axiom_mlp_demo import (
    SUPPLEMENTAL_QA,
    TEMPLATE,
    compute_axiom_kv,
    generate_with_mlp,
    make_axiom_mlp,
    train,
)
from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS, _generic_boundary_examples
from marker.soft_prompt_plus import generate_synthetic_qa_pairs
from marker.value_token import AxiomTokens, register_axiom_tokens, train_axiom_tokens

# Novel identifier strings that cannot be emitted by the bolt alone.
# These become value tokens so they are single-token predictions.
_VALUE_SURFACES: dict[str, list[str]] = {
    "BalancePublisher": ["balances.raw"],
    "FluxomService": ["warehouse.fluxom_ingested"],
}

_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "is",
    "are",
    "and",
    "or",
    "for",
    "in",
    "on",
    "it",
    "its",
    "with",
    "no",
    "does",
    "do",
    "what",
    "how",
    "where",
    "which",
    "every",
    "our",
}


def _slug(term: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", term).strip("_") or "axiom"


def _key_tokens(answer: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_.]+", answer.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _overlap_score(generated: str, answer: str) -> float:
    keys = _key_tokens(answer)
    if not keys:
        return 0.0
    gen_words = set(re.findall(r"[A-Za-z0-9_.]+", generated.lower()))
    return sum(1 for k in keys if k in gen_words) / len(keys)


def _is_tier2(answer: str, value_surfaces: list[str]) -> bool:
    """True if the answer contains at least one novel identifier."""
    low = answer.lower()
    return any(s.lower() in low for s in value_surfaces)


@torch.no_grad()
def _generate_with_bolt(model, tokenizer, prompt, bolt, max_new=120):  # noqa: ANN001
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    handles = install_bolt_hooks(model, bolt)
    try:
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        remove_bolt_hooks(handles)
    new = out[0, ids.shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=False).strip()


def _seedify(text: str, term: str, axiom: AxiomTokens) -> str:
    """Replace the subject term with the seed token and leave value surfaces
    as-is (the tokenizer encodes them as value tokens automatically)."""
    return text.replace(term, f"<{axiom.seed.name}>")


def _load_model(model_name: str):  # noqa: ANN202
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]
    return model, tokenizer, n_layers, chosen_layers


def _run_fact_axiom(axiom, args, save_dir):  # noqa: ANN001
    name = axiom["name"]
    desc = axiom["description"]
    value_surfaces = _VALUE_SURFACES.get(name, [])
    model, tokenizer, n_layers, chosen_layers = _load_model(args.model_name)

    print("\n" + "#" * 78)
    print(f"# FACT axiom: {name}")
    print(f"# value surfaces: {value_surfaces}")
    print("#" * 78)

    # ── Build training data ────────────────────────────────────────────────────
    train_qa: list[tuple[str, str]] = []
    heldout_qs: list[str] = []
    heldout_answers: dict[str, str] = {}
    for f in axiom["facts"]:
        for q in f["questions_train"]:
            train_qa.append((q, f["answer"]))
        for q in f["questions_heldout"]:
            heldout_qs.append(q)
            heldout_answers[q] = f["answer"]

    prefix = Prefix.from_description(
        model,
        tokenizer,
        desc,
        max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
        target_layers=list(range(n_layers)),
    )
    print(f"=== generating {args.n_synthetic} synthetic Q+A via teacher ===")
    synth = generate_synthetic_qa_pairs(
        model, tokenizer, desc, prefix, n_pairs=args.n_synthetic, max_new=2200
    )
    synth = [(q if name.lower() in q.lower() else f"Regarding {name}: {q}", a) for q, a in synth]
    train_qa.extend(synth)
    train_qa.extend(SUPPLEMENTAL_QA.get(name, []))
    train_qa.extend([(f"Tell me about {name}.", desc), (f"What is {name}?", desc)])
    boundary_qa = _generic_boundary_examples(name)
    print(f"training set: {len(train_qa)} pairs + {len(boundary_qa)} boundary")

    # ── Train MLP+KV baseline ──────────────────────────────────────────────────
    print("\n[MLP+KV] training...")
    axiom_mlp = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=args.r)
    axiom_mlp.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
    t0 = time.time()
    mlp_losses = train(
        model, tokenizer, axiom_mlp, train_qa, boundary_pairs=boundary_qa, n_steps=args.n_steps
    )
    print(
        f"[MLP+KV] trained in {time.time() - t0:.1f}s  "
        f"loss {mlp_losses[0]:.3f}→{mlp_losses[-1]:.4f}"
    )

    # ── Register value tokens + train ─────────────────────────────────────────
    print("\n[val-tok] registering tokens + training...")
    axiom_tok = register_axiom_tokens(model, tokenizer, name, value_surfaces)
    bolt = make_bolt_selector(
        model, axiom_tok.seed, r=args.r, skill_mode=False, layers=args.fact_layers
    )
    # Seedify only the subject term; value surfaces stay literal so the
    # tokenizer encodes them as value tokens automatically.
    vt_qa = [(_seedify(q, name, axiom_tok), a) for q, a in train_qa]
    boundary_vt = [(_seedify(q, name, axiom_tok), a) for q, a in boundary_qa]
    vt_qa.extend(boundary_vt)
    t0 = time.time()
    vt_losses = train_axiom_tokens(
        model, tokenizer, axiom_tok, bolt, vt_qa, n_steps=args.n_steps, lr=args.lr
    )
    print(
        f"[val-tok] trained in {time.time() - t0:.1f}s  loss {vt_losses[0]:.3f}→{vt_losses[-1]:.4f}"
    )
    if save_dir is not None:
        _save(model, axiom_tok, bolt, save_dir / f"{_slug(name)}.pt")

    # ── Probes ────────────────────────────────────────────────────────────────
    def probe(label, questions):  # noqa: ANN001
        print(f"\n--- {label} ---")
        for q in questions:
            mlp_prompt = TEMPLATE.format(q=q)
            vt_prompt = TEMPLATE.format(q=_seedify(q, name, axiom_tok))
            out_a = generate_with_mlp(model, tokenizer, mlp_prompt, max_new=args.max_new)
            out_m = generate_with_mlp(model, tokenizer, mlp_prompt, axiom_mlp, max_new=args.max_new)
            out_v = _generate_with_bolt(model, tokenizer, vt_prompt, bolt, max_new=args.max_new)
            print(f"  Q: {q}")
            print(f"    [A no-axiom]: {out_a[:200].replace(chr(10), ' ')}")
            print(f"    [M mlp+kv]:   {out_m[:200].replace(chr(10), ' ')}")
            print(f"    [V val-tok]:  {out_v[:200].replace(chr(10), ' ')}")

    train_qs = [f["questions_train"][0] for f in axiom["facts"]]
    probe("TRAIN (1 per fact)", train_qs)
    probe("HELDOUT", heldout_qs)
    probe("BOUNDARY", axiom["boundary_probes"])
    probe("TELL_ME", [f"Tell me about {name}.", f"What is {name}?"])

    # ── Keyword-overlap score split by tier ────────────────────────────────────
    mlp_t1, mlp_t2, vt_t1, vt_t2 = [], [], [], []
    for q in heldout_qs:
        ans = heldout_answers[q]
        out_m = generate_with_mlp(
            model, tokenizer, TEMPLATE.format(q=q), axiom_mlp, max_new=args.max_new
        )
        out_v = _generate_with_bolt(
            model,
            tokenizer,
            TEMPLATE.format(q=_seedify(q, name, axiom_tok)),
            bolt,
            max_new=args.max_new,
        )
        sm, sv = _overlap_score(out_m, ans), _overlap_score(out_v, ans)
        if _is_tier2(ans, value_surfaces):
            mlp_t2.append(sm)
            vt_t2.append(sv)
        else:
            mlp_t1.append(sm)
            vt_t1.append(sv)

    def _mean(lst):  # noqa: ANN001
        return sum(lst) / len(lst) if lst else float("nan")

    print(f"\n  ★ {name} HELDOUT keyword-overlap:")
    print(
        f"      tier-1 (common values):     mlp+kv={_mean(mlp_t1):.2f}  val-tok={_mean(vt_t1):.2f}"
    )
    print(
        f"      tier-2 (novel identifiers): mlp+kv={_mean(mlp_t2):.2f}  val-tok={_mean(vt_t2):.2f}"
    )

    return {
        "axiom": name,
        "mlp_t1": _mean(mlp_t1),
        "vt_t1": _mean(vt_t1),
        "mlp_t2": _mean(mlp_t2),
        "vt_t2": _mean(vt_t2),
    }


def _save(model, axiom_tok: AxiomTokens, bolt, path: Path) -> None:  # noqa: ANN001
    from marker.seed_token import seed_embedding

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": axiom_tok.seed.name,
            "original_term": axiom_tok.seed.original_term,
            "original_bpe_ids": axiom_tok.seed.original_bpe_ids,
            "seed_vector": seed_embedding(model, axiom_tok.seed).detach().cpu().clone(),
            "values": [
                {
                    "surface": v.surface,
                    "token_id": v.token_id,
                    "original_bpe_ids": v.original_bpe_ids,
                    "vector": model.get_input_embeddings()
                    .weight[v.token_id]
                    .detach()
                    .cpu()
                    .clone(),
                }
                for v in axiom_tok.values
            ],
            "r": bolt.r,
            "fire_layers": bolt.fire_layers,
            "adapter_state": {k: v.cpu() for k, v in bolt.adapters.state_dict().items()},
        },
        path,
    )
    print(f"  saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--n-synthetic", type=int, default=30)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-new", type=int, default=120)
    parser.add_argument(
        "--fact-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices for the bolt (default: mid-third of the stack)",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    # Default: mid-third of stack (set after model load; override via flag).
    # If --fact-layers not given we resolve it per axiom inside _load_model's
    # returned n_layers, so we patch args here as a sentinel.
    if args.fact_layers is None:
        args._fact_layers_auto = True
    else:
        args._fact_layers_auto = False

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"saving value-token axioms to: {save_dir}\n")

    print(f"model: {args.model_name}  (fresh load per axiom for isolation)")
    print(f"fact layers: {'auto (mid-third)' if args._fact_layers_auto else args.fact_layers}\n")

    scores = []
    for axiom in TEST_AXIOMS:
        # Resolve auto layers once we know n_layers — peek without loading twice
        # by temporarily loading just the config.
        if args._fact_layers_auto:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(args.model_name)
            n = cfg.num_hidden_layers
            args.fact_layers = list(range(n // 3, (2 * n) // 3))
            print(f"auto fact-layers: {args.fact_layers}\n")
            args._fact_layers_auto = False  # only compute once

        scores.append(_run_fact_axiom(axiom, args, save_dir))

    print("\n" + "=" * 78)
    print("HELDOUT keyword-overlap summary")
    print("=" * 78)
    print(f"{'axiom':<24}{'mlp t1':>8}{'vt t1':>8}  {'mlp t2':>8}{'vt t2':>8}")
    for s in scores:
        print(
            f"{s['axiom']:<24}{s['mlp_t1']:>8.2f}{s['vt_t1']:>8.2f}"
            f"  {s['mlp_t2']:>8.2f}{s['vt_t2']:>8.2f}"
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
