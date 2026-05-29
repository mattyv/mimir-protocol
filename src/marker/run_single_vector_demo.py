"""Phase 3: head-to-head comparison of the single-vector (seed + bolt-on)
architecture against the existing MLP+KV architecture, on the same model, the
same Q+A data, and the same RNG seed.

For each axiom we print three columns per probe question:

    [A no-axiom]   the frozen base model, no axiom active
    [M mlp+kv]     the existing AxiomMLP + frozen description KV cache
    [S seed+bolt]  the single-vector architecture (new vocab token + bolt-on)

and a mechanical keyword-overlap score on the heldout set for both
architectures. Quality is judged from the side-by-side outputs; the score is
one cheap signal, not the verdict.

Run:
    PYTHONPATH=src python -m marker.run_single_vector_demo \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --n-steps 3000 --n-synthetic 30 --r 32 --lr 1e-4 \
        --save-dir ./single_vector_out --include-skills
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.bolt_selector import (
    install_bolt_hooks,
    make_bolt_selector,
    remove_bolt_hooks,
    train_seed_and_bolt,
)
from marker.prefix_tuning import Prefix
from marker.run_axiom_mlp_demo import (
    ILP_PROBES,
    SKILL_AXIOM,
    SKILL_AXIOM_ILP,
    SKILL_PROBES,
    SUPPLEMENTAL_QA,
    TEMPLATE,
    compute_axiom_kv,
    generate_with_mlp,
    make_axiom_mlp,
    train,
)
from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS, _generic_boundary_examples
from marker.seed_token import register_seed_token
from marker.single_vector_store import save_single_vector_axiom
from marker.soft_prompt_plus import generate_synthetic_qa_pairs

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
    """Lowercased content words of the canonical answer (no stopwords/punct)."""
    words = re.findall(r"[A-Za-z0-9_.]+", answer.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _overlap_score(generated: str, answer: str) -> float:
    """Fraction of the answer's key tokens present in the generation."""
    keys = _key_tokens(answer)
    if not keys:
        return 0.0
    gen_words = set(re.findall(r"[A-Za-z0-9_.]+", generated.lower()))
    return sum(1 for k in keys if k in gen_words) / len(keys)


@torch.no_grad()
def generate_with_bolt(model, tokenizer, prompt, bolt, max_new=120):  # noqa: ANN001
    """Greedy generation with the bolt-on hooks installed. The embedding
    pre-hook fires the bolt-on automatically on prefill (and, in skill mode,
    on subsequent decode steps)."""
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
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def _seedify(text: str, term: str, seed_str: str) -> str:
    """Replace the literal term with the seed token form so the gate fires."""
    return text.replace(term, seed_str)


def _load_model(model_name: str):  # noqa: ANN202
    """Load a FRESH model + tokenizer. We reload per axiom so each axiom is
    trained and evaluated in complete isolation — no embedding-table growth
    across axioms, and (critically) no stacking of the per-seed embedding
    grad-mask hooks, which would zero out the 2nd+ axiom's seed gradient."""
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
    seed_str = f"<{name}>"
    model, tokenizer, n_layers, chosen_layers = _load_model(args.model_name)
    print("\n" + "#" * 78)
    print(f"# FACT axiom: {name}")
    print("#" * 78)

    # ── Build shared training data (term form for MLP+KV) ──────────────────
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
    # Ensure term present in synthetic questions.
    synth = [(q if name.lower() in q.lower() else f"Regarding {name}: {q}", a) for q, a in synth]
    train_qa.extend(synth)
    train_qa.extend(SUPPLEMENTAL_QA.get(name, []))
    overview_qa = [
        (f"Tell me about {name}.", desc),
        (f"What is {name}?", desc),
    ]
    train_qa.extend(overview_qa)
    boundary_qa = _generic_boundary_examples(name)
    print(f"training set: {len(train_qa)} pairs + {len(boundary_qa)} boundary")

    # ── Train MLP+KV ───────────────────────────────────────────────────────
    print("\n[MLP+KV] training...")
    axiom_mlp = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=args.r)
    axiom_mlp.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
    t0 = time.time()
    mlp_losses = train(
        model, tokenizer, axiom_mlp, train_qa, boundary_pairs=boundary_qa, n_steps=args.n_steps
    )
    print(
        f"[MLP+KV] trained in {time.time() - t0:.1f}s  loss {mlp_losses[0]:.3f}→{mlp_losses[-1]:.4f}"
    )

    # ── Train seed + bolt ──────────────────────────────────────────────────
    print("\n[seed+bolt] training...")
    seed = register_seed_token(model, tokenizer, name)
    bolt = make_bolt_selector(model, seed, r=args.r, skill_mode=False)
    sv_qa = [(_seedify(q, name, seed_str), a) for q, a in train_qa]
    t0 = time.time()
    bolt_losses = train_seed_and_bolt(
        model, tokenizer, seed, bolt, sv_qa, n_steps=args.n_steps, lr=args.lr
    )
    print(
        f"[seed+bolt] trained in {time.time() - t0:.1f}s  "
        f"loss {bolt_losses[0]:.3f}→{bolt_losses[-1]:.4f}"
    )
    if save_dir is not None:
        save_single_vector_axiom(model, seed, bolt, save_dir / f"{_slug(name)}.pt")
        print(f"  saved → {save_dir / f'{_slug(name)}.pt'}")

    # ── Probes ───────────────────────────────────────────────────────────────
    def probe(label, questions):  # noqa: ANN001
        print(f"\n--- {label} ---")
        for q in questions:
            mlp_prompt = TEMPLATE.format(q=q)
            sv_prompt = TEMPLATE.format(q=_seedify(q, name, seed_str))
            out_a = generate_with_mlp(model, tokenizer, mlp_prompt, max_new=args.max_new)
            out_m = generate_with_mlp(model, tokenizer, mlp_prompt, axiom_mlp, max_new=args.max_new)
            out_s = generate_with_bolt(model, tokenizer, sv_prompt, bolt, max_new=args.max_new)
            print(f"  Q: {q}")
            print(f"    [A no-axiom]:  {out_a[:200].replace(chr(10), ' ')}")
            print(f"    [M mlp+kv]:    {out_m[:200].replace(chr(10), ' ')}")
            print(f"    [S seed+bolt]: {out_s[:200].replace(chr(10), ' ')}")

    train_qs = [f["questions_train"][0] for f in axiom["facts"]]
    probe("TRAIN (1 per fact)", train_qs)
    probe("HELDOUT", heldout_qs)
    probe("BOUNDARY", axiom["boundary_probes"])
    probe("TELL_ME", [f"Tell me about {name}.", f"What is {name}?"])

    # ── Keyword-overlap score on HELDOUT ─────────────────────────────────────
    mlp_scores, sv_scores = [], []
    for q in heldout_qs:
        ans = heldout_answers[q]
        out_m = generate_with_mlp(
            model, tokenizer, TEMPLATE.format(q=q), axiom_mlp, max_new=args.max_new
        )
        out_s = generate_with_bolt(
            model,
            tokenizer,
            TEMPLATE.format(q=_seedify(q, name, seed_str)),
            bolt,
            max_new=args.max_new,
        )
        mlp_scores.append(_overlap_score(out_m, ans))
        sv_scores.append(_overlap_score(out_s, ans))
    mlp_mean = sum(mlp_scores) / len(mlp_scores) if mlp_scores else 0.0
    sv_mean = sum(sv_scores) / len(sv_scores) if sv_scores else 0.0
    print(f"\n  ★ {name} HELDOUT keyword-overlap: mlp+kv={mlp_mean:.2f}  seed+bolt={sv_mean:.2f}")
    return {"axiom": name, "type": "fact", "mlp_kv": mlp_mean, "seed_bolt": sv_mean}


def _run_skill_axiom(skill, probes, args, save_dir):  # noqa: ANN001
    name = skill["term"]
    desc = skill["description"]
    seed_str = f"<{name}>"
    model, tokenizer, _n_layers, chosen_layers = _load_model(args.model_name)
    print("\n" + "#" * 78)
    print(f"# SKILL axiom: {name}")
    print("#" * 78)

    qa = skill["qa"]

    # ── Train MLP+KV (skill mode) ────────────────────────────────────────────
    print("\n[MLP+KV] training skill...")
    skill_mlp = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=args.skill_r)
    skill_mlp.skill_mode = True
    skill_mlp.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
    t0 = time.time()
    mlp_losses = train(model, tokenizer, skill_mlp, qa, n_steps=args.skill_n_steps)
    print(
        f"[MLP+KV] trained in {time.time() - t0:.1f}s  loss {mlp_losses[0]:.3f}→{mlp_losses[-1]:.4f}"
    )

    # ── Train seed + bolt (skill mode) ───────────────────────────────────────
    print("\n[seed+bolt] training skill...")
    seed = register_seed_token(model, tokenizer, name)
    bolt = make_bolt_selector(model, seed, r=args.skill_r, skill_mode=True)
    sv_qa = [(_seedify(q, name, seed_str), a) for q, a in qa]
    t0 = time.time()
    bolt_losses = train_seed_and_bolt(
        model, tokenizer, seed, bolt, sv_qa, n_steps=args.skill_n_steps, lr=args.lr
    )
    print(
        f"[seed+bolt] trained in {time.time() - t0:.1f}s  "
        f"loss {bolt_losses[0]:.3f}→{bolt_losses[-1]:.4f}"
    )
    if save_dir is not None:
        save_single_vector_axiom(model, seed, bolt, save_dir / f"{_slug(name)}.pt")
        print(f"  saved → {save_dir / f'{_slug(name)}.pt'}")

    # ── Probes ───────────────────────────────────────────────────────────────
    print("\n--- skill probes ---")
    for prompt in probes:
        q = prompt.replace("Q: ", "").replace("\nA:", "")
        sv_prompt = _seedify(prompt, name, seed_str)
        out_a = generate_with_mlp(model, tokenizer, prompt, max_new=args.max_new)
        out_m = generate_with_mlp(model, tokenizer, prompt, skill_mlp, max_new=args.max_new)
        out_s = generate_with_bolt(model, tokenizer, sv_prompt, bolt, max_new=args.max_new)
        print(f"\n  Q: {q}")
        print(f"    [A no-skill]:  {out_a[:220].replace(chr(10), ' ')}")
        print(f"    [M mlp+kv]:    {out_m[:220].replace(chr(10), ' ')}")
        print(f"    [S seed+bolt]: {out_s[:220].replace(chr(10), ' ')}")
    return {"axiom": name, "type": "skill", "mlp_kv": None, "seed_bolt": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--n-synthetic", type=int, default=30)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skill-r", type=int, default=64)
    parser.add_argument("--skill-n-steps", type=int, default=3000)
    parser.add_argument("--max-new", type=int, default=120)
    parser.add_argument("--include-skills", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    print(f"model: {args.model_name}  (fresh load per axiom for isolation)\n")

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"saving single-vector axioms to: {save_dir}\n")

    scores = []
    for axiom in TEST_AXIOMS:
        scores.append(_run_fact_axiom(axiom, args, save_dir))

    if args.include_skills:
        _run_skill_axiom(SKILL_AXIOM, SKILL_PROBES, args, save_dir)
        _run_skill_axiom(SKILL_AXIOM_ILP, ILP_PROBES, args, save_dir)

    # ── Score summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("HELDOUT keyword-overlap summary (fact axioms)")
    print("=" * 78)
    print(f"{'axiom':<24}{'mlp+kv':>10}{'seed+bolt':>12}")
    for s in scores:
        if s["type"] == "fact":
            print(f"{s['axiom']:<24}{s['mlp_kv']:>10.2f}{s['seed_bolt']:>12.2f}")


if __name__ == "__main__":
    main()
