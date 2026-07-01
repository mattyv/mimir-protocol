"""Hypernetwork KV store demo: per-axiom latent + shared decoder → KV.

Trains one shared KVHypernet on a set of axioms, then shows:

  1. STORAGE: full KV (MB) vs stored code (latent floats + fact_text bytes).
  2. RECONSTRUCTION: heldout Q+A under three KVs —
       [FULL]      compute_axiom_kv(description)         (baseline)
       [HYPER]     decode(z) ++ verbatim facts           (the new store)
       [SCAFFOLD]  decode(z) only, no facts              (proves facts matter)
     Axiom MLPs are left zero-init (no-op), so this isolates KV quality.
  3. REALTIME ADD: register a HELD-OUT axiom the hypernet never trained on,
     via encode() alone (no training), and probe it.

Run (GPU):
    PYTHONPATH=src python -m marker.run_hypernet_demo --model-name Qwen/Qwen2.5-7B
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.kv_hypernet import (
    KVHypernet,
    build_axiom_kv,
    facts_to_text,
    make_axiom_code,
    train_hypernet,
)
from marker.run_axiom_mlp_demo import (
    TEMPLATE,
    TEST_AXIOMS,
    compute_axiom_kv,
    generate_with_mlp,
    make_axiom_mlp,
)


def _kv_mb(kv) -> float:  # noqa: ANN001
    return sum(k.nbytes + v.nbytes for k, v in zip(kv.keys, kv.values, strict=True)) / 1024**2


def _code_kb(code) -> float:  # noqa: ANN001
    return (code.z.numel() * 4 + len(code.fact_text.encode())) / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--d-latent", type=int, default=512)
    parser.add_argument("--n-scaffold", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=60)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]

    # ── Build axioms: MLPs stay zero-init (no-op) so we isolate KV quality ──────
    train_axioms = TEST_AXIOMS[:-1]  # last one is held out for the realtime-add test
    heldout_axiom = TEST_AXIOMS[-1]

    axiom_mlps = []
    fact_texts: dict[str, str] = {}
    qa_map: dict[str, list[tuple[str, str]]] = {}
    for axiom in train_axioms:
        name, desc = axiom["name"], axiom["description"]
        a = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=4)
        a.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
        axiom_mlps.append(a)
        fact_texts[name] = facts_to_text(axiom)
        qa_map[name] = [(q, f["answer"]) for f in axiom["facts"] for q in f["questions_train"]]
        print(f"axiom {name}: full KV {_kv_mb(a.kv):.1f} MB")

    # ── Train the shared hypernet once ─────────────────────────────────────────
    hypernet = KVHypernet(
        n_layers=n_layers,
        n_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        d_latent=args.d_latent,
        n_scaffold=args.n_scaffold,
    )
    n_params = sum(p.numel() for p in hypernet.parameters())
    print(f"\nhypernet params: {n_params:,}  training {args.n_steps} steps...")
    hypernet = train_hypernet(
        model, tokenizer, hypernet, axiom_mlps, fact_texts, qa_map, n_steps=args.n_steps
    )

    # ── 1 + 2: storage + reconstruction on heldout Qs ──────────────────────────
    print("\n" + "=" * 78)
    print("STORAGE + RECONSTRUCTION (MLP is no-op → pure KV comparison)")
    print("=" * 78)
    for axiom, a in zip(train_axioms, axiom_mlps, strict=True):
        name, desc = axiom["name"], axiom["description"]
        code = make_axiom_code(hypernet, a.kv, fact_texts[name], name)
        full_kv = a.kv
        hyper_kv = build_axiom_kv(hypernet, code, model, tokenizer)
        scaffold_kv = hypernet.decode_scaffold(code.z, next(model.parameters()).device)

        print(f"\n### {name}")
        print(
            f"  storage: full {_kv_mb(full_kv):.1f} MB  →  code {_code_kb(code):.1f} KB "
            f"({code.z.numel()} latent floats + {len(code.fact_text)} fact chars)"
        )
        heldout = [q for f in axiom["facts"] for q in f["questions_heldout"]]
        for q in heldout[:4]:
            prompt = TEMPLATE.format(q=q)
            outs = {}
            for label, kv in [("FULL", full_kv), ("HYPER", hyper_kv), ("SCAFFOLD", scaffold_kv)]:
                a.kv = kv
                outs[label] = generate_with_mlp(model, tokenizer, prompt, a, max_new=args.max_new)
            a.kv = full_kv
            print(f"  Q: {q}")
            for label in ("FULL", "HYPER", "SCAFFOLD"):
                print(f"    [{label:8}] {outs[label][:110].replace(chr(10), ' ')}")

    # ── 3: realtime add of a never-trained axiom (encode only, no training) ─────
    print("\n" + "=" * 78)
    print(f"REALTIME ADD — {heldout_axiom['name']} (hypernet never saw it; encode-only)")
    print("=" * 78)
    name, desc = heldout_axiom["name"], heldout_axiom["description"]
    ho = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=4)
    full_kv = compute_axiom_kv(model, tokenizer, desc, term=name)
    code = make_axiom_code(hypernet, full_kv, facts_to_text(heldout_axiom), name)
    print(f"  encoded in one forward pass → code {_code_kb(code):.1f} KB (no training)")
    hyper_kv = build_axiom_kv(hypernet, code, model, tokenizer)
    for q in [qq for f in heldout_axiom["facts"] for qq in f["questions_heldout"]][:4]:
        prompt = TEMPLATE.format(q=q)
        ho.kv = full_kv
        out_full = generate_with_mlp(model, tokenizer, prompt, ho, max_new=args.max_new)
        ho.kv = hyper_kv
        out_hyper = generate_with_mlp(model, tokenizer, prompt, ho, max_new=args.max_new)
        print(f"  Q: {q}")
        print(f"    [FULL ] {out_full[:110].replace(chr(10), ' ')}")
        print(f"    [HYPER] {out_hyper[:110].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
