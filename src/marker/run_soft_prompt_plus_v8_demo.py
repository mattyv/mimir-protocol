"""Soft prompt+ v8 — v7 + description-context ghost initialization.

Replaces v7's zero-init ghost vectors with the embeddings of the
description's continuation tokens after the term. The optimizer starts
from an in-distribution, context-rich point and refines from there.

Hypothesis: better init → faster convergence + cleaner final state +
trained vector stays closer to natural embedding manifold.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.run_soft_prompt_plus_v4_demo import (
    TEST_AXIOMS,
    _generic_boundary_examples,
)
from marker.soft_prompt_plus import (
    SoftPromptPlus,
    generate_synthetic_qa_pairs,
    generate_with_soft_prompt_plus,
    train_soft_prompt_plus_qa_v6_batched,
)


@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt: str, max_new: int = 100) -> str:  # noqa: ANN001
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out_ids = ids.clone()
    for _ in range(max_new):
        out = model(out_ids)
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids = torch.cat([out_ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(out_ids[0, ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-ghost", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-start", type=float, default=0.05)
    parser.add_argument("--lr-end", type=float, default=0.005)
    parser.add_argument("--norm-anchor-lambda", type=float, default=0.01)
    parser.add_argument("--n-synthetic", type=int, default=30)
    parser.add_argument("--synth-replication", type=int, default=2)
    parser.add_argument("--boundary-keep", type=int, default=12)
    parser.add_argument("--max-new", type=int, default=120)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    n_layers = model.config.num_hidden_layers
    print(
        f"n_layers={n_layers}  n_ghost={args.n_ghost}  steps={args.n_steps}  "
        f"bs={args.batch_size}  n_synth={args.n_synthetic}\n"
    )

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}\n")

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        print(f"=== Generating {args.n_synthetic} synthetic Q+A pairs via teacher ===")
        t0 = time.time()
        synth_qa = generate_synthetic_qa_pairs(
            model, tokenizer, desc, prefix, n_pairs=args.n_synthetic, max_new=2200
        )
        print(f"  parsed {len(synth_qa)} pairs in {time.time() - t0:.1f}s")

        train_qa: list[tuple[str, str]] = []
        for qa in synth_qa:
            for _ in range(args.synth_replication):
                train_qa.append(qa)
        boundary = _generic_boundary_examples(name)[: args.boundary_keep]
        train_qa.extend(boundary)
        overview = [
            (f"Tell me about {name}.", desc),
            (f"Describe {name}.", desc),
            (f"What is {name}?", desc),
        ]
        train_qa.extend(overview)
        print(f"Total training set: {len(train_qa)} pairs")

        # KEY DIFFERENCE FROM v7: use from_term_with_context.
        sp = SoftPromptPlus.from_term_with_context(
            model, tokenizer, term=name, description=desc, n_ghost=args.n_ghost,
        )
        with torch.no_grad():
            init_norms = sp.vector.detach().float().norm(dim=-1).cpu().tolist()
        print(f"  init vector row L2 norms (context-init): {[f'{n:.2f}' for n in init_norms]}")

        t0 = time.time()
        model_losses, norm_losses = train_soft_prompt_plus_qa_v6_batched(
            model,
            tokenizer,
            sp,
            train_qa,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            append_eos=True,
            norm_anchor_lambda=args.norm_anchor_lambda,
        )
        with torch.no_grad():
            row_norms = sp.vector.detach().float().norm(dim=-1).cpu().tolist()
        print(
            f"trained sp+ v8 in {time.time() - t0:.1f}s. "
            f"model_loss: {model_losses[0]:.3f} -> {model_losses[-1]:.3f}"
        )
        print(f"  trained vector row L2 norms: {[f'{n:.2f}' for n in row_norms]}")

        heldout_qs: list[str] = []
        for f in axiom["facts"]:
            heldout_qs.extend(f["questions_heldout"])

        def run_probe_set(label: str, probes: list[str]) -> None:
            print(f"\n--- {label} probes ---")
            for probe in probes:
                full_prompt = f"Q: {probe}\nA:"
                print(f"\n  USER: {probe}")
                out_A = _greedy_generate(model, tokenizer, full_prompt, max_new=args.max_new)
                print(f"    [A no-axiom]:    {out_A.replace(chr(10), ' ').strip()[:280]}")

                out_P = generate_with_prefixes(
                    model, tokenizer, full_prompt, [prefix], args.max_new
                )
                print(f"    [P full-prefix]: {out_P.replace(chr(10), ' ').strip()[:280]}")

                out_T = generate_with_soft_prompt_plus(
                    model, tokenizer, sp, full_prompt, max_new=args.max_new
                )
                print(f"    [T+ sp+v8]:      {out_T.replace(chr(10), ' ').strip()[:280]}")

        run_probe_set("SYNTHETIC (training set sampled)", [q for q, _ in synth_qa[:3]])
        run_probe_set("HELDOUT (paraphrased)", heldout_qs)
        run_probe_set("BOUNDARY (out-of-scope)", axiom["boundary_probes"])
        run_probe_set("TELL_ME", [f"Tell me about {name}.", f"Describe {name}."])


if __name__ == "__main__":
    main()
