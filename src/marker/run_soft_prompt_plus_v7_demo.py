"""Soft prompt+ v7 — teacher-distilled training data.

The teacher (model + full KV prefix) generates synthetic Q+A pairs
from the description. The student trains on those plus hand-written
boundary + overview examples.

Why this should beat v6:
  - Instead of 23 hand-written paraphrases, we get 30+ teacher-
    generated Q+A pairs covering more aspects of the description.
  - The teacher's answers are by construction correct (the model
    has full access to the description via the prefix).
  - We're no longer bottlenecked on human Q+A authoring.

Combined with v6's batched training + H100 + n_steps=2000 from v6.
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
    parser.add_argument(
        "--n-synthetic", type=int, default=30, help="Number of teacher Q+A pairs to generate"
    )
    parser.add_argument(
        "--synth-replication",
        type=int,
        default=2,
        help="Replicate each synthetic pair this many times (boundary balance)",
    )
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
        f"bs={args.batch_size}  n_synth={args.n_synthetic}  boundary={args.boundary_keep}\n"
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

        # Teacher-generated synthetic Q+A pairs.
        print(
            f"=== Generating {args.n_synthetic} synthetic Q+A pairs via teacher (full prefix) ==="
        )
        t0 = time.time()
        synth_qa = generate_synthetic_qa_pairs(
            model, tokenizer, desc, prefix, n_pairs=args.n_synthetic, max_new=2200
        )
        print(f"  parsed {len(synth_qa)} pairs in {time.time() - t0:.1f}s")
        for i, (q, a) in enumerate(synth_qa[:6]):
            print(f"    [{i + 1}] Q: {q}")
            print(f"        A: {a}")
        if len(synth_qa) > 6:
            print(f"    ... ({len(synth_qa) - 6} more)")

        # Build training set: synthetic (replicated) + boundary + overview.
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
        print(
            f"\nTotal training set: {len(train_qa)} pairs "
            f"({len(synth_qa) * args.synth_replication} synth-replicated + "
            f"{len(boundary)} boundary + {len(overview)} overview)"
        )

        sp = SoftPromptPlus.from_term(model, tokenizer, term=name, n_ghost=args.n_ghost)
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
            f"trained sp+ v7 in {time.time() - t0:.1f}s. "
            f"model_loss: {model_losses[0]:.3f} -> {model_losses[-1]:.3f}"
        )
        print(f"  trained vector row L2 norms: {[f'{n:.2f}' for n in row_norms]}")

        # Probe set
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
                print(f"    [T+ sp+v7]:      {out_T.replace(chr(10), ' ').strip()[:280]}")

        # Sample a few synthetic queries
        run_probe_set("SYNTHETIC (training set sampled)", [q for q, _ in synth_qa[:3]])
        run_probe_set("HELDOUT (paraphrased)", heldout_qs)
        run_probe_set("BOUNDARY (out-of-scope)", axiom["boundary_probes"])
        run_probe_set("TELL_ME", [f"Tell me about {name}.", f"Describe {name}."])


if __name__ == "__main__":
    main()
