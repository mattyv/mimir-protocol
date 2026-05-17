"""Soft prompt+ v5 — v4 + L2-norm anchor regularization.

Adds a regularization term that keeps each trained vector row's L2 norm
close to the average L2 norm of natural vocabulary embeddings. The
hypothesis: keeping magnitudes natural keeps W_Q and W_K's outputs in
the distribution the model was built to handle, reducing the
off-manifold drift that causes hallucination.

Strength controlled by `norm_anchor_lambda` (default 0.01).
Everything else identical to v4.
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
    generate_with_soft_prompt_plus,
    train_soft_prompt_plus_qa_v5,
)

# v5 rebalance: cut boundary count and double-include fact Q+A so facts
# are sampled twice as often during training.
_V5_BOUNDARY_KEEP_DEFAULT = 12  # v4 used 20, v5 (initial) used 12
_V5_FACT_REPLICATION = 2  # each fact Q+A appears this many times in the list


def _build_training_set_v5(
    axiom: dict, boundary_keep: int = _V5_BOUNDARY_KEEP_DEFAULT
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    train_qa: list[tuple[str, str]] = []
    train_qs: list[str] = []
    heldout_qs: list[str] = []
    for f in axiom["facts"]:
        for q in f["questions_train"]:
            # Replicate each fact pair so it gets sampled more often.
            for _ in range(_V5_FACT_REPLICATION):
                train_qa.append((q, f["answer"]))
            train_qs.append(q)
        for q in f["questions_heldout"]:
            heldout_qs.append(q)
    # Reduced-count generic boundary examples (first K).
    boundary = _generic_boundary_examples(axiom["name"])[:boundary_keep]
    train_qa.extend(boundary)
    desc = axiom["description"]
    overview = [
        (f"Tell me about {axiom['name']}.", desc),
        (f"Describe {axiom['name']}.", desc),
        (f"What is {axiom['name']}?", desc),
    ]
    train_qa.extend(overview)
    return train_qa, train_qs, heldout_qs


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
    parser.add_argument("--n-steps", type=int, default=3500)
    parser.add_argument("--lr-start", type=float, default=0.05)
    parser.add_argument("--lr-end", type=float, default=0.005)
    parser.add_argument("--norm-anchor-lambda", type=float, default=0.01)
    parser.add_argument("--boundary-keep", type=int, default=_V5_BOUNDARY_KEEP_DEFAULT)
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
        f"lr {args.lr_start} -> {args.lr_end}  norm_lambda={args.norm_anchor_lambda}\n"
    )

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]
        train_qa, train_qs, heldout_qs = _build_training_set_v5(
            axiom, boundary_keep=args.boundary_keep
        )

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}")
        print(f"#train Q+A pairs: {len(train_qa)}\n")

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        sp = SoftPromptPlus.from_term(model, tokenizer, term=name, n_ghost=args.n_ghost)
        t0 = time.time()
        model_losses, norm_losses = train_soft_prompt_plus_qa_v5(
            model,
            tokenizer,
            sp,
            train_qa,
            n_steps=args.n_steps,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            append_eos=True,
            norm_anchor_lambda=args.norm_anchor_lambda,
        )
        # Show trained vector L2 norms vs natural norm.
        with torch.no_grad():
            row_norms = sp.vector.detach().float().norm(dim=-1).cpu().tolist()
        print(
            f"trained sp+ v5 in {time.time() - t0:.1f}s. "
            f"model_loss: {model_losses[0]:.3f} -> {model_losses[-1]:.3f}  "
            f"norm_loss: {norm_losses[0]:.3f} -> {norm_losses[-1]:.3f}"
        )
        print(f"  trained vector row L2 norms: {[f'{n:.2f}' for n in row_norms]}")

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
                print(f"    [T+ sp+v5]:      {out_T.replace(chr(10), ' ').strip()[:280]}")

        run_probe_set("TRAIN (sampled)", train_qs[:3])
        run_probe_set("HELDOUT (paraphrased)", heldout_qs)
        run_probe_set("BOUNDARY (out-of-scope)", axiom["boundary_probes"])
        run_probe_set(
            "TELL_ME",
            [f"Tell me about {name}.", f"Describe {name}."],
        )


if __name__ == "__main__":
    main()
