"""v9 — slot-assigned soft prompt training.

Each ghost position has a designated role (one Q+A pair). Gradient
masking ensures only the assigned slot updates per step. Each slot's
vector is initialized from informative tokens of its answer.

Architecture per axiom: N_facts slots + 1 overview slot + 3 boundary
slots = ~9 slots total for our 5-fact axioms.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS
from marker.soft_prompt_slots import (
    build_slot_qa_default,
    generate_with_slots,
    make_soft_prompt_slots,
    train_soft_prompt_slots,
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
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--lr-start", type=float, default=0.05)
    parser.add_argument("--lr-end", type=float, default=0.005)
    parser.add_argument("--boundary-slots", type=int, default=3)
    parser.add_argument("--no-train-term", action="store_true", help="Freeze term positions")
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
        f"n_layers={n_layers}  steps={args.n_steps}  "
        f"boundary_slots={args.boundary_slots}  "
        f"train_term={not args.no_train_term}\n"
    )

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}")

        slot_qa = build_slot_qa_default(
            axiom, include_overview=True, boundary_slots=args.boundary_slots
        )
        print(f"\n{len(slot_qa)} slots allocated:")
        for i, (qs, a) in enumerate(slot_qa):
            print(f"  slot {i}: A = {a[:80]}{'...' if len(a) > 80 else ''}")
            print(f"           Q variants ({len(qs)}): {qs[0][:70]}")

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        sp = make_soft_prompt_slots(model, tokenizer, term=name, slot_qa=slot_qa)
        with torch.no_grad():
            init_norms = sp.vector.detach().float().norm(dim=-1).cpu().tolist()
        print(f"\nInit row L2 norms: {[f'{n:.2f}' for n in init_norms]}")

        t0 = time.time()
        losses = train_soft_prompt_slots(
            model,
            tokenizer,
            sp,
            n_steps=args.n_steps,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            train_term_positions=not args.no_train_term,
            append_eos=True,
        )
        with torch.no_grad():
            row_norms = sp.vector.detach().float().norm(dim=-1).cpu().tolist()
        print(
            f"trained sp-slots in {time.time() - t0:.1f}s. "
            f"loss: {losses[0]:.3f} -> {losses[-1]:.3f}"
        )
        print(f"  trained row L2 norms: {[f'{n:.2f}' for n in row_norms]}")

        # Build probe sets
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

                out_T = generate_with_slots(model, tokenizer, sp, full_prompt, max_new=args.max_new)
                print(f"    [T+ slots]:      {out_T.replace(chr(10), ' ').strip()[:280]}")

        # Sample one training Q per slot
        train_probes = [slot_qa[i][0][0] for i in range(min(3, len(slot_qa)))]
        run_probe_set("TRAIN (sampled)", train_probes)
        run_probe_set("HELDOUT (paraphrased, not seen)", heldout_qs)
        run_probe_set("BOUNDARY (out-of-scope)", axiom["boundary_probes"])
        run_probe_set("TELL_ME", [f"Tell me about {name}.", f"Describe {name}."])


if __name__ == "__main__":
    main()
