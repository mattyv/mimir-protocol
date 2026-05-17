"""Slot-axiom v2: train on Q+A pairs, not on description reproduction.

Previous experiment found that training a slot to reproduce the
description text drove loss to ~0 but the slot didn't generalize to
question contexts. Hypothesis: the slot becomes condition-specific to
the training prefix.

This v2 tests the fix: train each slot on a curated set of
question/answer pairs covering the axiom's facts, then probe with both:
  - TRAIN probes (questions seen during training) — establishes whether
    the slot CAN hold question-conditioned answers at all.
  - HELDOUT probes (paraphrased questions not seen during training) —
    tests generalization.

Conditions per probe:
  A — no axiom (baseline)
  P — full KV prefix (upper bound)
  S — slot trained on Q+A (the experiment)
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.slot_axiom import SlotAxiom, install_slot_hooks, train_slot_qa

TEST_AXIOMS = [
    {
        "name": "BalancePublisher",
        "description": (
            "BalancePublisher is a microservice that polls our crypto "
            "exchange's REST API every 250 milliseconds for sub-account "
            "balances and publishes balance events to the Kafka topic "
            "balances.raw. BalancePublisher has no upstream dependencies."
        ),
        "qa_train": [
            (
                "What is BalancePublisher?",
                "BalancePublisher is a microservice that polls our crypto exchange's REST API for sub-account balances.",
            ),
            ("How often does BalancePublisher poll?", "Every 250 milliseconds."),
            (
                "What does BalancePublisher poll?",
                "It polls our crypto exchange's REST API for sub-account balances.",
            ),
            ("What does BalancePublisher publish?", "Balance events."),
            (
                "Where does BalancePublisher publish?",
                "To the Kafka topic balances.raw.",
            ),
            (
                "Does BalancePublisher have upstream dependencies?",
                "No, BalancePublisher has no upstream dependencies.",
            ),
        ],
        "probes_train": [
            "How often does BalancePublisher poll?",
            "Where does BalancePublisher publish?",
            "What does BalancePublisher poll?",
        ],
        "probes_heldout": [
            "Tell me about BalancePublisher.",
            "Which Kafka topic does BalancePublisher emit to?",
            "How fast is BalancePublisher's poll cycle?",
        ],
    },
    {
        "name": "FluxomService",
        "description": (
            "FluxomService is a data ingestion service that reads from "
            "S3 buckets every 60 seconds, transforms the records into "
            "Parquet format, and writes the output to the Iceberg table "
            "warehouse.fluxom_ingested. It retries failed reads up to 3 times."
        ),
        "qa_train": [
            (
                "What is FluxomService?",
                "FluxomService is a data ingestion service that reads from S3 buckets and writes Parquet to an Iceberg table.",
            ),
            ("How often does FluxomService read from S3?", "Every 60 seconds."),
            ("What format does FluxomService output?", "Parquet format."),
            (
                "Where does FluxomService write?",
                "To the Iceberg table warehouse.fluxom_ingested.",
            ),
            ("How does FluxomService handle failures?", "It retries failed reads up to 3 times."),
            (
                "What does FluxomService transform?",
                "It transforms the records read from S3 into Parquet format.",
            ),
        ],
        "probes_train": [
            "How often does FluxomService read from S3?",
            "What format does FluxomService output?",
            "How does FluxomService handle failures?",
        ],
        "probes_heldout": [
            "Tell me about FluxomService.",
            "What is the polling cadence of FluxomService?",
            "Where does FluxomService land its data?",
        ],
    },
]


@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt: str, max_new: int = 80) -> str:  # noqa: ANN001
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
    parser.add_argument("--slot-width", type=int, default=1024)
    parser.add_argument("--target-layer-frac", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-new", type=int, default=80)
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
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * args.target_layer_frac)
    print(
        f"hidden_size={hidden}  n_layers={n_layers}  target_layer={target_layer}  "
        f"slot_width={args.slot_width}\n"
    )

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]
        qa_train = axiom["qa_train"]

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}")
        print(f"#train Q+A pairs: {len(qa_train)}\n")

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        sa = SlotAxiom.new(
            name=name,
            slot_start=0,
            slot_width=args.slot_width,
            target_layer=target_layer,
            hidden_size=hidden,
        )
        t0 = time.time()
        losses = train_slot_qa(model, tokenizer, sa, qa_train, n_steps=args.n_steps, lr=args.lr)
        print(f"trained slot in {time.time() - t0:.1f}s. loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

        def run_probe_set(label: str, probes: list[str]) -> None:
            print(f"\n--- {label} probes ---")
            for probe in probes:
                full_prompt = f"Q: {probe}\nA:"
                print(f"\n  USER: {probe}")
                out_A = _greedy_generate(model, tokenizer, full_prompt, max_new=args.max_new)
                print(f"    [A no-axiom]:    {out_A.replace(chr(10), ' ').strip()[:300]}")

                out_P = generate_with_prefixes(
                    model, tokenizer, full_prompt, [prefix], args.max_new
                )
                print(f"    [P full-prefix]: {out_P.replace(chr(10), ' ').strip()[:300]}")

                handles = install_slot_hooks(model, [sa])
                try:
                    out_S = _greedy_generate(model, tokenizer, full_prompt, max_new=args.max_new)
                finally:
                    for h in handles:
                        h.remove()
                print(f"    [S slot]:        {out_S.replace(chr(10), ' ').strip()[:300]}")

        run_probe_set("TRAIN", axiom["probes_train"])
        run_probe_set("HELDOUT", axiom["probes_heldout"])


if __name__ == "__main__":
    main()
