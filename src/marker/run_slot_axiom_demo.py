"""Slot-axiom experiment: per-axiom MLP-trained slots in the residual stream.

For each axiom in the test set:
  1. Train a slot vector (one parameter per axiom) to make the frozen
     model reproduce the axiom's description when conditioned on
     "Description: ". Loss = next-token cross-entropy on the description.
  2. Probe with standard axiom questions ("What does X do?", "What's X's
     polling interval?", ...).
  3. Compare against:
       A — no axiom loaded (baseline)
       P — full KV prefix loaded (current approach, upper bound)
       S — slot trained, then injected at decode time (the experiment)

If S matches A on irrelevant queries and approaches P on the axiom's
own queries, the slot mechanism transmits useful information through
the model's downstream layers.

This is the first real test of whether the model's existing weights can
READ from a designated dimension slot in the residual stream.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.slot_axiom import SlotAxiom, install_slot_hooks, train_slot

# Single-axiom test set: name, description, probe questions.
TEST_AXIOMS = [
    {
        "name": "BalancePublisher",
        "description": (
            "BalancePublisher is a microservice that polls our crypto "
            "exchange's REST API every 250 milliseconds for sub-account "
            "balances and publishes balance events to the Kafka topic "
            "balances.raw. BalancePublisher has no upstream dependencies."
        ),
        "probes": [
            "What does BalancePublisher do?",
            "How often does BalancePublisher poll?",
            "Which Kafka topic does BalancePublisher publish to?",
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
        "probes": [
            "What does FluxomService do?",
            "How often does FluxomService read from S3?",
            "What format does FluxomService output to?",
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
    parser.add_argument("--slot-widths", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--target-layer-frac", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=300)
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
    print(f"hidden_size={hidden}  n_layers={n_layers}  target_layer={target_layer}\n")

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]
        probes = axiom["probes"]

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}\n")

        # P — full KV prefix as upper bound
        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        for slot_width in args.slot_widths:
            print(f"\n--- slot_width={slot_width} ---")
            sa = SlotAxiom.new(
                name=name,
                slot_start=0,
                slot_width=slot_width,
                target_layer=target_layer,
                hidden_size=hidden,
            )
            t0 = time.time()
            losses = train_slot(model, tokenizer, sa, desc, n_steps=args.n_steps, lr=args.lr)
            print(
                f"  trained slot in {time.time() - t0:.1f}s. "
                f"loss: {losses[0]:.3f} -> {losses[-1]:.3f}"
            )

            handles_S = install_slot_hooks(model, [sa])
            try:
                for probe in probes:
                    print(f"\n  USER: {probe}")
                    # A — no axiom
                    handles_S_save = handles_S
                    for h in handles_S_save:
                        h.remove()
                    out_A = _greedy_generate(model, tokenizer, probe, max_new=args.max_new)
                    print(f"    [A no-axiom]:    {out_A.replace(chr(10), ' ').strip()[:240]}")

                    # P — full prefix
                    out_P = generate_with_prefixes(model, tokenizer, probe, [prefix], args.max_new)
                    print(f"    [P full-prefix]: {out_P.replace(chr(10), ' ').strip()[:240]}")

                    # S — slot only
                    handles_S = install_slot_hooks(model, [sa])
                    out_S = _greedy_generate(model, tokenizer, probe, max_new=args.max_new)
                    print(f"    [S slot]:        {out_S.replace(chr(10), ' ').strip()[:240]}")
            finally:
                for h in handles_S:
                    h.remove()


if __name__ == "__main__":
    main()
