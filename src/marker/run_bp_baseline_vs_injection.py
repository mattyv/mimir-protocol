"""Direct baseline-vs-injection comparison for Balance Publisher.

Asks the same prompts twice — once with no injection (so we see what the
model says cold), once with the AutoInjector wired up exactly the way
the chat interface does. Lets us measure the actual delta the technique
produces on this hardest case (stolen-words on direct-definition prompts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from marker.auto_injector import AutoInjector
from marker.axiom_plan import build_axiom_plan
from marker.run_injection import QwenInjector
from marker.vector_builder import make_vector_builder

ROOT = Path(__file__).resolve().parents[2]

# Range of prompt structures from worst-case (definition) to best-case
# (in-context use), so we see where the boundary actually sits.
PROMPTS = [
    # The hardest: direct definition.
    "What is a balance publisher?",
    "Define balance publisher in one sentence.",
    # In between: usage but with definitional flavour.
    "Tell me about balance publisher.",
    "Why is balance publisher important?",
    # Best case: in-context, operational.
    "If our balance publisher goes down, what's the immediate effect on the trading system?",
    "Walk me through what balance publisher does between the exchange and our risk module.",
    "Explain balance publisher to a junior engineer joining the trading team.",
]


def _load(path: Path) -> list[str]:
    return json.loads(path.read_text())["positives"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max-new", type=int, default=120)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  model: {args.model_name}\n")

    qwen = QwenInjector(args.model_name, layer=20, device=device)
    model_layers = qwen.model.config.num_hidden_layers

    # Same setup as the chat interface — Balance Publisher with auto stack.
    paraphrases = _load(ROOT / "data" / "balance_publisher_paraphrases.json")
    builder = make_vector_builder(
        qwen,
        paraphrases=paraphrases,
        term="Balance Publisher",
        term_variants=["Balance Publisher", "balance publisher"],
        target_tokens=[],
    )
    plan = build_axiom_plan(
        term="Balance Publisher",
        term_variants=["Balance Publisher", "balance publisher"],
        paraphrases=paraphrases,
        model_layers=model_layers,
        vector_builder=builder,
    )
    if "steer" in plan.mechanisms and plan.target_tokens:
        real_builder = make_vector_builder(
            qwen,
            paraphrases=paraphrases,
            term="Balance Publisher",
            term_variants=["Balance Publisher", "balance publisher"],
            target_tokens=plan.target_tokens,
        )
        plan.mechanisms["steer"]["vector"] = real_builder(
            "steer", plan.mechanisms["steer"]["layer"]
        )
    print(f"plan stack: {list(plan.mechanisms.keys())}")
    for kind, m in plan.mechanisms.items():
        print(f"  {kind}: layer={m['layer']}  α={m['alpha']}")
    print(f"target tokens: {plan.target_tokens}")
    print()

    # Show what each injected vector "says" by projecting through the
    # unembedding matrix (logit lens). This tells us whether the vector
    # carries the meaning we think it does, BEFORE asking the full model
    # to use it.
    base = qwen.model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
    final_norm = base.model.norm if hasattr(base.model, "norm") else None

    @torch.no_grad()
    def top_tokens(v, k: int = 15) -> list[str]:  # noqa: ANN001
        device = next(qwen.model.parameters()).device
        x = torch.tensor(v, dtype=torch.float32, device=device)
        if final_norm is not None:
            x = final_norm(x.unsqueeze(0)).squeeze(0)
        logits = lm_head(x)
        top = torch.topk(logits, k * 2)
        out: list[str] = []
        for idx in top.indices.tolist():
            tok = qwen.tokenizer.decode([idx]).strip()
            if tok and any(c.isalpha() for c in tok):
                out.append(tok)
            if len(out) >= k:
                break
        return out

    print("=== what each injected vector projects to (top tokens via unembedding) ===")
    for kind, m in plan.mechanisms.items():
        toks = top_tokens(m["vector"])
        print(f"  {kind:>8s} (L{m['layer']:>2d}, α={m['alpha']:>4.0f}): {', '.join(toks)}")
    print()

    inj_off = AutoInjector(qwen.model, qwen.tokenizer, plans=[])
    inj_on = AutoInjector(qwen.model, qwen.tokenizer, plans=[plan])

    for prompt in PROMPTS:
        print("=" * 78)
        print(f"USER: {prompt}")
        print()
        baseline = inj_off.generate(prompt, max_new_tokens=args.max_new)
        injected = inj_on.generate(prompt, max_new_tokens=args.max_new)
        print(f"  [baseline ]: {baseline.replace(chr(10), ' ').strip()[:400]}")
        print(f"  [injected ]: {injected.replace(chr(10), ' ').strip()[:400]}")
        # Highlight where they diverge.
        if baseline.strip() == injected.strip():
            print("  [delta    ]: IDENTICAL")
        else:
            # Find first divergence char position
            for i, (a, b) in enumerate(zip(baseline, injected)):
                if a != b:
                    head = baseline[:i].replace("\n", " ")[-60:]
                    print(f"  [delta    ]: differ at char {i}, after '...{head}'")
                    break
        print()


if __name__ == "__main__":
    main()
