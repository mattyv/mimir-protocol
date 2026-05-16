"""APE on a hierarchical 5-axiom DAG (recursive/compositional axioms).

Tests whether APE's q_scale + shared-prefix fix scales beyond 3 axioms
to a real composition tree, where one axiom (DataPipeline) is composed
of sub-axioms that themselves depend on a shared leaf (EventLog).

DAG (depth 3, 5 nodes):

  DataPipeline ─┬─→ KafkaRouter   ──→ EventLog
                ├─→ FeatureStore  ──→ EventLog
                └─→ ModelServer   ──→ FeatureStore
                                   ──→ KafkaRouter

Conditions per prompt:
  A — no prefix (control)
  C — naive concat + RoPE fix (3-prefix regression baseline)
  F[q,sp] — APE: q_scale ∈ {1.2, 1.3, 1.5, 1.7} × shared-prefix ∈
            {"\\n", "### Context:\\n"}
  E — Path 2 joint encoding (always-correct upper bound)

Hallucination guardrail: after each generation we scan the output for
CamelCase / dotted-name identifiers and flag any that are NOT in
`HIERARCHICAL_KNOWN_ENTITIES`. The flag count is printed alongside the
output preview. Lower is better; >0 means the model invented something.
"""

from __future__ import annotations

import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from marker.axiom_registry import (
    HIERARCHICAL_AXIOMS,
    HIERARCHICAL_KNOWN_ENTITIES,
)
from marker.historical.ape import generate_with_ape
from marker.historical.per_block_attention import generate_with_per_block
from marker.prefix_tuning import Prefix, generate_with_prefixes

# Hierarchical prompts probing the DAG at increasing depth.
HIERARCHY_PROMPTS: list[tuple[list[str], str]] = [
    # depth 1: leaf only
    (["event_log"], "What does EventLog store and how is it partitioned?"),
    # depth 2: leaf + immediate consumer
    (
        ["event_log", "feature_store"],
        "How does FeatureStore know about new user events?",
    ),
    # depth 3: full chain on one branch
    (
        ["event_log", "kafka_router", "model_server"],
        "ModelServer needs the current serving model. Walk through how it "
        "ends up there starting from a raw user click.",
    ),
    # full DAG, top-down walkthrough
    (
        ["event_log", "kafka_router", "feature_store", "model_server", "data_pipeline"],
        "Walk through DataPipeline end-to-end: a user click happens, what "
        "components see it and in what order, until ModelServer can use it?",
    ),
    # full DAG, counterfactual that requires the dependency graph
    (
        ["event_log", "kafka_router", "feature_store", "model_server", "data_pipeline"],
        "If EventLog is corrupted and stops accepting writes, which "
        "components in DataPipeline still work and which fail? Be specific "
        "about what each one depends on.",
    ),
    # full DAG, SLA reasoning
    (
        ["event_log", "kafka_router", "feature_store", "model_server", "data_pipeline"],
        "What is DataPipeline's end-to-end SLA, and where does that number come from?",
    ),
    # full DAG, fallback path
    (
        ["event_log", "kafka_router", "feature_store", "model_server", "data_pipeline"],
        "A user has never been seen before, so FeatureStore has no key for "
        "them. What happens when ModelServer gets a request for that user?",
    ),
]


# Regex: CamelCase identifiers (e.g. KafkaRouter, EventLog) and
# dotted/snake topic names (events.raw, feat:user). We avoid matching
# "ML", "AI" etc. by requiring at least 6 chars.
_NAMED_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+){2,}\b"  # CamelCase, ≥2 segments (KafkaRouter)
    r"|\bevents\.[a-z]+\b"  # dotted topic names (events.raw)
    r"|\bfeat:\{?[a-z_]+\}?\b"  # redis key pattern
)


def hallucination_flags(text: str) -> tuple[int, list[str]]:
    """Return (count, list of unique flagged tokens) — entities in `text`
    that are NOT in HIERARCHICAL_KNOWN_ENTITIES.
    """
    matches = _NAMED_ENTITY_RE.findall(text)
    seen: set[str] = set()
    flagged: list[str] = []
    for m in matches:
        if m in HIERARCHICAL_KNOWN_ENTITIES:
            continue
        if m in seen:
            continue
        seen.add(m)
        flagged.append(m)
    return len(flagged), flagged


def _build(model, tokenizer, axiom_key: str, n_tokens: int, target_layers):  # noqa: ANN001
    cfg = HIERARCHICAL_AXIOMS[axiom_key]
    return Prefix.from_description(
        model,
        tokenizer,
        cfg["description"],
        max_tokens=n_tokens,
        target_layers=target_layers,
    )


@torch.no_grad()
def _generate_with_joint_encoding(
    model,  # noqa: ANN001
    tokenizer,
    prompt: str,
    descriptions: list[str],
    max_new: int = 180,
) -> str:
    """Path 2: concatenate descriptions + run a fresh prefill, then decode."""
    device = next(model.parameters()).device
    joint_text = "\n\n".join(descriptions)
    joint_ids = tokenizer(joint_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(joint_ids, past_key_values=DynamicCache(), use_cache=True)
    cache: DynamicCache = out.past_key_values

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, nxt], dim=1)
    if int(nxt.item()) == tokenizer.eos_token_id:
        return ""
    for _ in range(max_new - 1):
        out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        full_ids = torch.cat([full_ids, nxt], dim=1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
    new_ids = full_ids[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--n-prefix-tokens", type=int, default=48)
    parser.add_argument("--max-new", type=int, default=180)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument(
        "--q-scales",
        type=float,
        nargs="+",
        default=[1.2, 1.3, 1.5, 1.7],
        help="Sweep these q_scale values for APE.",
    )
    parser.add_argument(
        "--shared-prefixes",
        type=str,
        nargs="+",
        default=["\n", "### Context:\n"],
        help="Sweep these shared-prefix texts (one extra Modal-friendly "
        "encoding: pass strings with literal \\n which we'll replace).",
    )
    parser.add_argument(
        "--combiners",
        type=str,
        nargs="+",
        default=["uniform", "cosine"],
        choices=["uniform", "lse", "cosine"],
        help="Per-block attention combiners to sweep (G conditions).",
    )
    parser.add_argument(
        "--only-3plus",
        action="store_true",
        help="Skip 1- and 2-prefix prompts (depth-1 / depth-2 in this DAG).",
    )
    args = parser.parse_args()

    # Allow Modal-friendly pass-through where '\n' arrives as the literal
    # 2-char string "\n" — convert to a real newline.
    shared_prefixes = [sp.replace("\\n", "\n") for sp in args.shared_prefixes]

    torch.manual_seed(0)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")
    print(f"q_scales: {args.q_scales}")
    print(f"shared_prefixes: {[repr(sp) for sp in shared_prefixes]}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    all_keys = sorted(HIERARCHICAL_AXIOMS.keys())
    prefixes: dict[str, Prefix] = {}
    descriptions: dict[str, str] = {}
    print(f"=== building prefixes for {len(all_keys)} hierarchical axioms ===")
    for k in all_keys:
        descriptions[k] = HIERARCHICAL_AXIOMS[k]["description"]
        t0 = time.time()
        prefixes[k] = _build(model, tokenizer, k, args.n_prefix_tokens, args.target_layers)
        print(f"  {k}: {time.time() - t0:.1f}s")
    print()

    def fmt(p: str) -> str:
        if not args.use_chat:
            return p
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return p

    def run_one(keys: list[str], prompt: str) -> None:
        formatted = fmt(prompt)
        loaded = [prefixes[k] for k in keys]
        descs = [descriptions[k] for k in keys]

        def timed(fn) -> tuple[str, float]:  # noqa: ANN001
            t0 = time.time()
            out = fn()
            return out, time.time() - t0

        n = len(keys)
        rows: list[tuple[str, str, float, int, list[str]]] = []

        def record(label: str, out: str, dt: float) -> None:
            n_hall, flags = hallucination_flags(out)
            rows.append((label, out, dt, n_hall, flags))

        # A — no prefix
        out, dt = timed(
            lambda: generate_with_prefixes(model, tokenizer, formatted, [], args.max_new)
        )
        record("A no-prefix", out, dt)
        # C — RoPE-fix
        out, dt = timed(
            lambda: generate_with_prefixes(
                model, tokenizer, formatted, loaded, args.max_new, rope_correct=True
            )
        )
        record("C rope-fix ", out, dt)
        # F — APE q_scale × shared_prefix sweep
        for sp_idx, sp in enumerate(shared_prefixes):
            sp_label = f"sp{sp_idx}"
            for qs in args.q_scales:
                qs_local, sp_local = qs, sp
                out, dt = timed(
                    lambda qs=qs_local, sp=sp_local: generate_with_ape(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=formatted,
                        prefixes=loaded,
                        shared_prefix_text=sp,
                        q_scale=qs,
                        max_new=args.max_new,
                        rope_correct=True,
                    )
                )
                record(f"F APE(q={qs:.1f},{sp_label})", out, dt)
        # G — per-block attention sweep (custom SDPA)
        for combiner in args.combiners:
            comb_local = combiner
            out, dt = timed(
                lambda c=comb_local: generate_with_per_block(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=formatted,
                    prefixes=loaded,
                    combiner=c,
                    max_new=args.max_new,
                    rope_correct=True,
                )
            )
            record(f"G per-block({combiner})", out, dt)
        # E — Path 2
        out, dt = timed(
            lambda: _generate_with_joint_encoding(model, tokenizer, formatted, descs, args.max_new)
        )
        record("E joint-enc", out, dt)

        print(f"\n[loaded: {' + '.join(keys)}] (n={n})")
        print(f"USER: {prompt}")
        for label, out, dt, n_hall, flags in rows:
            preview = out.replace(chr(10), " ").strip()[:500]
            hall_tag = f"[hall={n_hall}]" if n_hall == 0 else f"[HALL={n_hall}: {flags[:5]}]"
            print(f"  [{label}] ({dt:5.1f}s) {hall_tag}: {preview}")

    print("\n" + "#" * 78)
    print("# HIERARCHICAL DAG: DataPipeline = {EventLog, KafkaRouter, FeatureStore, ModelServer}")
    print("#" * 78)
    for keys, prompt in HIERARCHY_PROMPTS:
        if args.only_3plus and len(keys) < 3:
            continue
        run_one(keys, prompt)

    print("\n" + "=" * 78)
    print("# Shared-prefix legend:")
    for sp_idx, sp in enumerate(shared_prefixes):
        print(f"#   sp{sp_idx} = {sp!r}")
    print("=" * 78)


if __name__ == "__main__":
    main()
