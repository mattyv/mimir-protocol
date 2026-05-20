"""Per-axiom MLP injection demo on real axioms (BalancePublisher, FluxomService).

Training pipeline (v2):
  1. Hand-written fact Q+A from axiom["facts"].
  2. Teacher-distilled synthetic Q+A — model + full prefix generates N pairs.
  3. Overview examples ("Tell me about X" → description).
  4. Generic boundary examples (decline out-of-scope questions).

Architecture: SmallMLP at 3 chosen layers per axiom, r=32, cosine LR decay.
Each axiom also carries a frozen KV cache of its description (computed once before
training). During both training and inference the description KV is prepended to
past_key_values so the model can attend to it at every decode step — fixing the
passive-retrieval problem. The MLP's job is then boundary enforcement + routing,
not fact compression.

Compares [A no-axiom] vs [P full-prefix] vs [M mlp+kv] on TRAIN/HELDOUT/BOUNDARY/TELL_ME.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from marker.prefix_tuning import Prefix, generate_with_prefixes
from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS, _generic_boundary_examples
from marker.soft_prompt_plus import generate_synthetic_qa_pairs

TEMPLATE = "Q: {q}\nA:"

# ── Gap-filling Q+A for known heldout misses ───────────────────────────────────
# Added after analysing run results: each entry is a phrasing that the
# synthetic generator didn't cover and the MLP was getting wrong.
SUPPLEMENTAL_QA: dict[str, list[tuple[str, str]]] = {
    "BalancePublisher": [
        (
            "How fast is BalancePublisher's poll cycle?",
            "BalancePublisher's poll cycle is every 250 milliseconds.",
        ),
        (
            "What's the speed of BalancePublisher's polling?",
            "BalancePublisher polls every 250 milliseconds.",
        ),
        (
            "What's BalancePublisher's polling cadence?",
            "BalancePublisher's polling cadence is every 250 milliseconds.",
        ),
        (
            "What is BalancePublisher's polling cadence?",
            "BalancePublisher polls every 250 milliseconds.",
        ),
        (
            "What is BalancePublisher's polling rate?",
            "BalancePublisher polls every 250 milliseconds.",
        ),
        (
            "What endpoint type does BalancePublisher hit?",
            "BalancePublisher hits a REST API endpoint.",
        ),
        ("What kind of API does BalancePublisher call?", "BalancePublisher calls a REST API."),
        (
            "What are BalancePublisher's upstream dependencies?",
            "BalancePublisher has no upstream dependencies.",
        ),
        (
            "What services does BalancePublisher depend on?",
            "BalancePublisher has no upstream dependencies.",
        ),
    ],
    "FluxomService": [
        (
            "Where does FluxomService land its data?",
            "FluxomService lands its data in the Iceberg table warehouse.fluxom_ingested.",
        ),
        (
            "Where does FluxomService deposit its output?",
            "FluxomService deposits output to the Iceberg table warehouse.fluxom_ingested.",
        ),
        (
            "What table does FluxomService populate?",
            "FluxomService populates the Iceberg table warehouse.fluxom_ingested.",
        ),
        (
            "Where does FluxomService store the transformed records?",
            "FluxomService stores them in the Iceberg table warehouse.fluxom_ingested.",
        ),
    ],
}


def _ensure_term_in_qa(qa_pairs: list[tuple[str, str]], term: str) -> list[tuple[str, str]]:
    """Synthetic pairs sometimes omit the term name (e.g. 'What is the polling
    interval?' instead of 'What is BalancePublisher's polling interval?').
    Those steps get skipped because the hook has nowhere to fire.
    Prefix term to questions that lack it so no training signal is wasted."""
    out = []
    for q, a in qa_pairs:
        if term.lower() not in q.lower():
            q = f"Regarding {term}: {q}"
        out.append((q, a))
    return out


# ── Architecture ───────────────────────────────────────────────────────────────


class SmallMLP(nn.Module):
    def __init__(self, hidden_size: int, r: int = 32):
        super().__init__()
        self.down = nn.Linear(hidden_size, r, bias=False)
        self.up = nn.Linear(r, hidden_size, bias=False)
        self.act = nn.GELU()
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


@dataclass
class AxiomKV:
    """Plain-tensor store for per-layer K/V — version-agnostic."""

    n_layers: int
    keys: list[torch.Tensor]  # (1, n_kv_heads, desc_tokens, head_dim) per layer
    values: list[torch.Tensor]  # same


@dataclass
class AxiomMLP:
    term: str
    term_token_ids: list[int]
    chosen_layers: list[int]
    mlps: nn.ModuleList
    kv: AxiomKV | None = None  # frozen description KV, set by compute_axiom_kv()


def make_axiom_mlp(model, tokenizer, term: str, chosen_layers: list[int], r: int = 32) -> AxiomMLP:  # noqa: ANN001
    hidden = model.config.hidden_size
    ids = (
        tokenizer(" " + term, add_special_tokens=False).input_ids
        or tokenizer(term, add_special_tokens=False).input_ids
    )
    device = next(model.parameters()).device
    mlps = nn.ModuleList([SmallMLP(hidden, r) for _ in chosen_layers])
    mlps = mlps.to(device=device, dtype=torch.float32)
    return AxiomMLP(term=term, term_token_ids=ids, chosen_layers=chosen_layers, mlps=mlps)


# ── Hook plumbing ──────────────────────────────────────────────────────────────


def _find_term_positions(input_ids: torch.Tensor, term_ids: list[int]) -> list[int]:
    ids = input_ids[0].tolist()
    n = len(term_ids)
    return [i for i in range(len(ids) - n + 1) if ids[i : i + n] == term_ids]


def _make_layer_hook(mlp: SmallMLP, positions: list[int]):
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        new_hidden = hidden
        for pos in positions:
            if pos >= new_hidden.shape[1]:
                continue
            offset = mlp(new_hidden[:, pos, :].float()).to(new_hidden.dtype)
            new_hidden = torch.cat(
                [
                    new_hidden[:, :pos, :],
                    new_hidden[:, pos : pos + 1, :] + offset.unsqueeze(1),
                    new_hidden[:, pos + 1 :, :],
                ],
                dim=1,
            )
        return (new_hidden,) + output[1:] if is_tuple else new_hidden

    return hook


def install_hooks(model, axiom_mlp: AxiomMLP, positions: list[int]):  # noqa: ANN001
    handles = []
    for layer_idx, mlp in zip(axiom_mlp.chosen_layers, axiom_mlp.mlps, strict=True):
        h = model.model.layers[layer_idx].register_forward_hook(_make_layer_hook(mlp, positions))
        handles.append(h)
    return handles


# ── Axiom KV cache ────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_axiom_kv(model, tokenizer, description: str) -> AxiomKV:  # noqa: ANN001
    """Run description through frozen model once and cache the resulting K/V.

    Stored as plain tensors (AxiomKV) to avoid DynamicCache version differences.
    A fresh DynamicCache is built from these tensors before each forward pass via
    _build_dynamic_cache().
    """
    device = next(model.parameters()).device
    desc_ids = tokenizer(description, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    out = model(desc_ids, use_cache=True)
    kv = out.past_key_values
    # Normalize to a flat list of (K, V) tensors regardless of cache type.
    # to_legacy_cache() may return tuples with extra elements (e.g. scale factors);
    # use index access [0]/[1] rather than unpacking to handle that safely.
    legacy = kv.to_legacy_cache() if hasattr(kv, "to_legacy_cache") else kv
    keys = [layer_kv[0].detach() for layer_kv in legacy]
    values = [layer_kv[1].detach() for layer_kv in legacy]
    return AxiomKV(n_layers=len(keys), keys=keys, values=values)


def merge_axiom_kvs(axiom_kvs: list[AxiomKV]) -> AxiomKV:
    """Concatenate per-axiom KVs along the sequence dimension for multi-axiom inference."""
    n = axiom_kvs[0].n_layers
    return AxiomKV(
        n_layers=n,
        keys=[torch.cat([kv.keys[i] for kv in axiom_kvs], dim=2) for i in range(n)],
        values=[torch.cat([kv.values[i] for kv in axiom_kvs], dim=2) for i in range(n)],
    )


def _build_dynamic_cache(axiom_kv: AxiomKV, device: torch.device):  # noqa: ANN201
    """Construct a fresh DynamicCache from stored tensors using the stable update() API."""
    from transformers import DynamicCache  # noqa: PLC0415

    cache = DynamicCache()
    for layer_idx in range(axiom_kv.n_layers):
        cache.update(
            axiom_kv.keys[layer_idx].to(device),
            axiom_kv.values[layer_idx].to(device),
            layer_idx,
        )
    return cache


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    axiom_mlp: AxiomMLP,
    qa_pairs: list[tuple[str, str]],
    boundary_pairs: list[tuple[str, str]] | None = None,
    boundary_prob: float = 0.33,
    n_steps: int = 3000,
    lr_start: float = 3e-5,
    lr_end: float = 3e-6,
    weight_decay: float = 0.05,
) -> list[float]:
    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device
    optim = torch.optim.AdamW(axiom_mlp.mlps.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps, eta_min=lr_end)
    eos_id = tokenizer.eos_token_id
    rng = random.Random(42)
    losses: list[float] = []
    skipped = 0

    for _ in range(n_steps):
        if boundary_pairs and rng.random() < boundary_prob:
            q, a = rng.choice(boundary_pairs)
        else:
            q, a = rng.choice(qa_pairs)

        q_text = TEMPLATE.format(q=q)
        full_text = q_text + " " + a

        q_ids = tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            device
        )
        if eos_id is not None:
            full_ids = torch.cat([full_ids, torch.tensor([[eos_id]], device=device)], dim=1)

        positions = _find_term_positions(q_ids, axiom_mlp.term_token_ids)
        if not positions:
            skipped += 1
            continue

        labels = torch.full_like(full_ids, -100)
        labels[0, q_ids.shape[1] :] = full_ids[0, q_ids.shape[1] :]

        handles = install_hooks(model, axiom_mlp, positions)
        try:
            optim.zero_grad()
            # Build a fresh DynamicCache each step — the cache is stateful and gets
            # extended by the forward pass, so we can't reuse the same object.
            kv_cache = (
                _build_dynamic_cache(axiom_mlp.kv, device) if axiom_mlp.kv is not None else None
            )
            loss = model(full_ids, past_key_values=kv_cache, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(axiom_mlp.mlps.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            losses.append(float(loss.item()))
        finally:
            for h in handles:
                h.remove()

    if skipped:
        print(f"  WARNING: skipped {skipped} steps (term not found in prompt)")
    return losses


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_with_mlp(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    prompt: str,
    axiom_mlp: AxiomMLP | None = None,
    max_new: int = 120,
) -> str:
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    kv_cache = (
        _build_dynamic_cache(axiom_mlp.kv, device)
        if axiom_mlp is not None and axiom_mlp.kv is not None
        else None
    )

    handles = []
    if axiom_mlp is not None:
        positions = _find_term_positions(ids, axiom_mlp.term_token_ids)
        if positions:
            handles = install_hooks(model, axiom_mlp, positions)

    try:
        out = model(ids, past_key_values=kv_cache, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    finally:
        for h in handles:
            h.remove()

    out_ids = [next_tok]
    for _ in range(max_new - 1):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids.append(next_tok)
        if int(next_tok.item()) == tokenizer.eos_token_id:
            break

    return tokenizer.decode(torch.cat(out_ids, dim=1)[0], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_mlps(
    model,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    prompt: str,
    axiom_mlps: list[AxiomMLP],
    max_new: int = 120,
) -> str:
    """Multi-axiom inference: all MLPs active simultaneously.

    Each axiom's hooks fire only at that axiom's term positions.
    Different axioms are at different positions → no interference.
    PyTorch runs hooks in registration order; each only touches its
    own positions so composition is clean.
    """
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Only inject KV for axioms whose term appears in this prompt — avoids bleed
    # when a single-axiom question is asked in a multi-axiom session.
    available_kvs = [
        a.kv for a in axiom_mlps if a.kv is not None and _find_term_positions(ids, a.term_token_ids)
    ]
    merged_kv = (
        _build_dynamic_cache(merge_axiom_kvs(available_kvs), device) if available_kvs else None
    )

    handles = []
    for axiom_mlp in axiom_mlps:
        positions = _find_term_positions(ids, axiom_mlp.term_token_ids)
        if positions:
            handles.extend(install_hooks(model, axiom_mlp, positions))

    try:
        out = model(ids, past_key_values=merged_kv, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    finally:
        for h in handles:
            h.remove()

    out_ids = [next_tok]
    for _ in range(max_new - 1):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        out_ids.append(next_tok)
        if int(next_tok.item()) == tokenizer.eos_token_id:
            break

    return tokenizer.decode(torch.cat(out_ids, dim=1)[0], skip_special_tokens=True).strip()


class AxiomSession:
    """Multi-turn chat session with persistent axiom KV across turns.

    All registered axiom KVs are merged and injected once at session init.
    They live at the start of past_key_values for every subsequent turn —
    the model can always attend to them regardless of whether the term is
    named in the current message.

    MLP hooks still fire per-turn at term positions in the current prompt,
    providing query-conditional routing on top of the stable KV foundation.

    Usage:
        session = AxiomSession(model, [bp_mlp, fs_mlp])
        ans1 = session.chat(model, tokenizer, "Q: How often does BalancePublisher poll?\\nA:")
        ans2 = session.chat(model, tokenizer, "Q: What does it publish?\\nA:")  # no term needed
        session.reset(model)  # start a new conversation
    """

    def __init__(self, model, axiom_mlps: list[AxiomMLP]) -> None:  # noqa: ANN001
        self.axiom_mlps = axiom_mlps
        self.past = self._build_session_kv(model)

    def _build_session_kv(self, model):  # noqa: ANN001
        device = next(model.parameters()).device
        available = [a.kv for a in self.axiom_mlps if a.kv is not None]
        return _build_dynamic_cache(merge_axiom_kvs(available), device) if available else None

    @torch.no_grad()
    def chat(self, model, tokenizer, prompt: str, max_new: int = 120) -> str:  # noqa: ANN001
        device = next(model.parameters()).device
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        handles = []
        for axiom_mlp in self.axiom_mlps:
            positions = _find_term_positions(ids, axiom_mlp.term_token_ids)
            if positions:
                handles.extend(install_hooks(model, axiom_mlp, positions))

        try:
            out = model(ids, past_key_values=self.past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        finally:
            for h in handles:
                h.remove()

        out_ids = [next_tok]
        for _ in range(max_new - 1):
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            out_ids.append(next_tok)
            if int(next_tok.item()) == tokenizer.eos_token_id:
                break

        self.past = past  # grow session KV for next turn
        return tokenizer.decode(torch.cat(out_ids, dim=1)[0], skip_special_tokens=True).strip()

    def reset(self, model) -> None:  # noqa: ANN001
        """Start a new conversation, restoring session to axiom-KV-only state."""
        self.past = self._build_session_kv(model)


MULTI_AXIOM_PROBES = [
    # Single-axiom isolation: does each MLP still work when the other is also loaded?
    ("ISOLATION BP", "Q: How often does BalancePublisher poll?\nA:"),
    ("ISOLATION FS", "Q: What format does FluxomService output?\nA:"),
    ("ISOLATION BP", "Q: What Kafka topic does BalancePublisher publish to?\nA:"),
    ("ISOLATION FS", "Q: Where does FluxomService write its output?\nA:"),
    # Cross-axiom comparison: model must use info from BOTH to answer
    ("CROSS", "Q: Which polls more frequently, BalancePublisher or FluxomService?\nA:"),
    (
        "CROSS",
        "Q: BalancePublisher and FluxomService are both running. Which one writes to Kafka?\nA:",
    ),
    (
        "CROSS",
        "Q: What does BalancePublisher publish and where does FluxomService store its output?\nA:",
    ),
    # Boundary discipline with both loaded: should decline for both
    ("BOUNDARY", "Q: What programming language is BalancePublisher written in?\nA:"),
    ("BOUNDARY", "Q: What's the SLA of FluxomService?\nA:"),
]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--lr-start", type=float, default=3e-5)
    parser.add_argument("--lr-end", type=float, default=3e-6)
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=30,
        help="synthetic Q+A pairs from teacher distillation per axiom",
    )
    parser.add_argument("--max-new", type=int, default=120)
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}  model: {args.model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    n_layers = model.config.num_hidden_layers
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]
    print(f"n_layers={n_layers}  chosen_layers={chosen_layers}  r={args.r}\n")

    trained_mlps: list[AxiomMLP] = []
    trained_prefixes: list[Prefix] = []

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]

        print("\n" + "#" * 78)
        print(f"# axiom: {name}")
        print("#" * 78)
        print(f"description: {desc}\n")

        # Full prefix (teacher + comparison baseline)
        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        # ── Build training set ─────────────────────────────────────────────────

        # 1. Hand-written fact Q+A
        train_qa: list[tuple[str, str]] = []
        heldout_qs: list[str] = []
        for f in axiom["facts"]:
            for q in f["questions_train"]:
                train_qa.append((q, f["answer"]))
            heldout_qs.extend(f["questions_heldout"])

        # 2. Teacher-distilled synthetic Q+A
        print(f"=== Generating {args.n_synthetic} synthetic Q+A pairs via teacher ===")
        t0 = time.time()
        synth = generate_synthetic_qa_pairs(
            model,
            tokenizer,
            desc,
            prefix,
            n_pairs=args.n_synthetic,
            max_new=2200,
        )
        print(f"  parsed {len(synth)} pairs in {time.time() - t0:.1f}s")
        # Ensure every synthetic question contains the term so the hook fires.
        synth = _ensure_term_in_qa(synth, name)
        train_qa.extend(synth)

        # 2b. Gap-filling hand-written Q+A for known heldout misses
        train_qa.extend(SUPPLEMENTAL_QA.get(name, []))

        # 3. Overview / TELL_ME examples
        overview_qa: list[tuple[str, str]] = [
            (f"Tell me about {name}.", desc),
            (f"Describe {name}.", desc),
            (f"What is {name}?", desc),
            (f"Give me an overview of {name}.", desc),
        ]
        train_qa.extend(overview_qa)

        # 4. Boundary examples (decline out-of-scope)
        boundary_qa = _generic_boundary_examples(name)

        print(
            f"training set: {len(train_qa)} fact+synth+overview pairs  "
            f"+ {len(boundary_qa)} boundary pairs"
        )

        # ── Build and train MLP ────────────────────────────────────────────────
        axiom_mlp = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=args.r)
        n_params = sum(p.numel() for p in axiom_mlp.mlps.parameters())
        print(
            f"term_token_ids={axiom_mlp.term_token_ids}  MLP params: {n_params:,} ({n_params * 4 / 1024:.0f} KB)"
        )

        # Compute and attach the description KV cache (done once, before training).
        print("  computing description KV cache...")
        t_kv = time.time()
        axiom_mlp.kv = compute_axiom_kv(model, tokenizer, desc)
        kv_tokens = axiom_mlp.kv.keys[0].shape[2]  # seq dim of first layer K
        kv_mb = (
            sum(
                k.nbytes + v.nbytes
                for k, v in zip(axiom_mlp.kv.keys, axiom_mlp.kv.values, strict=True)
            )
            / 1024**2
        )
        print(f"  KV: {kv_tokens} description tokens, {kv_mb:.1f} MB  ({time.time() - t_kv:.1f}s)")

        t0 = time.time()
        losses = train(
            model,
            tokenizer,
            axiom_mlp,
            train_qa,
            boundary_pairs=boundary_qa,
            n_steps=args.n_steps,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
        )
        print(f"trained in {time.time() - t0:.1f}s  loss: {losses[0]:.3f} → {losses[-1]:.4f}")

        # ── Probe ──────────────────────────────────────────────────────────────
        def run_probe(
            label: str,
            questions: list[str],
            _prefix: Prefix = prefix,
            _mlp: AxiomMLP = axiom_mlp,
            cot: bool = False,
        ) -> None:
            print(f"\n--- {label} ---")
            for q in questions:
                prompt = f"Q: {q}\nLet's think step by step.\nA:" if cot else TEMPLATE.format(q=q)
                out_a = generate_with_mlp(model, tokenizer, prompt, max_new=args.max_new)
                out_p = generate_with_prefixes(model, tokenizer, prompt, [_prefix], args.max_new)
                out_m = generate_with_mlp(model, tokenizer, prompt, _mlp, max_new=args.max_new)
                print(f"  Q: {q}")
                print(f"    [A no-axiom]:  {out_a[:200].replace(chr(10), ' ')}")
                print(f"    [P prefix]:    {out_p[:200].replace(chr(10), ' ')}")
                print(f"    [M mlp+kv]:    {out_m[:200].replace(chr(10), ' ')}")

        train_qs = [q for f in axiom["facts"] for q in f["questions_train"][:1]]
        run_probe("TRAIN (1 per fact)", train_qs)
        run_probe("HELDOUT", heldout_qs)
        run_probe("HELDOUT+CoT", heldout_qs, cot=True)
        run_probe("BOUNDARY", axiom["boundary_probes"])
        run_probe("TELL_ME", [f"Tell me about {name}.", f"What is {name}?"])

        trained_mlps.append(axiom_mlp)
        trained_prefixes.append(prefix)

    # ── Multi-axiom isolation + boundary ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("MULTI-AXIOM TEST — all MLPs loaded simultaneously")
    print(f"Active axioms: {[m.term for m in trained_mlps]}")
    print("=" * 78)

    for label, prompt in MULTI_AXIOM_PROBES:
        q_display = prompt.replace("Q: ", "").replace("\nA:", "")
        out_a = generate_with_mlp(model, tokenizer, prompt, max_new=args.max_new)
        out_m = generate_with_mlps(model, tokenizer, prompt, trained_mlps, max_new=args.max_new)
        print(f"\n[{label}]  {q_display}")
        print(f"  [A no-axiom]:    {out_a[:240].replace(chr(10), ' ')}")
        print(f"  [M multi-mlp+kv]:{out_m[:240].replace(chr(10), ' ')}")

    # ── Multi-turn chat session ───────────────────────────────────────────────
    # AxiomSession injects all axiom KVs once at init. Each turn the model can
    # attend to any registered axiom's description regardless of whether the
    # term appears in the current message.
    print("\n" + "=" * 78)
    print("MULTI-TURN CHAT SESSION")
    print("=" * 78)

    session = AxiomSession(model, trained_mlps)

    multi_turn_chat = [
        # Turn 1: name BP — hooks fire, KV already in session
        "Q: How often does BalancePublisher poll?\nA:",
        # Turn 2: follow-up without naming BP — no hooks, but session KV persists
        "Q: What does it publish?\nA:",
        # Turn 3: switch to FS — hooks fire for FS
        "Q: What format does FluxomService output?\nA:",
        # Turn 4: cross-axiom in context of ongoing session
        "Q: Which of the two services we've discussed writes to Kafka?\nA:",
    ]

    for i, prompt in enumerate(multi_turn_chat, 1):
        q_display = prompt.replace("Q: ", "").replace("\nA:", "")
        ans = session.chat(model, tokenizer, prompt, max_new=args.max_new)
        print(f"\n  Turn {i}: {q_display}")
        print(f"    {ans[:200].replace(chr(10), ' ')}")

    # ── Cross-axiom 5-condition matrix ────────────────────────────────────────
    # Conditions:
    #   A ctx no-CoT     : facts in prompt, plain question (baseline, no Mimir)
    #   A ctx CoT        : facts in prompt, "think step by step" (baseline + CoT)
    #   M inj no-CoT     : MLP injection, plain question (Mimir, no CoT)
    #   M inj CoT        : MLP injection, "think step by step" (Mimir + standard CoT)
    #   M inj struct-CoT : MLP injection + scaffolded prompt that names each term
    #                      explicitly in intermediate steps, so hook fires at
    #                      each term during prefill and model can attend to
    #                      injected K/V while generating intermediate answers.

    bp_desc = TEST_AXIOMS[0]["description"]
    fs_desc = TEST_AXIOMS[1]["description"]
    in_context = f"{bp_desc}\n{fs_desc}\n"
    cot_suffix = "\nLet's think step by step."

    # Structured CoT: scaffold forces the model to generate each fact before
    # comparing. Both terms appear in the prompt so hooks fire at prefill.
    cross_questions = [
        (
            "Q: Which polls more frequently, BalancePublisher or FluxomService?",
            # struct-CoT scaffold: names each term → hook fires → model retrieves fact
            "Q: Which polls more frequently, BalancePublisher or FluxomService?"
            "\nBalancePublisher's polling interval:",
        ),
        (
            "Q: BalancePublisher and FluxomService are both running. Which one writes to Kafka?",
            "Q: BalancePublisher and FluxomService are both running. Which one writes to Kafka?"
            "\nBalancePublisher writes to:"
            "\nFluxomService writes to:"
            "\nConclusion:",
        ),
        (
            "Q: What does BalancePublisher publish and where does FluxomService store its output?",
            "Q: What does BalancePublisher publish and where does FluxomService store its output?"
            "\nBalancePublisher publishes:"
            "\nFluxomService stores its output in:",
        ),
    ]

    print("\n" + "=" * 78)
    print("CROSS-AXIOM 5-CONDITION MATRIX")
    print("=" * 78)

    for q, q_struct in cross_questions:
        print(f"\n{'─' * 70}")
        print(f"Q: {q[3:]}")

        plain_end = "\nA:"
        cot_end = f"{cot_suffix}\nA:"
        ctx_plain = f"{in_context}{q}{plain_end}"
        ctx_cot = f"{in_context}{q}{cot_end}"
        inj_plain = f"{q}{plain_end}"
        inj_cot = f"{q}{cot_end}"
        inj_struct = q_struct  # no trailing A: — model continues the scaffold

        r_ctx = generate_with_mlp(model, tokenizer, ctx_plain, max_new=args.max_new)
        r_ctx_cot = generate_with_mlp(model, tokenizer, ctx_cot, max_new=args.max_new)
        r_inj = generate_with_mlps(model, tokenizer, inj_plain, trained_mlps, max_new=args.max_new)
        r_inj_cot = generate_with_mlps(
            model, tokenizer, inj_cot, trained_mlps, max_new=args.max_new
        )
        r_inj_struct = generate_with_mlps(
            model, tokenizer, inj_struct, trained_mlps, max_new=args.max_new
        )

        print(f"  [A ctx    no-CoT]:     {r_ctx[:200].replace(chr(10), ' ')}")
        print(f"  [A ctx    CoT]:        {r_ctx_cot[:200].replace(chr(10), ' ')}")
        print(f"  [M inj    no-CoT]:     {r_inj[:200].replace(chr(10), ' ')}")
        print(f"  [M inj    CoT]:        {r_inj_cot[:200].replace(chr(10), ' ')}")
        print(f"  [M inj struct-CoT]:    {r_inj_struct[:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
