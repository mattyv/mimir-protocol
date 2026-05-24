"""Modal wrapper to run the blend battery on a GPU.

Usage:
  pip install modal && modal setup       # one-time
  modal run modal_blends.py              # default: Qwen2.5-7B on A10G
  modal run modal_blends.py --model Qwen/Qwen2.5-14B --gpu A100

The HF model cache is persisted to a Modal Volume so subsequent runs
skip the download.
"""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.1",
        "transformers>=5.5.0",  # gemma4 support
        "accelerate>=1.0.1",
        "sentencepiece",  # gemma tokenizer
        "numpy<2",
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/src")
    .add_local_dir(str(PROJECT_ROOT / "data"), remote_path="/root/data")
)

hf_cache = modal.Volume.from_name("mimir-hf-cache", create_if_missing=True)
axiom_vol = modal.Volume.from_name("mimir-axioms", create_if_missing=True)

app = modal.App("mimir-blends", image=image)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_blends_big(
    model_name: str,
    axiom: str,
    max_new: int,
    layers: list[int] | None = None,
    extra_args: list[str] | None = None,
) -> str:
    return _run_blends_impl(model_name, axiom, max_new, layers, extra_args)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_blends(model_name: str, axiom: str, max_new: int, layers: list[int] | None = None) -> str:
    return _run_blends_impl(model_name, axiom, max_new, layers)


def _run_blends_impl(
    model_name: str,
    axiom: str,
    max_new: int,
    layers: list[int] | None = None,
    extra_args: list[str] | None = None,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    sys.argv = [
        "run_blends",
        "--model-name",
        model_name,
        "--axiom",
        axiom,
        "--max-new",
        str(max_new),
    ]
    if layers:
        sys.argv += ["--layers", *map(str, layers)]
    if extra_args:
        sys.argv += extra_args

    import io
    from contextlib import redirect_stdout

    from marker.run_blends import main as run_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_probe(model_name: str) -> str:
    """Activation patching probe to find BP hot-spot layers."""
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = ["probe_full_patching", "--model-name", model_name]

    import io
    from contextlib import redirect_stdout

    from marker.probe_full_patching import main as probe_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        probe_main()
    return buf.getvalue()


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-7B",
    axiom: str = "both",
    max_new: int = 60,
) -> None:
    print(f"running blends with {model}, axiom={axiom}, max_new={max_new}")
    output = run_blends.remote(model, axiom, max_new)
    print(output)


@app.local_entrypoint()
def probe(model: str = "Qwen/Qwen2.5-32B") -> None:
    print(f"running BP activation-patching probe on {model}")
    output = run_probe.remote(model)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_gemma_probe(model_name: str) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = ["probe_gemma_patching", "--model-name", model_name]

    import io
    from contextlib import redirect_stdout

    from marker.probe_gemma_patching import main as gp_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        gp_main()
    return buf.getvalue()


@app.local_entrypoint()
def gemma_probe(model: str = "google/gemma-4-31B") -> None:
    print(f"running Gemma activation-patching probe on {model}")
    output = run_gemma_probe.remote(model)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_gemma_iti(
    model_name: str, top_k: int = 16, max_new: int = 60, quote_axiom: bool = False
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_iti_gemma",
        "--model-name",
        model_name,
        "--top-k",
        str(top_k),
        "--max-new",
        str(max_new),
    ]
    if quote_axiom:
        sys.argv.append("--quote-axiom")
    import io
    from contextlib import redirect_stdout

    from marker.run_iti_gemma import main as gi_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        gi_main()
    return buf.getvalue()


@app.local_entrypoint()
def gemma_iti(
    model: str = "google/gemma-4-31B-it", top_k: int = 16, quote_axiom: bool = False
) -> None:
    label = " [quoted axiom]" if quote_axiom else ""
    print(f"running ITI on {model} with top-{top_k} heads{label}")
    output = run_gemma_iti.remote(model, top_k, 60, quote_axiom)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_register_axiom(
    model_name: str, layers_str: str, max_new: int = 50, axiom: str = "bp"
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    layers = layers_str.split(",")
    sys.argv = [
        "register_axiom",
        "--model-name",
        model_name,
        "--layers",
        *layers,
        "--max-new",
        str(max_new),
        "--use-chat",
        "--bf16",
        "--axiom",
        axiom,
    ]
    import io
    from contextlib import redirect_stdout

    from marker.register_axiom import main as ra_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        ra_main()
    return buf.getvalue()


@app.local_entrypoint()
def gemma_register(
    model: str = "google/gemma-4-31B-it", layers: str = "23,53", axiom: str = "bp"
) -> None:
    """Tier 3 closed-form online registration on Gemma 4-IT."""
    print(f"running register_axiom on {model} at layers {layers}, axiom={axiom}")
    output = run_register_axiom.remote(model, layers, 50, axiom)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_combined(
    model_name: str,
    layers_str: str,
    n_steps: int,
    target_count: int,
    max_new: int,
    use_chat: bool = False,
    batch_size: int = 4,
    axioms: str = "",
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    layers = layers_str.split(",")
    sys.argv = [
        "run_combined_demo",
        "--model-name",
        model_name,
        "--layers",
        *layers,
        "--n-steps",
        str(n_steps),
        "--target-count",
        str(target_count),
        "--max-new",
        str(max_new),
        "--batch-size",
        str(batch_size),
    ]
    if axioms:
        sys.argv += ["--axioms", *axioms.split(",")]
    if use_chat:
        sys.argv.append("--use-chat")
    import io
    from contextlib import redirect_stdout

    from marker.run_combined_demo import main as combined_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        combined_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_prefix(
    model_name: str,
    n_steps: int,
    max_new: int,
    n_prefix_tokens: int,
    axioms: str = "",
    target_layers: str = "",
    skip_training: bool = False,
    use_chat: bool = False,
    bleed: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_prefix_demo",
        "--model-name",
        model_name,
        "--n-steps",
        str(n_steps),
        "--max-new",
        str(max_new),
        "--n-prefix-tokens",
        str(n_prefix_tokens),
    ]
    if target_layers:
        sys.argv += ["--target-layers", *target_layers.split(",")]
    if axioms:
        sys.argv += ["--axioms", *axioms.split(",")]
    if skip_training:
        sys.argv.append("--skip-training")
    if use_chat:
        sys.argv.append("--use-chat")
    if bleed:
        sys.argv.append("--bleed")
    import io
    from contextlib import redirect_stdout

    from marker.run_prefix_demo import main as prefix_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        prefix_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_reasoning(
    model_name: str,
    n_prefix_tokens: int,
    max_new: int,
    target_layers: str = "",
    use_chat: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_reasoning_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--max-new",
        str(max_new),
    ]
    if target_layers:
        sys.argv += ["--target-layers", *target_layers.split(",")]
    if use_chat:
        sys.argv.append("--use-chat")
    import io
    from contextlib import redirect_stdout

    from marker.run_reasoning_demo import main as reasoning_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        reasoning_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_chain(
    model_name: str,
    n_prefix_tokens: int,
    max_new: int,
    target_layers: str = "",
    use_chat: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_chain_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--max-new",
        str(max_new),
    ]
    if target_layers:
        sys.argv += ["--target-layers", *target_layers.split(",")]
    if use_chat:
        sys.argv.append("--use-chat")
    import io
    from contextlib import redirect_stdout

    from marker.run_chain_demo import main as chain_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        chain_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_slot_axiom(
    model_name: str,
    slot_widths: str = "256,512,1024",
    target_layer_frac: float = 0.5,
    n_steps: int = 300,
    lr: float = 0.05,
    max_new: int = 80,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_slot_axiom_demo",
        "--model-name",
        model_name,
        "--slot-widths",
        *slot_widths.split(","),
        "--target-layer-frac",
        str(target_layer_frac),
        "--n-steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_slot_axiom_demo import main as slot_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        slot_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_slot_axiom_qa(
    model_name: str,
    slot_width: int = 1024,
    target_layer_frac: float = 0.5,
    n_steps: int = 400,
    lr: float = 0.05,
    max_new: int = 80,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_slot_axiom_qa_demo",
        "--model-name",
        model_name,
        "--slot-width",
        str(slot_width),
        "--target-layer-frac",
        str(target_layer_frac),
        "--n-steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_slot_axiom_qa_demo import main as qa_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        qa_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_qa(
    model_name: str,
    n_steps: int = 400,
    lr: float = 0.05,
    max_new: int = 80,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_qa_demo",
        "--model-name",
        model_name,
        "--n-steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_qa_demo import main as sp_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        sp_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 120,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_plus_qa(
    model_name: str,
    n_ghost: int = 8,
    n_steps: int = 800,
    lr: float = 0.05,
    max_new: int = 80,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_plus_qa_demo",
        "--model-name",
        model_name,
        "--n-ghost",
        str(n_ghost),
        "--n-steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_plus_qa_demo import main as spp_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        spp_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 120,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_plus_v3(
    model_name: str,
    n_ghost: int = 8,
    n_steps: int = 1200,
    lr: float = 0.05,
    max_new: int = 120,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_plus_v3_demo",
        "--model-name",
        model_name,
        "--n-ghost",
        str(n_ghost),
        "--n-steps",
        str(n_steps),
        "--lr",
        str(lr),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_plus_v3_demo import main as v3_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        v3_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 180,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_plus_v4(
    model_name: str,
    n_ghost: int = 8,
    n_steps: int = 3500,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    max_new: int = 120,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_plus_v4_demo",
        "--model-name",
        model_name,
        "--n-ghost",
        str(n_ghost),
        "--n-steps",
        str(n_steps),
        "--lr-start",
        str(lr_start),
        "--lr-end",
        str(lr_end),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_plus_v4_demo import main as v4_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        v4_main()
    return buf.getvalue()


@app.function(
    gpu="H100",
    timeout=60 * 120,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_plus_v6(
    model_name: str,
    n_ghost: int = 8,
    n_steps: int = 2000,
    batch_size: int = 4,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    norm_anchor_lambda: float = 0.01,
    boundary_keep: int = 12,
    max_new: int = 120,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_plus_v6_demo",
        "--model-name",
        model_name,
        "--n-ghost",
        str(n_ghost),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(batch_size),
        "--lr-start",
        str(lr_start),
        "--lr-end",
        str(lr_end),
        "--norm-anchor-lambda",
        str(norm_anchor_lambda),
        "--boundary-keep",
        str(boundary_keep),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_plus_v6_demo import main as v6_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        v6_main()
    return buf.getvalue()


@app.function(
    gpu="H100",
    timeout=60 * 120,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_plus_v7(
    model_name: str,
    n_ghost: int = 8,
    n_steps: int = 2000,
    batch_size: int = 4,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    norm_anchor_lambda: float = 0.01,
    n_synthetic: int = 30,
    synth_replication: int = 2,
    boundary_keep: int = 12,
    max_new: int = 120,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_plus_v7_demo",
        "--model-name",
        model_name,
        "--n-ghost",
        str(n_ghost),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(batch_size),
        "--lr-start",
        str(lr_start),
        "--lr-end",
        str(lr_end),
        "--norm-anchor-lambda",
        str(norm_anchor_lambda),
        "--n-synthetic",
        str(n_synthetic),
        "--synth-replication",
        str(synth_replication),
        "--boundary-keep",
        str(boundary_keep),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_plus_v7_demo import main as v7_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        v7_main()
    return buf.getvalue()


@app.function(
    gpu="H100",
    timeout=60 * 120,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_slots(
    model_name: str,
    n_steps: int = 2500,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    boundary_slots: int = 3,
    no_train_term: bool = False,
    max_new: int = 120,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_slots_demo",
        "--model-name",
        model_name,
        "--n-steps",
        str(n_steps),
        "--lr-start",
        str(lr_start),
        "--lr-end",
        str(lr_end),
        "--boundary-slots",
        str(boundary_slots),
        "--max-new",
        str(max_new),
    ]
    if no_train_term:
        sys.argv.append("--no-train-term")
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_slots_demo import main as slots_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        slots_main()
    return buf.getvalue()


@app.local_entrypoint()
def soft_prompt_slots(
    model: str = "Qwen/Qwen2.5-32B",
    n_steps: int = 2500,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    boundary_slots: int = 3,
    no_train_term: bool = False,
    max_new: int = 120,
) -> None:
    """v9: slot-assigned soft prompts. Each ghost slot has one Q+A role,
    initialized from informative tokens of its answer, trained via
    gradient-masked updates (only the assigned slot updates per step)."""
    print(
        f"soft-prompt slots on {model} steps={n_steps} "
        f"boundary_slots={boundary_slots} no_train_term={no_train_term}"
    )
    output = run_soft_prompt_slots.remote(
        model, n_steps, lr_start, lr_end, boundary_slots, no_train_term, max_new
    )
    print(output)


@app.local_entrypoint()
def soft_prompt_plus_v7(
    model: str = "Qwen/Qwen2.5-32B",
    n_ghost: int = 8,
    n_steps: int = 2000,
    batch_size: int = 4,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    norm_anchor_lambda: float = 0.01,
    n_synthetic: int = 30,
    synth_replication: int = 2,
    boundary_keep: int = 12,
    max_new: int = 120,
) -> None:
    """v6 + teacher-distilled Q+A pairs (teacher = model + full prefix)."""
    print(
        f"soft-prompt+ v7 on {model} bs={batch_size} steps={n_steps} "
        f"n_synth={n_synthetic} boundary_keep={boundary_keep}"
    )
    output = run_soft_prompt_plus_v7.remote(
        model,
        n_ghost,
        n_steps,
        batch_size,
        lr_start,
        lr_end,
        norm_anchor_lambda,
        n_synthetic,
        synth_replication,
        boundary_keep,
        max_new,
    )
    print(output)


@app.local_entrypoint()
def soft_prompt_plus_v6(
    model: str = "Qwen/Qwen2.5-32B",
    n_ghost: int = 8,
    n_steps: int = 2000,
    batch_size: int = 4,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    norm_anchor_lambda: float = 0.01,
    boundary_keep: int = 12,
    max_new: int = 120,
) -> None:
    """v5 + batched training + H100. Expected ~4-6× faster than v5/A100."""
    print(
        f"soft-prompt+ v6 on {model} bs={batch_size} steps={n_steps} "
        f"boundary_keep={boundary_keep} norm_lambda={norm_anchor_lambda}"
    )
    output = run_soft_prompt_plus_v6.remote(
        model,
        n_ghost,
        n_steps,
        batch_size,
        lr_start,
        lr_end,
        norm_anchor_lambda,
        boundary_keep,
        max_new,
    )
    print(output)


@app.function(
    gpu="H100",
    timeout=60 * 180,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_soft_prompt_plus_v5(
    model_name: str,
    n_ghost: int = 8,
    n_steps: int = 3500,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    norm_anchor_lambda: float = 0.01,
    boundary_keep: int = 12,
    max_new: int = 120,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_soft_prompt_plus_v5_demo",
        "--model-name",
        model_name,
        "--n-ghost",
        str(n_ghost),
        "--n-steps",
        str(n_steps),
        "--lr-start",
        str(lr_start),
        "--lr-end",
        str(lr_end),
        "--norm-anchor-lambda",
        str(norm_anchor_lambda),
        "--boundary-keep",
        str(boundary_keep),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_soft_prompt_plus_v5_demo import main as v5_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        v5_main()
    return buf.getvalue()


@app.local_entrypoint()
def soft_prompt_plus_v5(
    model: str = "Qwen/Qwen2.5-32B",
    n_ghost: int = 8,
    n_steps: int = 3500,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    norm_anchor_lambda: float = 0.01,
    boundary_keep: int = 12,
    max_new: int = 120,
) -> None:
    """v4 + L2-norm anchor regularization to keep trained vector close
    to natural embedding magnitudes."""
    print(
        f"soft-prompt+ v5 on {model} n_ghost={n_ghost} steps={n_steps} "
        f"lr {lr_start} -> {lr_end} norm_lambda={norm_anchor_lambda} "
        f"boundary_keep={boundary_keep}"
    )
    output = run_soft_prompt_plus_v5.remote(
        model, n_ghost, n_steps, lr_start, lr_end, norm_anchor_lambda, boundary_keep, max_new
    )
    print(output)


@app.local_entrypoint()
def soft_prompt_plus_v4(
    model: str = "Qwen/Qwen2.5-32B",
    n_ghost: int = 8,
    n_steps: int = 3500,
    lr_start: float = 0.05,
    lr_end: float = 0.005,
    max_new: int = 120,
) -> None:
    """Soft prompt+ v4: longer training + cosine LR decay + EOS in
    targets + generic boundary examples. Robust generic recipe."""
    print(f"soft-prompt+ v4 on {model} n_ghost={n_ghost} steps={n_steps} lr {lr_start} -> {lr_end}")
    output = run_soft_prompt_plus_v4.remote(model, n_ghost, n_steps, lr_start, lr_end, max_new)
    print(output)


@app.local_entrypoint()
def soft_prompt_plus_v3(
    model: str = "Qwen/Qwen2.5-32B",
    n_ghost: int = 8,
    n_steps: int = 1200,
    lr: float = 0.05,
    max_new: int = 120,
) -> None:
    """Soft prompt + ghosts + paraphrased Q+A + boundary + overview
    training to suppress hallucinations and improve overview answers."""
    print(f"soft-prompt+ v3 on {model} n_ghost={n_ghost} steps={n_steps} lr={lr}")
    output = run_soft_prompt_plus_v3.remote(model, n_ghost, n_steps, lr, max_new)
    print(output)


@app.local_entrypoint()
def soft_prompt_plus_qa(
    model: str = "Qwen/Qwen2.5-32B",
    n_ghost: int = 8,
    n_steps: int = 800,
    lr: float = 0.05,
    max_new: int = 80,
) -> None:
    """Soft prompt + ghost tokens + paraphrased Q+A training."""
    print(f"soft-prompt-plus QA on {model} n_ghost={n_ghost} steps={n_steps} lr={lr}")
    output = run_soft_prompt_plus_qa.remote(model, n_ghost, n_steps, lr, max_new)
    print(output)


@app.local_entrypoint()
def soft_prompt_qa(
    model: str = "Qwen/Qwen2.5-32B",
    n_steps: int = 400,
    lr: float = 0.05,
    max_new: int = 80,
) -> None:
    """Term-position-at-L0 soft prompt, trained on Q+A pairs.
    Compare against slot_axiom_qa: same training data, different
    injection location."""
    print(f"soft-prompt QA on {model} steps={n_steps} lr={lr}")
    output = run_soft_prompt_qa.remote(model, n_steps, lr, max_new)
    print(output)


@app.local_entrypoint()
def slot_axiom_qa(
    model: str = "Qwen/Qwen2.5-32B",
    slot_width: int = 1024,
    target_layer_frac: float = 0.5,
    n_steps: int = 400,
    lr: float = 0.05,
    max_new: int = 80,
) -> None:
    """Slot-axiom v2: train per-axiom slot on Q+A pairs, probe with both
    training and held-out questions."""
    print(
        f"slot-axiom QA on {model} slot_width={slot_width} "
        f"layer_frac={target_layer_frac} steps={n_steps} lr={lr}"
    )
    output = run_slot_axiom_qa.remote(model, slot_width, target_layer_frac, n_steps, lr, max_new)
    print(output)


@app.local_entrypoint()
def slot_axiom(
    model: str = "Qwen/Qwen2.5-32B",
    slot_widths: str = "256,512,1024",
    target_layer_frac: float = 0.5,
    n_steps: int = 300,
    lr: float = 0.05,
    max_new: int = 80,
) -> None:
    """Per-axiom slot training: gradient-train a slot vector to make the
    model reproduce an axiom's description, then probe with questions."""
    print(
        f"slot-axiom test on {model} slot_widths={slot_widths} "
        f"layer_frac={target_layer_frac} steps={n_steps} lr={lr}"
    )
    output = run_slot_axiom.remote(model, slot_widths, target_layer_frac, n_steps, lr, max_new)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_composed_axiom(
    model_name: str,
    n_prefix_tokens: int,
    max_new: int,
    top_axiom: str = "data_pipeline",
    use_chat: bool = False,
    only_3plus: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_composed_axiom_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--max-new",
        str(max_new),
        "--top-axiom",
        top_axiom,
    ]
    if use_chat:
        sys.argv.append("--use-chat")
    if only_3plus:
        sys.argv.append("--only-3plus")
    import io
    from contextlib import redirect_stdout

    from marker.run_composed_axiom_demo import main as composed_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        composed_main()
    return buf.getvalue()


@app.local_entrypoint()
def composed_axiom(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 256,
    max_new: int = 180,
    top_axiom: str = "data_pipeline",
    use_chat: bool = False,
    only_3plus: bool = False,
) -> None:
    """Capture compositional concept ONCE at understanding time, query
    with single prefix → collapses 3+ axiom problem to n=1.
    """
    print(
        f"composed-axiom test on {model} top_axiom={top_axiom} "
        f"prefix_tokens={n_prefix_tokens} only_3plus={only_3plus}"
    )
    output = run_composed_axiom.remote(
        model, n_prefix_tokens, max_new, top_axiom, use_chat, only_3plus
    )
    print(output)


@app.local_entrypoint()
def chain_test(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 32,
    max_new: int = 180,
    target_layers: str = "",
    use_chat: bool = False,
) -> None:
    """Dependency-chain test: axioms that reference each other +
    C++ functions that call each other + stdlib knowledge.
    """
    print(
        f"chain test on {model} prefix_tokens={n_prefix_tokens} "
        f"target_layers={target_layers or 'ALL'} use_chat={use_chat}"
    )
    output = run_chain.remote(model, n_prefix_tokens, max_new, target_layers, use_chat)
    print(output)


@app.local_entrypoint()
def reasoning_test(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 32,
    max_new: int = 120,
    target_layers: str = "",
    use_chat: bool = False,
) -> None:
    """Reasoning composition test: does prefix injection enable reasoning
    with axiom facts, or only recitation?
    """
    print(
        f"reasoning test on {model} prefix_tokens={n_prefix_tokens} "
        f"target_layers={target_layers or 'ALL'} use_chat={use_chat}"
    )
    output = run_reasoning.remote(model, n_prefix_tokens, max_new, target_layers, use_chat)
    print(output)


@app.local_entrypoint()
def prefix_gauntlet(
    model: str = "Qwen/Qwen2.5-32B",
    n_steps: int = 80,
    max_new: int = 60,
    n_prefix_tokens: int = 32,
    axioms: str = "",
    target_layers: str = "",
    skip_training: bool = False,
    use_chat: bool = False,
    bleed: bool = False,
) -> None:
    """Per-axiom prefix tuning on `model`. `target_layers` is a comma
    list of layer indices (e.g. '16,32,48'); default empty = all layers.
    """
    print(
        f"prefix-tuning gauntlet on {model} steps={n_steps} "
        f"prefix_tokens={n_prefix_tokens} target_layers={target_layers or 'ALL'} "
        f"skip_training={skip_training} use_chat={use_chat} bleed={bleed} "
        f"axioms={axioms or 'ALL'}"
    )
    output = run_prefix.remote(
        model,
        n_steps,
        max_new,
        n_prefix_tokens,
        axioms,
        target_layers,
        skip_training,
        use_chat,
        bleed,
    )
    print(output)


@app.local_entrypoint()
def gauntlet(
    model: str = "Qwen/Qwen2.5-32B",
    layers: str = "23,53",
    n_steps: int = 200,
    target_count: int = 80,
    batch_size: int = 4,
    axioms: str = "",
) -> None:
    """Full gauntlet: every axiom in axiom_registry on `model`."""
    print(
        f"running gauntlet on {model} layers={layers} batch={batch_size} axioms={axioms or 'ALL'}"
    )
    output = run_combined.remote(
        model, layers, n_steps, target_count, 60, False, batch_size, axioms
    )
    print(output)


@app.local_entrypoint()
def gemma_combined(
    model: str = "google/gemma-4-31B-it",
    layers: str = "23,53",
    n_steps: int = 200,
    target_count: int = 80,
    use_chat: bool = True,
    batch_size: int = 4,
) -> None:
    """Combined soft-prompt + v_residual on Gemma 4-IT."""
    print(
        f"running combined demo on {model} layers={layers} use_chat={use_chat} batch={batch_size}"
    )
    output = run_combined.remote(model, layers, n_steps, target_count, 60, use_chat, batch_size)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_better_inject(model_name: str, layer: int, max_new: int = 60) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_better_inject",
        "--model-name",
        model_name,
        "--layer",
        str(layer),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_better_inject import main as bi_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        bi_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_iti_plus_mult(model_name: str, residual_layer: int, max_new: int = 60) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_iti_plus_mult",
        "--model-name",
        model_name,
        "--residual-layer",
        str(residual_layer),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_iti_plus_mult import main as ipm_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        ipm_main()
    return buf.getvalue()


@app.local_entrypoint()
def itimult(model: str = "Qwen/Qwen2.5-32B", layer: int = 60) -> None:
    """ITI + multiplicative residual injection on 32B."""
    print(f"running iti+mult on {model} at L{layer}")
    output = run_iti_plus_mult.remote(model, layer)
    print(output)


@app.local_entrypoint()
def better(model: str = "Qwen/Qwen2.5-32B", layer: int = 60, max_new: int = 60) -> None:
    """Run Fisher + multiplicative injection sweep on 32B at L60."""
    print(f"running better-inject on {model} at L{layer}")
    output = run_better_inject.remote(model, layer, max_new)
    print(output)


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_axiom_mlp_mini(
    model_name: str,
    n_steps: int = 2000,
    r: int = 16,
    lr: float = 3e-5,
    max_new: int = 80,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_axiom_mlp_mini",
        "--model-name",
        model_name,
        "--n-steps",
        str(n_steps),
        "--r",
        str(r),
        "--lr",
        str(lr),
        "--max-new",
        str(max_new),
    ]
    import io
    from contextlib import redirect_stdout

    from marker.run_axiom_mlp_mini import main as mlp_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        mlp_main()
    return buf.getvalue()


@app.local_entrypoint()
def axiom_mlp(
    model: str = "Qwen/Qwen2.5-32B",
    n_steps: int = 2000,
    r: int = 16,
    lr: float = 3e-5,
    max_new: int = 80,
) -> None:
    """Per-axiom MLP injection mini-test. Trains a small MLP at each of 3
    chosen layers to inject the Glorbox fictional axiom at the term position.
    Tests multi-layer query-conditional residual injection."""
    print(f"axiom-mlp on {model} steps={n_steps} r={r} lr={lr}")
    output = run_axiom_mlp_mini.remote(model, n_steps, r, lr, max_new)
    print(output)


@app.function(
    gpu="H100",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_axiom_mlp_demo(
    model_name: str,
    n_steps: int = 3000,
    r: int = 32,
    lr_start: float = 3e-5,
    lr_end: float = 3e-6,
    n_synthetic: int = 30,
    max_new: int = 120,
    skill_r: int = 64,
    skill_n_steps: int = 3000,
    compress_kv: bool = False,
    n_compressed_tokens: int = 4,
    compressor_steps: int = 1000,
    r_sweep: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_axiom_mlp_demo",
        "--model-name",
        model_name,
        "--n-steps",
        str(n_steps),
        "--r",
        str(r),
        "--lr-start",
        str(lr_start),
        "--lr-end",
        str(lr_end),
        "--n-synthetic",
        str(n_synthetic),
        "--max-new",
        str(max_new),
        "--skill-r",
        str(skill_r),
        "--skill-n-steps",
        str(skill_n_steps),
        "--n-compressed-tokens",
        str(n_compressed_tokens),
        "--compressor-steps",
        str(compressor_steps),
    ]
    if compress_kv:
        sys.argv.append("--compress-kv")
    if r_sweep:
        sys.argv.append("--r-sweep")
    import io
    from contextlib import redirect_stdout

    from marker.run_axiom_mlp_demo import main as demo_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        demo_main()
    return buf.getvalue()


@app.local_entrypoint()
def axiom_mlp_demo(
    model: str = "Qwen/Qwen2.5-32B",
    n_steps: int = 3000,
    r: int = 32,
    lr_start: float = 3e-5,
    lr_end: float = 3e-6,
    n_synthetic: int = 30,
    max_new: int = 120,
    skill_r: int = 64,
    skill_n_steps: int = 3000,
    compress_kv: bool = False,
    n_compressed_tokens: int = 4,
    compressor_steps: int = 1000,
    r_sweep: bool = False,
) -> None:
    """Per-axiom MLP v2: hand-written Q+A + teacher distillation + overview + boundary.
    r=32, cosine LR decay, 3000 steps. Compares A/P/M on TRAIN/HELDOUT/BOUNDARY/TELL_ME.
    Pass --compress-kv to enable KV compression. Pass --r-sweep to compare r=4/8/16/32."""
    print(
        f"axiom-mlp-demo on {model} steps={n_steps} r={r} n_synthetic={n_synthetic} "
        f"skill_r={skill_r} compress_kv={compress_kv} r_sweep={r_sweep}"
    )
    output = run_axiom_mlp_demo.remote(
        model,
        n_steps,
        r,
        lr_start,
        lr_end,
        n_synthetic,
        max_new,
        skill_r,
        skill_n_steps,
        compress_kv,
        n_compressed_tokens,
        compressor_steps,
        r_sweep,
    )
    print(output)


@app.local_entrypoint()
def big(
    model: str = "Qwen/Qwen2.5-32B",
    layers: str = "40,60",
    axiom: str = "both",
    max_new: int = 60,
    logit_alpha_axiom: float = 0.04,
    iti_alpha: float = 2.0,
    layer_alpha: float = 0.5,
) -> None:
    """Defaults are 32B-tuned: layers from probe, alphas scaled down 10x for L60 vector magnitude."""
    layer_list = [int(x) for x in layers.split(",")]
    extra = [
        "--logit-alpha-axiom",
        str(logit_alpha_axiom),
        "--iti-alpha",
        str(iti_alpha),
        "--layer-alpha",
        str(layer_alpha),
    ]
    print(f"running blends on {model}, layers={layer_list}, axiom={axiom}, extra={extra}")
    output = run_blends_big.remote(model, axiom, max_new, layer_list, extra)
    print(output)


# ── Axiom persistence ─────────────────────────────────────────────────────────


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/axioms": axiom_vol,
    },
)
def run_save_axioms(model_name: str = "Qwen/Qwen2.5-32B") -> str:
    """Train all demo axioms and save to the mimir-axioms volume."""
    import io
    import os
    import sys
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marker.axiom_store import save_axiom
    from marker.prefix_tuning import Prefix
    from marker.run_axiom_mlp_demo import (
        SKILL_AXIOM,
        SKILL_AXIOM_ILP,
        compute_axiom_kv,
        make_axiom_mlp,
        train,
    )
    from marker.run_soft_prompt_plus_v4_demo import TEST_AXIOMS, _generic_boundary_examples
    from marker.soft_prompt_plus import generate_synthetic_qa_pairs

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    chosen_layers = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]
    log = []

    # ── Fact axioms ───────────────────────────────────────────────────────────
    from marker.run_axiom_mlp_demo import (
        SUPPLEMENTAL_QA,
        _ensure_term_in_qa,
    )

    for axiom in TEST_AXIOMS:
        name = axiom["name"]
        desc = axiom["description"]
        print(f"\nTraining {name}...")

        prefix = Prefix.from_description(
            model,
            tokenizer,
            desc,
            max_tokens=max(64, len(tokenizer(desc, add_special_tokens=False).input_ids)),
            target_layers=list(range(n_layers)),
        )

        train_qa = [(q, f["answer"]) for f in axiom["facts"] for q in f["questions_train"]]

        buf = io.StringIO()
        with redirect_stdout(buf):
            synth = generate_synthetic_qa_pairs(
                model, tokenizer, desc, prefix, n_pairs=20, max_new=2200
            )
        synth = _ensure_term_in_qa(synth, name)
        train_qa.extend(synth)
        train_qa.extend(SUPPLEMENTAL_QA.get(name, []))
        train_qa += [(f"Tell me about {name}.", desc), (f"What is {name}?", desc)]
        boundary_qa = _generic_boundary_examples(name)

        axiom_mlp = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=32)
        axiom_mlp.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
        train(model, tokenizer, axiom_mlp, train_qa, boundary_pairs=boundary_qa, n_steps=3000)
        save_axiom(axiom_mlp, f"/axioms/{name}.pt")
        axiom_vol.commit()
        log.append(f"saved {name}")
        print(f"  saved {name}")

    # ── Skill axioms ──────────────────────────────────────────────────────────
    for skill_def in [SKILL_AXIOM, SKILL_AXIOM_ILP]:
        name = skill_def["term"]
        desc = skill_def["description"]
        print(f"\nTraining skill {name}...")
        skill_mlp = make_axiom_mlp(model, tokenizer, name, chosen_layers, r=64)
        skill_mlp.skill_mode = True
        skill_mlp.kv = compute_axiom_kv(model, tokenizer, desc, term=name)
        train(model, tokenizer, skill_mlp, skill_def["qa"], n_steps=3000)
        save_axiom(skill_mlp, f"/axioms/{name}.pt")
        axiom_vol.commit()
        log.append(f"saved skill {name}")
        print(f"  saved skill {name}")

    return "\n".join(log)


@app.local_entrypoint()
def save_axioms(model: str = "Qwen/Qwen2.5-32B") -> None:
    """Train and save all axioms to the mimir-axioms Modal Volume."""
    print(f"Training and saving axioms for {model}...")
    result = run_save_axioms.remote(model)
    print(result)


# ── Chat server ───────────────────────────────────────────────────────────────


@app.function(
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/axioms": axiom_vol,
    },
    min_containers=0,
)
@modal.asgi_app()
def chat_app() -> object:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    from marker.chat_server import create_app

    return create_app(model_name="Qwen/Qwen2.5-32B", axiom_dir="/axioms")


@app.local_entrypoint()
def deploy_chat() -> None:
    """Deploy the Mimir chat server. Run once; the endpoint stays live."""
    print("Deploying chat server...")
    print("Once deployed, get the URL from: modal app list")
    print("Then update docs/index.html with your endpoint URL.")
