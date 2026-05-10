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
def run_chain_selective_recompute(
    model_name: str,
    n_prefix_tokens: int,
    max_new: int,
    target_layers: str = "",
    use_chat: bool = False,
    top_k_pct: float = 0.15,
    selective_layer: int = 1,
    only_3plus: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_chain_selective_recompute_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--max-new",
        str(max_new),
        "--top-k-pct",
        str(top_k_pct),
        "--selective-layer",
        str(selective_layer),
    ]
    if target_layers:
        sys.argv += ["--target-layers", *target_layers.split(",")]
    if use_chat:
        sys.argv.append("--use-chat")
    if only_3plus:
        sys.argv.append("--only-3plus")
    import io
    from contextlib import redirect_stdout

    from marker.run_chain_selective_recompute_demo import main as csr_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        csr_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_chain_ape(
    model_name: str,
    n_prefix_tokens: int,
    max_new: int,
    target_layers: str = "",
    use_chat: bool = False,
    q_scales: str = "1.5,2.0,3.0",
    shared_prefix: str = "\n",
    only_3plus: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_chain_ape_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--max-new",
        str(max_new),
        "--q-scales",
        *q_scales.split(","),
        "--shared-prefix",
        shared_prefix,
    ]
    if target_layers:
        sys.argv += ["--target-layers", *target_layers.split(",")]
    if use_chat:
        sys.argv.append("--use-chat")
    if only_3plus:
        sys.argv.append("--only-3plus")
    import io
    from contextlib import redirect_stdout

    from marker.run_chain_ape_demo import main as ape_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        ape_main()
    return buf.getvalue()


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_chain_ape_recursive(
    model_name: str,
    n_prefix_tokens: int,
    max_new: int,
    target_layers: str = "",
    use_chat: bool = False,
    q_scales: str = "1.2,1.3,1.5,1.7",
    shared_prefixes: str = r"\n|### Context:\n",
    combiners: str = "uniform,cosine",
    only_3plus: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sps = shared_prefixes.split("|")
    sys.argv = [
        "run_chain_ape_recursive_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--max-new",
        str(max_new),
        "--q-scales",
        *q_scales.split(","),
        "--shared-prefixes",
        *sps,
        "--combiners",
        *combiners.split(","),
    ]
    if target_layers:
        sys.argv += ["--target-layers", *target_layers.split(",")]
    if use_chat:
        sys.argv.append("--use-chat")
    if only_3plus:
        sys.argv.append("--only-3plus")
    import io
    from contextlib import redirect_stdout

    from marker.run_chain_ape_recursive_demo import main as ape_rec_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        ape_rec_main()
    return buf.getvalue()


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


@app.function(
    gpu="A100-80GB",
    timeout=60 * 90,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_combine_vectors(
    model_name: str,
    n_prefix_tokens: int,
    composed_prefix_tokens: int,
    max_new: int,
    top_axiom: str = "data_pipeline",
    only_3plus: bool = False,
) -> str:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")
    sys.argv = [
        "run_combine_vectors_demo",
        "--model-name",
        model_name,
        "--n-prefix-tokens",
        str(n_prefix_tokens),
        "--composed-prefix-tokens",
        str(composed_prefix_tokens),
        "--max-new",
        str(max_new),
        "--top-axiom",
        top_axiom,
    ]
    if only_3plus:
        sys.argv.append("--only-3plus")
    import io
    from contextlib import redirect_stdout

    from marker.run_combine_vectors_demo import main as cv_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        cv_main()
    return buf.getvalue()


@app.local_entrypoint()
def combine_vectors(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 64,
    composed_prefix_tokens: int = 768,
    max_new: int = 180,
    top_axiom: str = "data_pipeline",
    only_3plus: bool = False,
) -> None:
    """Combine existing per-axiom prefix vectors to approximate the
    composed capture (M_concat, M_avg, M_bind vs H composed vs E joint).
    """
    print(
        f"combine-vectors test on {model} top={top_axiom} "
        f"n_prefix_tokens={n_prefix_tokens} only_3plus={only_3plus}"
    )
    output = run_combine_vectors.remote(
        model, n_prefix_tokens, composed_prefix_tokens, max_new, top_axiom, only_3plus
    )
    print(output)


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
def chain_ape_recursive(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 48,
    max_new: int = 180,
    target_layers: str = "",
    use_chat: bool = False,
    q_scales: str = "1.5",
    shared_prefixes: str = r"\n",
    combiners: str = "uniform,cosine",
    only_3plus: bool = False,
) -> None:
    """5-axiom hierarchical DAG test with APE + per-block attention.

    Conditions per prompt: A no-prefix, C rope-fix, F APE(q × sp),
    G per-block(combiner), E joint-enc.
    """
    print(
        f"recursive APE+per-block test on {model} prefix_tokens={n_prefix_tokens} "
        f"q_scales={q_scales} shared_prefixes={shared_prefixes!r} "
        f"combiners={combiners} only_3plus={only_3plus}"
    )
    output = run_chain_ape_recursive.remote(
        model,
        n_prefix_tokens,
        max_new,
        target_layers,
        use_chat,
        q_scales,
        shared_prefixes,
        combiners,
        only_3plus,
    )
    print(output)


@app.local_entrypoint()
def chain_ape(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 32,
    max_new: int = 180,
    target_layers: str = "",
    use_chat: bool = False,
    q_scales: str = "1.5,2.0,3.0",
    shared_prefix: str = "\n",
    only_3plus: bool = False,
) -> None:
    """3+ prefix chain test with APE (shared-prefix attention sink + Q
    sharpening) compared against rope-fix (C) and Path 2 joint encoding (E).
    """
    print(
        f"chain APE test on {model} prefix_tokens={n_prefix_tokens} "
        f"q_scales={q_scales} shared_prefix={shared_prefix!r} only_3plus={only_3plus}"
    )
    output = run_chain_ape.remote(
        model,
        n_prefix_tokens,
        max_new,
        target_layers,
        use_chat,
        q_scales,
        shared_prefix,
        only_3plus,
    )
    print(output)


@app.local_entrypoint()
def chain_selective(
    model: str = "Qwen/Qwen2.5-32B",
    n_prefix_tokens: int = 32,
    max_new: int = 180,
    target_layers: str = "",
    use_chat: bool = False,
    top_k_pct: float = 0.15,
    selective_layer: int = 1,
    only_3plus: bool = False,
) -> None:
    """3+ prefix chain test with CacheBlend selective recompute (D) +
    Path 2 joint encoding (E) compared against naive (B) and
    RoPE-corrected (C) baselines.
    """
    print(
        f"chain selective-recompute test on {model} prefix_tokens={n_prefix_tokens} "
        f"top_k_pct={top_k_pct} selective_layer={selective_layer} only_3plus={only_3plus}"
    )
    output = run_chain_selective_recompute.remote(
        model,
        n_prefix_tokens,
        max_new,
        target_layers,
        use_chat,
        top_k_pct,
        selective_layer,
        only_3plus,
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
