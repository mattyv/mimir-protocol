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
        "torch==2.5.1",
        "transformers==4.46.0",
        "accelerate==1.0.1",
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
        "--model-name", model_name,
        "--axiom", axiom,
        "--max-new", str(max_new),
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
        "--logit-alpha-axiom", str(logit_alpha_axiom),
        "--iti-alpha", str(iti_alpha),
        "--layer-alpha", str(layer_alpha),
    ]
    print(f"running blends on {model}, layers={layer_list}, axiom={axiom}, extra={extra}")
    output = run_blends_big.remote(model, axiom, max_new, layer_list, extra)
    print(output)
