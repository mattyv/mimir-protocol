"""Render decoder (Stage-3 render path): reconstruct a step's TEXT from its
thought — the on-demand "show me this thought".

Every decode elsewhere runs the CONTINUATION direction (what follows a thought,
what the encoder was trained for). Render is RECONSTRUCTION of the source step
— transcription, an easier task, but never trained, so it needs a small
decoder. The frozen Stage-1 'default' adapter produces the thought (gist_kv);
a second trainable 'render' LoRA decodes the step back out, CE on the source
span given the injected thought.

(Exact literals — numbers, names — stay lossy at any compression; the literals
ledger, spliced by the render decoder, keeps those deterministic. Ledger is a
separate build; render.py is meaning-reconstruction.)
"""

from __future__ import annotations

import torch

from marker.gist_model import QWEN_TARGETS


def attach_render(peft_model, r: int = 16, alpha: int = 32, targets=None):  # noqa: ANN001
    """Add a trainable RENDER LoRA adapter to the already-gist-adapted model
    and make it active. Encoding uses the frozen 'default' (Stage-1 encoder);
    render decode uses this 'render' adapter — call set_adapter('default')
    before gist_kv, set_adapter('render') before render_nll. Returns
    [(name, param)] of the render trainables."""
    from peft import LoraConfig  # noqa: PLC0415

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets or QWEN_TARGETS,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model.add_adapter("render", cfg)
    peft_model.set_adapter("render")
    return [(n, p) for n, p in peft_model.named_parameters() if "render" in n and p.requires_grad]


def render_nll(
    peft_model,  # noqa: ANN001
    thought_kv,  # noqa: ANN001
    cont_start: int,
    span_ids: list[int],
) -> torch.Tensor:
    """CE of reconstructing span_ids from the injected thought — the render
    training loss. The 'render' adapter must be active; thought_kv came from
    the frozen encoder. Teacher-forced tail (span_ids[1:] from span_ids[:-1]
    over the injected cache at positions cont_start..). Differentiable into the
    render LoRA: the model attends over the constant thought cache through
    render-adapted layers (no detach — GRAD_OK tested)."""
    import torch.nn.functional as F  # noqa: N812, PLC0415
    from transformers import DynamicCache  # noqa: PLC0415

    if len(span_ids) < 2:
        raise ValueError("need >= 2 span tokens to score a reconstruction tail")
    device = next(peft_model.parameters()).device
    cache = DynamicCache()
    for i in range(thought_kv.n_layers):
        cache.update(thought_kv.keys[i].to(device), thought_kv.values[i].to(device), i)
    m = len(span_ids) - 1
    pos = torch.arange(cont_start, cont_start + m, device=device).unsqueeze(0)
    out = peft_model(
        torch.tensor([span_ids[:-1]], device=device),
        past_key_values=cache,
        position_ids=pos,
        use_cache=True,
    )
    return F.cross_entropy(out.logits[0], torch.tensor(span_ids[1:], device=device))
