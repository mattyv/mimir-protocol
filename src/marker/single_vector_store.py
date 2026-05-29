"""Save and load single-vector axioms (seed embedding + bolt-on adapters).

Mirrors ``axiom_store.py`` but for the Phase 1/2 architecture. The on-disk
record stores the seed's identity (name + the trained embedding row) and the
bolt-on adapter weights. On load we re-register the seed token against the
target tokenizer/model (which re-adds the vocab entry and resizes the
embedding table), overwrite the new row with the saved vector, rebuild the
bolt-on, and load its state dict.

We deliberately do NOT serialize the tokenizer. The seed token id is
re-derived deterministically by ``register_seed_token`` on load, exactly as
``axiom_store`` recomputes ``term_token_ids`` from the term string.
"""

from __future__ import annotations

from pathlib import Path

import torch

from marker.bolt_selector import BoltSelector, make_bolt_selector
from marker.seed_token import SeedToken, register_seed_token, seed_embedding


def save_single_vector_axiom(
    model,  # noqa: ANN001
    seed: SeedToken,
    bolt: BoltSelector,
    path: str | Path,
) -> None:
    """Serialize a single-vector axiom to a ``.pt`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": seed.name,
            "original_term": seed.original_term,
            "original_bpe_ids": seed.original_bpe_ids,
            "seed_vector": seed_embedding(model, seed).detach().cpu().clone(),
            "r": bolt.r,
            "skill_mode": bolt.skill_mode,
            "adapter_state": {k: v.cpu() for k, v in bolt.adapters.state_dict().items()},
        },
        path,
    )


def load_single_vector_axiom(path: str | Path, model, tokenizer):  # noqa: ANN001, ANN201
    """Load a single-vector axiom onto ``model``/``tokenizer``.

    Returns ``(SeedToken, BoltSelector)`` ready for ``install_bolt_hooks``.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    seed = register_seed_token(model, tokenizer, data["name"])

    # Overwrite the BPE-mean init with the trained seed vector.
    device = next(model.parameters()).device
    dtype = model.get_input_embeddings().weight.dtype
    with torch.no_grad():
        seed_embedding(model, seed).copy_(data["seed_vector"].to(device=device, dtype=dtype))

    bolt = make_bolt_selector(model, seed, r=data["r"], skill_mode=data["skill_mode"])
    bolt.adapters.load_state_dict(data["adapter_state"])
    bolt.adapters = bolt.adapters.to(device=device, dtype=torch.float32)
    return seed, bolt
