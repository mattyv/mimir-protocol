"""Save and load trained AxiomMLP objects to/from disk."""

from __future__ import annotations

from pathlib import Path

import torch

from marker.run_axiom_mlp_demo import AxiomKV, AxiomMLP, make_axiom_mlp


def save_axiom(axiom_mlp: AxiomMLP, path: str | Path) -> None:
    """Serialize a trained AxiomMLP to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "term": axiom_mlp.term,
            "term_token_ids": axiom_mlp.term_token_ids,
            "chosen_layers": axiom_mlp.chosen_layers,
            "r": axiom_mlp.mlps[0].down.weight.shape[0],
            "mlp_state": {k: v.cpu() for k, v in axiom_mlp.mlps.state_dict().items()},
            "kv_keys": [k.cpu() for k in axiom_mlp.kv.keys] if axiom_mlp.kv else None,
            "kv_values": [v.cpu() for v in axiom_mlp.kv.values] if axiom_mlp.kv else None,
            "kv_n_layers": axiom_mlp.kv.n_layers if axiom_mlp.kv else None,
            "dependencies": axiom_mlp.dependencies,
            "skill_mode": axiom_mlp.skill_mode,
        },
        path,
    )


def load_axiom(path: str | Path, model, tokenizer) -> AxiomMLP:  # noqa: ANN001
    """Load a serialized AxiomMLP from a .pt file onto the model's device."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    axiom_mlp = make_axiom_mlp(model, tokenizer, data["term"], data["chosen_layers"], r=data["r"])
    axiom_mlp.mlps.load_state_dict(data["mlp_state"])
    device = next(model.parameters()).device
    axiom_mlp.mlps = axiom_mlp.mlps.to(device=device, dtype=torch.float32)
    if data["kv_keys"] is not None:
        axiom_mlp.kv = AxiomKV(
            n_layers=data["kv_n_layers"],
            keys=data["kv_keys"],
            values=data["kv_values"],
        )
    axiom_mlp.dependencies = data["dependencies"]
    axiom_mlp.skill_mode = data["skill_mode"]
    return axiom_mlp


def load_all_axioms(axiom_dir: str | Path, model, tokenizer) -> list[AxiomMLP]:  # noqa: ANN001
    """Load all .pt axiom files from a directory, sorted by filename."""
    axiom_dir = Path(axiom_dir)
    axioms = []
    for pt_file in sorted(axiom_dir.glob("*.pt")):
        axioms.append(load_axiom(pt_file, model, tokenizer))
    return axioms
