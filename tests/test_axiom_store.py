"""Round-trip tests for save_axiom / load_axiom."""

from __future__ import annotations

import torch
import torch.nn as nn

from marker.axiom_store import save_axiom
from marker.run_axiom_mlp_demo import AxiomKV, AxiomMLP, SmallMLP


def _fake_axiom(skill_mode: bool = False, with_kv: bool = True) -> AxiomMLP:
    mlps = nn.ModuleList([SmallMLP(8, r=4) for _ in range(2)])
    kv = None
    if with_kv:
        kv = AxiomKV(
            n_layers=2,
            keys=[torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)],
            values=[torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)],
        )
    return AxiomMLP(
        term="Foo",
        term_token_ids=[10, 11],
        chosen_layers=[3, 7],
        mlps=mlps,
        kv=kv,
        dependencies=["Bar"],
        skill_mode=skill_mode,
    )


def test_save_axiom_writes_all_fields(tmp_path):
    axiom = _fake_axiom()
    path = tmp_path / "foo.pt"

    save_axiom(axiom, path)

    assert path.exists()
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["term"] == "Foo"
    assert data["term_token_ids"] == [10, 11]
    assert data["chosen_layers"] == [3, 7]
    assert data["r"] == 4
    assert data["dependencies"] == ["Bar"]
    assert data["skill_mode"] is False
    assert data["kv_n_layers"] == 2
    assert len(data["kv_keys"]) == 2
    assert len(data["kv_values"]) == 2


def test_save_axiom_preserves_tensors(tmp_path):
    axiom = _fake_axiom()
    path = tmp_path / "foo.pt"

    save_axiom(axiom, path)

    data = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in axiom.mlps.state_dict().items():
        assert torch.equal(data["mlp_state"][k], v.cpu())
    for layer_idx in range(2):
        assert torch.equal(data["kv_keys"][layer_idx], axiom.kv.keys[layer_idx].cpu())
        assert torch.equal(data["kv_values"][layer_idx], axiom.kv.values[layer_idx].cpu())


def test_save_axiom_skill_mode_and_no_kv(tmp_path):
    axiom = _fake_axiom(skill_mode=True, with_kv=False)
    path = tmp_path / "foo.pt"

    save_axiom(axiom, path)

    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["skill_mode"] is True
    assert data["kv_keys"] is None
    assert data["kv_values"] is None
    assert data["kv_n_layers"] is None


def test_save_axiom_creates_parent_dir(tmp_path):
    axiom = _fake_axiom(with_kv=False)
    path = tmp_path / "nested" / "subdir" / "foo.pt"

    save_axiom(axiom, path)

    assert path.exists()
