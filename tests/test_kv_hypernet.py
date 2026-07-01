"""Mechanical invariants for the KVHypernet axiom store.

Model-free: exercises the net's shapes, the store round-trip, RoPE-corrected
assembly, and the realtime-add (encode) path. The end-to-end training loop is
covered by run_hypernet_demo.py on GPU.
"""

from __future__ import annotations

import torch

from marker.kv_hypernet import (
    AxiomCode,
    KVHypernet,
    assemble_kv,
    load_axiom_code,
    load_hypernet,
    make_axiom_code,
    save_axiom_code,
    save_hypernet,
)
from marker.run_axiom_mlp_demo import AxiomKV

THETA = 10000.0
DIMS = dict(n_layers=4, n_kv_heads=2, head_dim=8)


def _full_kv(n_tokens: int = 20, seed: int = 0) -> AxiomKV:
    g = torch.Generator().manual_seed(seed)
    shape = (1, DIMS["n_kv_heads"], n_tokens, DIMS["head_dim"])
    return AxiomKV(
        n_layers=DIMS["n_layers"],
        keys=[torch.randn(shape, generator=g) for _ in range(DIMS["n_layers"])],
        values=[torch.randn(shape, generator=g) for _ in range(DIMS["n_layers"])],
    )


def test_encode_returns_latent_vector():
    net = KVHypernet(**DIMS, d_latent=64)
    z = net.encode(_full_kv())
    assert z.shape == (64,)


def test_decode_produces_n_scaffold_tokens_per_layer():
    net = KVHypernet(**DIMS, d_latent=64, n_scaffold=4)
    scaffold = net.decode_scaffold(torch.zeros(64), torch.device("cpu"))
    assert scaffold.n_layers == DIMS["n_layers"]
    for layer in range(scaffold.n_layers):
        assert scaffold.keys[layer].shape == (1, DIMS["n_kv_heads"], 4, DIMS["head_dim"])


def test_decode_is_deterministic():
    net = KVHypernet(**DIMS, d_latent=64)
    z = torch.randn(64)
    a = net.decode_scaffold(z, torch.device("cpu"))
    b = net.decode_scaffold(z, torch.device("cpu"))
    assert torch.equal(a.keys[0], b.keys[0])


def test_encode_is_differentiable():
    net = KVHypernet(**DIMS, d_latent=64)
    z = net.encode(_full_kv())
    assert z.requires_grad and z.grad_fn is not None


def test_assemble_concatenates_scaffold_then_facts():
    net = KVHypernet(**DIMS, d_latent=64, n_scaffold=4)
    scaffold = net.decode_scaffold(torch.zeros(64), torch.device("cpu"))
    facts = _full_kv(n_tokens=7, seed=3)
    merged = assemble_kv(scaffold, facts, THETA)
    for layer in range(DIMS["n_layers"]):
        assert merged.keys[layer].shape[2] == 4 + 7  # scaffold tokens + fact tokens
        # scaffold half is preserved (cast only); facts half is RoPE-rotated, so differs
        assert torch.allclose(merged.values[layer][:, :, 4:], facts.values[layer])


def test_make_axiom_code_stores_cpu_latent_and_facts():
    net = KVHypernet(**DIMS, d_latent=64)
    code = make_axiom_code(net, _full_kv(), "poll=250ms", "Foo")
    assert code.term == "Foo"
    assert code.z.shape == (64,) and code.z.device.type == "cpu"
    assert not code.z.requires_grad
    assert code.fact_text == "poll=250ms"


def test_code_is_smaller_than_full_kv():
    net = KVHypernet(**DIMS, d_latent=64)
    full = _full_kv(n_tokens=50)
    code = make_axiom_code(net, full, "poll=250ms; topic=balances.raw", "Foo")
    code_bytes = code.z.numel() * 4 + len(code.fact_text.encode())
    full_bytes = sum(k.nbytes + v.nbytes for k, v in zip(full.keys, full.values, strict=True))
    assert code_bytes < full_bytes


def test_axiom_code_roundtrip(tmp_path):
    net = KVHypernet(**DIMS, d_latent=64)
    code = make_axiom_code(net, _full_kv(), "poll=250ms", "Foo")
    p = tmp_path / "code.pt"
    save_axiom_code(code, p)
    loaded = load_axiom_code(p)
    assert loaded.term == code.term
    assert loaded.fact_text == code.fact_text
    assert torch.equal(loaded.z, code.z)


def test_hypernet_roundtrip(tmp_path):
    net = KVHypernet(**DIMS, d_latent=64, n_scaffold=4)
    p = tmp_path / "hn.pt"
    save_hypernet(net, p)
    loaded = load_hypernet(p)
    z = torch.randn(64)
    a = net.decode_scaffold(z, torch.device("cpu"))
    b = loaded.decode_scaffold(z, torch.device("cpu"))
    assert torch.equal(a.keys[0], b.keys[0])


def test_axiom_code_dataclass_shape():
    code = AxiomCode(term="X", z=torch.zeros(8), fact_text="a=1")
    assert code.z.shape == (8,)
