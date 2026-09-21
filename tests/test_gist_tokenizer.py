"""Tests for gist_tokenizer.py -- the stage-2 tokenizer that turns a step's
text into [8] dictionary ids (GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3", Corpus
v0). Splitting/caching logic is pure CPU work tested with a FAKE whitespace
tokenizer (no network, no model download); encode_step/decode_ids are tested
with a monkeypatched encode_canonical + a tiny fake dictionary (same "spy,
never call the encoder on a cache hit" pattern run_gist_dict's own tests use)
plus one real-model test for the "matches gist_dict.tokenize" claim end to
end.
"""

from __future__ import annotations

import json

import pytest
import torch

from marker.gist_dict import detokenize, tokenize
from tests.test_run_gist_dict import _fake_kv

_GEO = {"n_layers": 2, "n_kv_heads": 2, "head_dim": 4}


class _FakeTok:
    """Whitespace tokenizer: one id per word (a running vocab), so `.decode`
    can reconstruct words exactly. No network, deterministic -- exercises
    split_long_step's tiering (sentence/clause/hard-split) without a real
    subword tokenizer's irregular boundaries getting in the way."""

    def __init__(self):
        self.vocab: list[str] = []

    def _id(self, w: str) -> int:
        if w not in self.vocab:
            self.vocab.append(w)
        return self.vocab.index(w)

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ARG002
        class _Out:
            pass

        out = _Out()
        out.input_ids = [self._id(w) for w in text.split()]
        return out

    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001, ARG002
        return " ".join(self.vocab[i] for i in ids)


def _norm(s: str) -> str:
    return " ".join(s.split())


def _fake_kv_dict(k_slots=2, K=4, seed=0) -> dict:
    """A tiny 'kv' dictionary (build_dict_kv's exact entry shape) with random
    centroids -- enough for gist_dict.tokenize/detokenize, no model needed."""
    g = torch.Generator().manual_seed(seed)
    d = _GEO["n_layers"] * 2 * _GEO["n_kv_heads"] * _GEO["head_dim"]
    slots = [
        {
            "centroids": torch.randn(K, d, generator=g),
            "mu_readout": torch.randn(K, 3, generator=g),
            "usage": torch.zeros(K, dtype=torch.long),
            "K": K,
            "seed": seed,
        }
        for _ in range(k_slots)
    ]
    return {
        "cfg": f"kv_K{K}",
        "kind": "kv",
        "geometry": {**_GEO, "k_slots": k_slots},
        "slots": slots,
    }


# ── split_long_step: pure splitting logic ───────────────────────────────────


def test_split_long_step_no_split_when_short():
    from marker.gist_tokenizer import split_long_step

    tok = _FakeTok()
    text = "three plus four is seven"
    assert split_long_step(text, tok, max_span=10) == [text]


def test_split_long_step_sentence_splits_and_reproduces_original():
    from marker.gist_tokenizer import split_long_step

    tok = _FakeTok()
    text = "one two three four. five six seven eight. nine ten eleven twelve."
    pieces = split_long_step(text, tok, max_span=4)
    assert len(pieces) > 1
    for p in pieces:
        assert len(tok(p, add_special_tokens=False).input_ids) <= 4
    assert _norm(" ".join(pieces)) == _norm(text)


def test_split_long_step_clause_splits_an_overlong_sentence():
    from marker.gist_tokenizer import split_long_step

    tok = _FakeTok()
    # one "sentence" (no .!?), too long even whole -- must fall to clauses
    text = "alpha beta gamma, delta epsilon zeta; eta theta iota"
    pieces = split_long_step(text, tok, max_span=3)
    assert len(pieces) >= 3
    for p in pieces:
        assert len(tok(p, add_special_tokens=False).input_ids) <= 3
    assert _norm(" ".join(pieces)) == _norm(text)


def test_split_long_step_hard_splits_when_no_punctuation():
    from marker.gist_tokenizer import split_long_step

    tok = _FakeTok()
    text = " ".join(f"word{i}" for i in range(10))  # one long run, no punctuation at all
    pieces = split_long_step(text, tok, max_span=3)
    assert len(pieces) == 4  # ceil(10/3)
    for p in pieces:
        assert len(tok(p, add_special_tokens=False).input_ids) <= 3
    assert _norm(" ".join(pieces)) == _norm(text)


# ── GistTokenizer.encode_step / decode_ids ──────────────────────────────────


def _patched_tokenizer(monkeypatch, dict_, cache_dir=None, base=64, max_span=64, calls=None):
    import marker.gist_tokenizer as gt

    calls = calls if calls is not None else []

    def _fake_encode_canonical(pm, gist, ids, base=64, max_span=64):  # noqa: ANN001, ARG001
        calls.append(tuple(ids))
        kv = _fake_kv(dict_["geometry"]["k_slots"], geo=_GEO, seed=len(calls))
        readout = torch.randn(dict_["geometry"]["k_slots"], 3)
        return kv, readout, base + dict_["geometry"]["k_slots"]

    monkeypatch.setattr(gt, "encode_canonical", _fake_encode_canonical)
    tok = _FakeTok()
    gtok = gt.GistTokenizer(
        pm=None, gist=None, dict_=dict_, tok=tok, base=base, max_span=max_span, cache_dir=cache_dir
    )
    return gtok, calls


def test_encode_step_returns_one_group_per_piece_in_natural_order(monkeypatch):
    dict_ = _fake_kv_dict()
    gtok, calls = _patched_tokenizer(monkeypatch, dict_, max_span=4)
    text = "one two three four. five six seven eight."
    groups = gtok.encode_step(text)
    assert len(groups) == 2  # two sentences, each its own piece/group
    assert len(calls) == 2
    for group in groups:
        assert len(group) == dict_["geometry"]["k_slots"]
        assert all(isinstance(i, int) for i in group)


def test_decode_ids_matches_gist_dict_detokenize(monkeypatch):
    dict_ = _fake_kv_dict()
    gtok, _calls = _patched_tokenizer(monkeypatch, dict_)
    ids8 = [1, 2]
    got = gtok.decode_ids(ids8)
    want = detokenize(ids8, dict_, dict_["geometry"])
    for a, b in zip(got.keys, want.keys, strict=True):
        assert torch.equal(a, b)

    got16 = gtok.decode_ids(ids8, dtype=torch.float16)
    assert got16.keys[0].dtype == torch.float16


def test_encode_step_cache_hit_returns_identical_ids_and_skips_encoder(monkeypatch, tmp_path):
    dict_ = _fake_kv_dict()
    gtok, calls = _patched_tokenizer(monkeypatch, dict_, cache_dir=tmp_path)
    text = "one two three"

    first = gtok.encode_step(text)
    assert len(calls) == 1
    assert gtok.misses == 1 and gtok.hits == 0

    second = gtok.encode_step(text)
    assert len(calls) == 1  # encoder NOT called again
    assert gtok.hits == 1
    assert second == first

    # on-disk layout: cache_dir/<key[:2]>/<key>.json
    files = list(tmp_path.glob("*/*.json"))
    assert len(files) == 1
    key = files[0].stem
    assert files[0].parent.name == key[:2]
    assert json.loads(files[0].read_text()) == first[0]


def test_cache_key_changes_when_a_setting_changes(monkeypatch, tmp_path):
    from marker.gist_tokenizer import _cache_key

    dict_ = _fake_kv_dict()
    settings_a = {"model": "m", "dict_sha1": "d", "base": 64, "max_span": 64, "batching": "single"}
    settings_b = {**settings_a, "base": 65}
    assert _cache_key("text", settings_a) != _cache_key("text", settings_b)

    # end to end: two tokenizers differing only in `base`, same cache dir --
    # each must be its own cache miss (never read the other's cached ids)
    calls: list = []
    gtok_a, _ = _patched_tokenizer(monkeypatch, dict_, cache_dir=tmp_path, base=64, calls=calls)
    gtok_b, _ = _patched_tokenizer(monkeypatch, dict_, cache_dir=tmp_path, base=65, calls=calls)
    gtok_a.encode_step("one two three")
    gtok_b.encode_step("one two three")
    assert len(calls) == 2
    assert gtok_a.misses == 1 and gtok_b.misses == 1


def test_dict_sha1_changes_with_centroid_content():
    from marker.gist_tokenizer import _dict_sha1

    d1 = _fake_kv_dict(seed=0)
    d2 = _fake_kv_dict(seed=1)
    d3 = _fake_kv_dict(seed=0)
    assert _dict_sha1(d1) != _dict_sha1(d2)
    assert _dict_sha1(d1) == _dict_sha1(d3)


# ── real tiny model: ids match gist_dict.tokenize on the SAME kv ────────────


@pytest.mark.slow
def test_encode_step_ids_match_gist_dict_tokenize_on_same_kv():
    from marker.gist_dict import kv_slot_matrix
    from marker.gist_tokenizer import GistTokenizer
    from marker.run_gist_dict import encode_canonical
    from marker.run_stage2 import _load_stage1

    pm, gist, tok = _load_stage1("Qwen/Qwen2.5-0.5B", None, "cpu", False)
    pm.set_adapter("default")
    k_slots = gist.shape[0]

    text = "Three plus four is seven."
    ids = tok(text, add_special_tokens=False).input_ids
    kv, readout, _cs = encode_canonical(pm, gist, ids, base=64, max_span=64)
    d = kv_slot_matrix(kv).shape[1]

    # a tiny real-geometry 'kv' dictionary with random centroids -- only the
    # geometry needs to be real here, the codebook content is incidental
    g = torch.Generator().manual_seed(0)
    K = 5
    slots = [
        {
            "centroids": torch.randn(K, d, generator=g),
            "mu_readout": torch.randn(K, readout.shape[-1], generator=g),
            "usage": torch.zeros(K, dtype=torch.long),
            "K": K,
            "seed": 0,
        }
        for _ in range(k_slots)
    ]
    geometry = {
        "n_layers": kv.n_layers,
        "n_kv_heads": kv.keys[0].shape[1],
        "head_dim": kv.keys[0].shape[3],
        "k_slots": k_slots,
    }
    dict_ = {"cfg": "kv_K5", "kind": "kv", "geometry": geometry, "slots": slots}

    gtok = GistTokenizer(pm, gist, dict_, tok, base=64, max_span=64)
    groups = gtok.encode_step(text)
    assert len(groups) == 1
    want = tokenize(kv, dict_)
    assert groups[0] == want
