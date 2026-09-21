"""Stage-2 gist tokenizer (GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3", Corpus v0):
turns a reasoning step's TEXT into `k_slots` dictionary ids. Wraps the
stage-1 pieces (run_gist_dict.encode_canonical for the one-forward canonical
encode, gist_dict.tokenize/detokenize for the per-slot dictionary lookup)
behind one on-disk-cached call, and handles steps longer than the encoder's
max span (64 tokens) by SPLITTING them into several sub-max-span pieces --
never dropping text, since a dropped piece would break the sequence the
corpus builder is assembling around it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

from marker.gist import split_sentences
from marker.gist_dict import detokenize, tokenize
from marker.run_gist_dict import encode_canonical

# split AFTER a clause-ending mark, so the mark stays attached to the clause
# it ends -- " ".join(pieces) then still reads like the original punctuation
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def _token_len(tok, text: str) -> int:  # noqa: ANN001
    return len(tok(text, add_special_tokens=False).input_ids)


def _clause_split(sentence: str) -> list[str]:
    return [p for p in _CLAUSE_SPLIT.split(sentence.strip()) if p]


def _hard_split(tok, text: str, max_span: int) -> list[str]:  # noqa: ANN001
    """Last resort: cut at max_span TOKENS (not a boundary a human would
    recognize) and decode each chunk back to text -- for a clause that still
    has no punctuation to split on. decode->re-encode is NOT length-stable
    for byte-level BPE (a cut mid character cluster can re-tokenize LONGER),
    and encode_canonical hard-asserts on an over-length piece -- so each
    chunk is shrunk until its DECODED TEXT re-tokenizes within max_span."""
    ids = tok(text, add_special_tokens=False).input_ids
    pieces: list[str] = []
    start = 0
    while start < len(ids):
        take = min(max_span, len(ids) - start)
        piece = tok.decode(ids[start : start + take], skip_special_tokens=True).strip()
        while take > 1 and _token_len(tok, piece) > max_span:
            take -= 1
            piece = tok.decode(ids[start : start + take], skip_special_tokens=True).strip()
        start += take
        if piece:  # a chunk of pure whitespace/special bytes decodes to ""
            pieces.append(piece)
    return pieces


def split_long_step(text: str, tok, max_span: int) -> list[str]:  # noqa: ANN001
    """A step's text -> 1+ pieces, each <= max_span tokens: sentence split,
    then (only on a still-too-long sentence) clause split on ',;:', then
    (only on a still-too-long clause) a hard token-count cut. Never drops
    text: `" ".join(split_long_step(text, ...))` reproduces `text` modulo
    whitespace."""
    if _token_len(tok, text) <= max_span:
        return [text]
    pieces: list[str] = []
    for sentence in split_sentences(text):
        if _token_len(tok, sentence) <= max_span:
            pieces.append(sentence)
            continue
        for clause in _clause_split(sentence):
            if _token_len(tok, clause) <= max_span:
                pieces.append(clause)
            else:
                pieces.extend(_hard_split(tok, clause, max_span))
    return pieces or [text]  # split_sentences found nothing to split on -- never empty


def _dict_sha1(dict_: dict) -> str:
    """Fingerprint of a dictionary's ACTUAL codebook content (its centroid
    tensors, not just its file/config name) -- a cache-key input, so
    switching --dict-name, or rebuilding the same-named dictionary with
    different fit data, invalidates every cached tokenization instead of
    silently reusing stale ids."""
    h = hashlib.sha1()
    h.update(dict_.get("cfg", "").encode())
    h.update(dict_.get("kind", "").encode())
    slots = dict_["entry"]["slots"] if dict_.get("kind") == "whole" else dict_.get("slots", [])
    for slot in slots:
        for key in sorted(slot):
            v = slot[key]
            if torch.is_tensor(v):
                h.update(key.encode())
                h.update(v.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _cache_key(text: str, settings: dict) -> str:
    payload = text + json.dumps(settings, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


class GistTokenizer:
    """encode_step/decode_ids: the stage-2 corpus's only entry point into the
    stage-1 dictionary machinery. `dict_` must carry its own "geometry" --
    every dictionary build_all_dicts registers does (see
    run_gist_dict._register) -- decode_ids reads it from there, so this class
    takes no separate geometry argument."""

    def __init__(self, pm, gist, dict_, tok, base=64, max_span=64, cache_dir=None):  # noqa: ANN001
        self.pm = pm
        self.gist = gist
        self.dict_ = dict_
        self.tok = tok
        self.base = base
        self.max_span = max_span
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.hits = 0
        self.misses = 0
        self._dict_sha1 = _dict_sha1(dict_)

    def _settings(self) -> dict:
        return {
            "model": getattr(self.tok, "name_or_path", "unknown"),
            "dict_sha1": self._dict_sha1,
            "base": self.base,
            "max_span": self.max_span,
            "batching": "single",
        }

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, text: str):
        if self.cache_dir is None:
            return None
        p = self._cache_path(_cache_key(text, self._settings()))
        if p.exists():
            self.hits += 1
            return json.loads(p.read_text())
        self.misses += 1
        return None

    def _cache_put(self, text: str, ids8: list[int]) -> None:
        if self.cache_dir is None:
            return
        p = self._cache_path(_cache_key(text, self._settings()))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ids8))

    def encode_step(self, text: str) -> list[list[int]]:
        """One [k_slots]-id group per piece of `text` (>=1; see
        split_long_step), in the dictionary's NATURAL slot order -- the
        SLOT_ORDER permutation for stage-3 layout/generation is that stage's
        job, not this tokenizer's."""
        out = []
        for piece in split_long_step(text, self.tok, self.max_span):
            cached = self._cache_get(piece)
            if cached is not None:
                out.append(cached)
                continue
            ids = self.tok(piece, add_special_tokens=False).input_ids
            kv, readout, _cs = encode_canonical(
                self.pm, self.gist, ids, base=self.base, max_span=self.max_span
            )
            ids8 = tokenize(kv, self.dict_, readout=readout)
            self._cache_put(piece, ids8)
            out.append(ids8)
        return out

    def decode_ids(self, ids8: list[int], dtype=None):  # noqa: ANN001
        return detokenize(ids8, self.dict_, self.dict_["geometry"], dtype=dtype)
