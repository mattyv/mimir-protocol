"""Tests for run_tokenize_corpus.py, the stage-2 corpus builder (see
scratchpad/stage2_build_order.md, GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3").
Pure-logic pieces (answer extraction, question normalization, the exclusion
function, shard tallying/resume) run fast, no model; the end-to-end --smoke
manifest test uses a real tiny model and is marked slow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_gist_tokenizer import _FakeTok

# ── answer extraction ────────────────────────────────────────────────────────


def test_extract_answer_openr1_prefers_last_boxed():
    from marker.run_tokenize_corpus import extract_answer_openr1

    sol = r"We compute \boxed{3} first, then correct it: \boxed{42}."
    assert extract_answer_openr1(sol) == "42"


def test_extract_answer_openr1_handles_nested_braces():
    from marker.run_tokenize_corpus import extract_answer_openr1

    sol = r"So the answer is \boxed{\frac{1}{2}}."
    assert extract_answer_openr1(sol) == r"\frac{1}{2}"


def test_extract_answer_openr1_falls_back_to_last_nonempty_line():
    from marker.run_tokenize_corpus import extract_answer_openr1

    sol = "step one\nstep two\n\nthe final result is 7\n"
    assert extract_answer_openr1(sol) == "the final result is 7"


def test_extract_answer_openr1_none_when_solution_is_empty():
    from marker.run_tokenize_corpus import extract_answer_openr1

    assert extract_answer_openr1("   \n  ") is None


def test_extract_answer_dispatches_by_src():
    from marker.run_tokenize_corpus import extract_answer

    gsm8k_sol = "Step one.\nStep two.\n#### 12"
    assert extract_answer("gsm8k", gsm8k_sol) == "12"
    assert extract_answer("openr1", r"final: \boxed{5}") == "5"


# ── normalize_question ───────────────────────────────────────────────────────


def test_normalize_question_lowercases_collapses_whitespace_strips_punctuation():
    from marker.run_tokenize_corpus import normalize_question

    a = normalize_question("  What  is 2+2,  really?! ")
    b = normalize_question("what is 2+2 really")
    assert a == b == "what is 22 really"


# ── exclude_doc: one reason at a time ───────────────────────────────────────


def _excl(**overrides):
    base = {
        "tok": _FakeTok(),
        "max_span": 8,
        "max_groups": 4,
        "seq_cap": 512,
        "norm_questions": set(),
        "k_slots": 8,
        "splitter": lambda s: s.split("\n"),
    }
    base.update(overrides)
    return base


def test_exclude_doc_keeps_a_normal_doc():
    from marker.run_tokenize_corpus import exclude_doc

    assert exclude_doc("what is 1 plus 1", "a short step\nanother short step", _excl()) is None


def test_exclude_doc_contaminated_matches_normalized_question():
    from marker.run_tokenize_corpus import exclude_doc, normalize_question

    q = "What is 2 + 2?"
    excl = _excl(norm_questions={normalize_question(q)})
    assert exclude_doc(q, "a step\nanother step", excl) == "contaminated"


def test_exclude_doc_pathological_when_a_step_needs_too_many_groups():
    from marker.run_tokenize_corpus import exclude_doc

    # max_span=8 fake-tokens (words), no punctuation at all -> hard-split into
    # ceil(40/8)=5 pieces, strictly over max_groups=4
    long_step = " ".join(f"w{i}" for i in range(40))
    excl = _excl(max_span=8, max_groups=4)
    assert exclude_doc("q", long_step, excl) == "pathological"


def test_exclude_doc_seq_cap_when_laid_out_sequence_too_long():
    from marker.run_tokenize_corpus import exclude_doc

    question = " ".join(f"q{i}" for i in range(20))
    steps = "\n".join(" ".join(f"s{i}{j}" for j in range(6)) for i in range(3))
    excl = _excl(max_span=8, max_groups=4, seq_cap=10, k_slots=8)
    assert exclude_doc(question, steps, excl) == "seq_cap"


def test_exclude_doc_reason_priority_contaminated_before_others():
    from marker.run_tokenize_corpus import exclude_doc, normalize_question

    q = "duplicate question"
    excl = _excl(norm_questions={normalize_question(q)}, seq_cap=1)  # would ALSO fail seq_cap
    assert exclude_doc(q, "a step", excl) == "contaminated"


# ── list_existing_shards / write_jsonl / _tally_shards ──────────────────────


def test_list_existing_shards_filters_to_subdir_shard_jsonl_names():
    from marker.run_tokenize_corpus import list_existing_shards

    files = [
        "gist_corpus_K4096/shard_0000.jsonl",
        "gist_corpus_K4096/shard_0001.jsonl",
        "gist_corpus_K4096/manifest.json",
        "gist_corpus_K4096/eval_gsm8k_test.jsonl",
        "other_subdir/shard_0000.jsonl",
    ]
    got = list_existing_shards("user/repo", "gist_corpus_K4096", lister=lambda repo: files)
    assert got == {"shard_0000.jsonl", "shard_0001.jsonl"}


def test_list_existing_shards_empty_without_out_repo():
    from marker.run_tokenize_corpus import list_existing_shards

    assert list_existing_shards(None, "sub") == set()


def test_write_jsonl_and_tally_shards_round_trip(tmp_path):
    from marker.run_tokenize_corpus import _tally_shards, write_jsonl

    records = [
        {"steps": ["a", "b"], "ids": [[1], [2, 3]], "n_groups": 3},
        {"steps": ["c"], "ids": [[4]], "n_groups": 1},
    ]
    p = tmp_path / "shard_0000.jsonl"
    write_jsonl(records, p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == records[0]

    tally = _tally_shards([p])
    assert tally == {"docs": 2, "steps": 3, "groups": 4, "split_steps": 1}


# ── build_corpus: fresh build, then resume with more docs ──────────────────


class _SpyGistTokenizer:
    """A GistTokenizer stand-in: encode_step returns one [k_slots]-id group
    per NEWLINE-separated 'piece' -- deterministic, no model, and counts
    calls so a resumed run's skip-existing-shards behaviour is provable."""

    def __init__(self, k_slots=2):
        self.k_slots = k_slots
        self.calls = 0
        self.hits = 3
        self.misses = 5

    def encode_step(self, text):
        self.calls += 1
        return [[0] * self.k_slots]


def _sources(docs, src="gsm8k"):
    return [(src, [(f"q{i}", d) for i, d in enumerate(docs)], lambda s: s.split("\n"))]


def test_build_corpus_writes_shards_and_counts_reasons(tmp_path):
    from marker.run_tokenize_corpus import _tally_shards, build_corpus

    docs = [f"step a{i}\nstep b{i}" for i in range(5)]
    gtok = _SpyGistTokenizer()
    excl = _excl(norm_questions=set())
    out_dir = tmp_path / "shards"
    shard_events = []
    counts = build_corpus(
        _sources(docs),
        excl,
        gtok,
        shard_size=2,
        out_dir=out_dir,
        existing_shards=set(),
        on_shard=lambda name, path: shard_events.append(name),
    )
    assert shard_events == ["shard_0000.jsonl", "shard_0001.jsonl", "shard_0002.jsonl"]
    assert counts["docs_seen"] == {"gsm8k": 5}
    assert counts["excluded"] == {}
    assert gtok.calls == 10  # 5 docs x 2 steps each
    tally = _tally_shards(sorted(out_dir.glob("shard_*.jsonl")))
    assert tally["docs"] == 5
    assert tally["steps"] == 10


def test_build_corpus_resume_skips_existing_shards_and_continues_numbering(tmp_path):
    from marker.run_tokenize_corpus import _tally_shards, build_corpus

    docs = [f"step a{i}\nstep b{i}" for i in range(5)]
    gtok1 = _SpyGistTokenizer()
    out_dir = tmp_path / "shards"
    build_corpus(
        _sources(docs), _excl(), gtok1, shard_size=2, out_dir=out_dir, existing_shards=set()
    )
    assert gtok1.calls == 10

    # a SECOND run over MORE docs (20k -> 57k-style growth), told shards 0
    # and 1 already exist -- it must not re-encode their 4 docs, but MUST
    # still write shard 2 (already produced last time, re-encoded here since
    # it wasn't marked existing) and the NEW shard 3 from the extra doc
    more_docs = docs + [f"step a{i}\nstep b{i}" for i in range(5, 7)]
    gtok2 = _SpyGistTokenizer()
    shard_events = []
    build_corpus(
        _sources(more_docs),
        _excl(),
        gtok2,
        shard_size=2,
        out_dir=out_dir,
        existing_shards={"shard_0000.jsonl", "shard_0001.jsonl"},
        on_shard=lambda name, path: shard_events.append(name),
    )
    assert shard_events == ["shard_0002.jsonl", "shard_0003.jsonl"]
    assert gtok2.calls == 6  # docs 4,5,6 (2 steps each) -- docs 0-3 skipped
    tally = _tally_shards(sorted(out_dir.glob("shard_*.jsonl")))
    assert tally["docs"] == 7  # shards 0,1 untouched (still their old 4 docs) + 3 new


def test_complete_shards_partial_trailing_shard_is_not_complete(tmp_path):
    """The resume data-loss guard: a shard with fewer than shard_size records
    (the trailing shard of a shorter earlier run) must NOT be skippable --
    build_corpus drops every doc numbered into a skipped shard, so treating
    a partial shard as complete silently loses the docs a larger resume
    numbers into its unfilled tail."""
    from marker.run_tokenize_corpus import complete_shards, write_jsonl

    rec = {"steps": ["a"], "ids": [[1]], "n_groups": 1}
    write_jsonl([rec, rec], tmp_path / "shard_0000.jsonl")  # full
    write_jsonl([rec], tmp_path / "shard_0001.jsonl")  # partial trailing
    names = {"shard_0000.jsonl", "shard_0001.jsonl", "shard_0002.jsonl"}  # 0002 never downloaded
    assert complete_shards(tmp_path, names, shard_size=2) == {"shard_0000.jsonl"}


def test_resume_after_partial_shard_loses_no_docs(tmp_path):
    """End to end over build_corpus: run 1 stops with a partial trailing
    shard; run 2 (more docs, existing_shards filtered through
    complete_shards) must re-encode the partial shard so its tail fills --
    every doc lands in exactly one shard, none lost."""
    from marker.run_tokenize_corpus import _tally_shards, build_corpus, complete_shards

    out_dir = tmp_path / "shards"
    docs = [f"step a{i}\nstep b{i}" for i in range(3)]  # run 1: shard 0 full, shard 1 partial
    build_corpus(_sources(docs), _excl(), _SpyGistTokenizer(), 2, out_dir, existing_shards=set())

    more_docs = docs + [f"step a{i}\nstep b{i}" for i in range(3, 5)]  # run 2: 5 docs
    on_hf = {"shard_0000.jsonl", "shard_0001.jsonl"}
    existing = complete_shards(out_dir, on_hf, shard_size=2)
    assert existing == {"shard_0000.jsonl"}
    gtok2 = _SpyGistTokenizer()
    build_corpus(_sources(more_docs), _excl(), gtok2, 2, out_dir, existing_shards=existing)
    assert gtok2.calls == 6  # docs 2,3,4 re/newly encoded; docs 0,1 skipped
    tally = _tally_shards(sorted(out_dir.glob("shard_*.jsonl")))
    assert tally["docs"] == 5  # nothing lost: 2 + 2 + 1 across shards 0,1,2


# ── end-to-end smoke ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_smoke_manifest_runs_full_pipeline_and_writes_shards():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "marker.run_tokenize_corpus", "--smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": "src" + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + "\n" + proc.stderr[-6000:]
    (line,) = (
        line_
        for line_ in proc.stdout.splitlines()
        if line_.startswith("[TOKENIZE_CORPUS MANIFEST]")
    )
    manifest = json.loads(line[len("[TOKENIZE_CORPUS MANIFEST] ") :])

    assert manifest["batching"] == "single"
    assert manifest["slot_order_note"] == "ids stored in natural order"
    assert "dict_sha1" in manifest
    counts = manifest["counts"]
    assert counts["docs_seen"] == {"gsm8k": 15, "openr1": 15}
    assert "cache_hits" in counts and "cache_misses" in counts
    assert isinstance(counts["excluded"], dict)
    assert manifest["n_eval_records"] > 0

    shard_dir = Path("/tmp/gist_corpus_cache_smoke/shards")  # noqa: S108
    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    assert len(shards) >= 1
    (eval_path,) = shard_dir.glob("eval_gsm8k_test.jsonl")
    rec = json.loads(eval_path.read_text().splitlines()[0])
    assert set(rec) == {
        "src",
        "doc_id",
        "question",
        "steps",
        "ids",
        "answer",
        "n_groups",
        "eval_step_index",
    }
    assert 0 <= rec["eval_step_index"] < len(rec["steps"])
    widths = {len(group) for step_groups in [rec["ids"]] for group in step_groups}
    assert len(widths) == 1  # every group the same k_slots width

    rec2 = json.loads(shards[0].read_text().splitlines()[0])
    assert set(rec2) == {"src", "doc_id", "question", "steps", "ids", "answer", "n_groups"}
    assert rec2["src"] in {"gsm8k", "openr1"}
