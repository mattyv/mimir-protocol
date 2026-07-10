"""Model-free tests for the gist data pipeline (gist_data.py). The HF streaming
call is a thin wrapper; the pairing / heldout / batching logic is pure."""

from __future__ import annotations

from marker.gist_data import batched, pairs_from_text, take_heldout_docs


class _CharTok:
    """Fake tokenizer: one id per non-space char."""

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001
        ids = [ord(c) for c in text if not c.isspace()]
        return type("Enc", (), {"input_ids": ids})()


def test_pairs_span_is_sentence_cont_is_the_rest():
    tok = _CharTok()
    pairs = pairs_from_text("Aaaa. Bbbb. Cccc.", tok, max_span=10, max_cont=10, min_cont=1)
    # 3 sentences -> 2 pairs (last has no continuation)
    assert len(pairs) == 2
    span0, cont0 = pairs[0]
    assert span0 == [ord(c) for c in "Aaaa."]  # first sentence
    assert cont0[: len("Bbbb.")] == [ord(c) for c in "Bbbb."]  # continuation starts w/ next


def test_pairs_drops_short_continuation():
    tok = _CharTok()
    # min_cont high -> the tail sentence is too short to qualify
    pairs = pairs_from_text("Hello world. Hi.", tok, max_span=20, max_cont=20, min_cont=10)
    assert pairs == []  # "Hi." is only 3 chars < min_cont 10


def test_pairs_single_sentence_yields_nothing():
    tok = _CharTok()
    assert pairs_from_text("Just one sentence here", tok, 20, 20, 1) == []


def test_take_heldout_docs_is_document_disjoint():
    # doc A has 2 pairs, doc B has 1, doc C has 2. min_pairs=3 -> heldout takes
    # ALL of A and B (whole docs); training starts at doc C.
    docs = iter(
        [
            [("a1", "x"), ("a2", "x")],
            [("b1", "x")],
            [("c1", "x"), ("c2", "x")],
        ]
    )
    held, train = take_heldout_docs(docs, min_pairs=3)
    assert [p[0] for p in held] == ["a1", "a2", "b1"]
    train_list = list(train)
    assert [p[0] for p in train_list] == ["c1", "c2"]
    # no pair from a heldout doc ever appears in training
    assert not {p[0][0] for p in train_list} & {"a", "b"}


def test_take_heldout_docs_never_splits_a_document():
    # min_pairs=1 but the first doc has 3 pairs: heldout takes all 3, not 1.
    docs = iter([[("a1", "x"), ("a2", "x"), ("a3", "x")], [("b1", "x")]])
    held, train = take_heldout_docs(docs, min_pairs=1)
    assert len(held) == 3
    assert [p[0] for p in list(train)] == ["b1"]


def test_batched_groups_and_flushes_remainder():
    pairs = iter([([i], [i + 100]) for i in range(5)])
    batches = list(batched(pairs, batch_size=2))
    assert [len(s) for s, _ in batches] == [2, 2, 1]  # last partial batch flushed
    assert batches[0] == ([[0], [1]], [[100], [101]])
