"""Model-free tests for the gist data pipeline (gist_data.py). The HF streaming
call is a thin wrapper; the pairing / heldout / batching logic is pure."""

from __future__ import annotations

from marker.gist_data import batched, pairs_from_text, take_heldout


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


def test_take_heldout_pulls_first_n():
    stream = iter([("s", "c")] * 10)
    held, rest = take_heldout(stream, 3)
    assert len(held) == 3
    assert len(list(rest)) == 7  # the rest of the stream remains


def test_batched_groups_and_flushes_remainder():
    pairs = iter([([i], [i + 100]) for i in range(5)])
    batches = list(batched(pairs, batch_size=2))
    assert [len(s) for s, _ in batches] == [2, 2, 1]  # last partial batch flushed
    assert batches[0] == ([[0], [1]], [[100], [101]])
