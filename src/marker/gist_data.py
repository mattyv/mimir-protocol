"""Stage-1 data: stream raw web text, emit (span, continuation) pairs.

A pair is one sentence (the span S, capped at max_span subwords) and the text
that follows it in the same document (the continuation C, capped at max_cont,
dropped if shorter than min_cont). Self-supervised: labels are free.

The streaming call (HF datasets) is a thin wrapper; the pairing logic
(pairs_from_text) is pure and unit-tested with a fake tokenizer.
"""

from __future__ import annotations

from collections.abc import Iterator

from marker.gist import make_pair, split_sentences

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"


def pairs_from_text(
    text: str,
    tokenizer,  # noqa: ANN001
    max_span: int,
    max_cont: int,
    min_cont: int,
) -> list[tuple[list[int], list[int]]]:
    """Every sentence in `text` becomes a span; its continuation is the rest of
    the document after it (joined following sentences), capped/filtered by
    make_pair. Returns the valid (span_ids, cont_ids) pairs."""
    sents = split_sentences(text)
    pairs = []
    for i in range(len(sents) - 1):
        continuation = " ".join(sents[i + 1 :])
        pair = make_pair(tokenizer, sents[i], continuation, max_span, max_cont, min_cont)
        if pair is not None:
            pairs.append(pair)
    return pairs


def stream_pairs(
    tokenizer,  # noqa: ANN001
    max_span: int = 64,
    max_cont: int = 64,
    min_cont: int = 16,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    text_key: str = "text",
) -> Iterator[tuple[list[int], list[int]]]:
    """Lazily stream the corpus and yield (span, cont) pairs one at a time.
    Streaming = no full download; the iterator is effectively unbounded."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    for row in ds:
        text = row.get(text_key) or ""
        yield from pairs_from_text(text, tokenizer, max_span, max_cont, min_cont)


def take_heldout(
    pair_iter: Iterator[tuple[list[int], list[int]]], n: int
) -> tuple[list[tuple[list[int], list[int]]], Iterator[tuple[list[int], list[int]]]]:
    """Pull the first n pairs off the stream as a fixed held-out eval set;
    return (heldout, remaining_iterator). Deterministic given a deterministic
    stream — the eval set is the first n pairs, set aside before training."""
    heldout = []
    for pair in pair_iter:
        heldout.append(pair)
        if len(heldout) >= n:
            break
    return heldout, pair_iter


def batched(
    pair_iter: Iterator[tuple[list[int], list[int]]], batch_size: int
) -> Iterator[tuple[list[list[int]], list[list[int]]]]:
    """Group pairs into (spans, conts) batches."""
    spans, conts = [], []
    for span, cont in pair_iter:
        spans.append(span)
        conts.append(cont)
        if len(spans) == batch_size:
            yield spans, conts
            spans, conts = [], []
    if spans:
        yield spans, conts
