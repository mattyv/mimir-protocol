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


def stream_doc_pairs(
    tokenizer,  # noqa: ANN001
    max_span: int = 64,
    max_cont: int = 64,
    min_cont: int = 16,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    text_key: str = "text",
) -> Iterator[list[tuple[list[int], list[int]]]]:
    """Lazily stream the corpus, yielding each DOCUMENT's pairs as one list
    (empty docs skipped). Document granularity matters: pairs within a doc
    overlap heavily (sentence i's continuation contains sentence i+1's span),
    so any heldout/train split must happen at doc boundaries or the eval
    leaks into training (Fable review finding #2 — the v10 lesson)."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    for row in ds:
        text = row.get(text_key) or ""
        pairs = pairs_from_text(text, tokenizer, max_span, max_cont, min_cont)
        if pairs:
            yield pairs


def take_heldout_docs(
    doc_iter: Iterator[list[tuple[list[int], list[int]]]], min_pairs: int
) -> tuple[list[tuple[list[int], list[int]]], Iterator[tuple[list[int], list[int]]]]:
    """Accumulate WHOLE documents into the held-out set until it has at least
    min_pairs pairs; return (heldout_pairs, train_pair_iterator). The training
    iterator starts at the next document, so heldout and train are
    document-disjoint by construction."""
    heldout: list[tuple[list[int], list[int]]] = []
    for doc_pairs in doc_iter:
        heldout.extend(doc_pairs)
        if len(heldout) >= min_pairs:
            break

    def _train() -> Iterator[tuple[list[int], list[int]]]:
        for doc_pairs in doc_iter:
            yield from doc_pairs

    return heldout, _train()


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
