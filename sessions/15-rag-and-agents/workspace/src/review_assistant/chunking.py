"""Chunking: split long documents into shorter, overlapping pieces before embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One piece of a document. `doc_id` lets us cite the original document."""

    chunk_id: str
    doc_id: str
    text: str


def chunk_words(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Split `text` into windows of `size` words that overlap by `overlap` words.

    >>> [len(c.split()) for c in chunk_words("w " * 250, size=120, overlap=20)]
    [120, 120, 50]
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    starts = range(0, max(len(words) - overlap, 1), step)
    return [" ".join(words[s : s + size]) for s in starts]


def chunk_sentences(text: str, max_words: int = 120) -> list[str]:
    """Split `text` at sentence ends and pack whole sentences into chunks of at most `max_words` words.

    TODO (exercise 2): implement this alternative to fixed word windows.
      1. Split the text into sentences with `SENTENCE_END.split(text.strip())`.
      2. Add sentences to the current chunk while the chunk stays within `max_words` words.
      3. A single sentence longer than `max_words` becomes a chunk of its own (do not cut it).
      4. Return a list of strings; an empty text gives an empty list.
    The tests in tests/test_exercises.py describe the expected behaviour.
    """
    raise NotImplementedError("exercise 2: implement chunk_sentences")


def chunk_documents(
    docs: dict[str, str], size: int = 120, overlap: int = 20
) -> list[Chunk]:
    """Chunk every document of a collection {doc_id: text}; chunk ids are '<doc_id>#<n>'."""
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        for n, piece in enumerate(chunk_words(text, size=size, overlap=overlap)):
            chunks.append(Chunk(chunk_id=f"{doc_id}#{n}", doc_id=doc_id, text=piece))
    return chunks
