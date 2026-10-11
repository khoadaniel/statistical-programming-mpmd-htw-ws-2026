"""Vector stores with one interface: `add(chunks, vectors, headings)` and `search(vector, k, chapter)`.

- `NumpyVectorStore`: everything in memory; exact search with one matrix product.
- `SQLiteVectorStore`: chunks and metadata in an SQLite table (`sqlite3`, part of Python), embeddings as
  float32 BLOBs in the same table; SQL selects the rows (the chapter filter: exercise 1), NumPy ranks them.
  With a file path the index survives a restart.
Every chunk carries the heading of its decision as metadata; `chapter="64"` restricts a search to the
decisions of one HS chapter (the first two digits of the heading).
All vectors must have length 1 (use `embeddings.normalise`), so that cosine similarity is a dot product.

Exact (brute-force) search compares the query with every stored vector. For collections of 10^4 to 10^5
chunks that is one matrix product of a few milliseconds; for millions of vectors, an approximate index in a
vector database or a database extension is the usual next step.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str  # the BTI reference of the decision
    text: str
    score: float
    heading: str | None = None


def _check(chunks: Sequence[Chunk], vectors: np.ndarray, headings: Sequence[str | None] | None):
    if len(chunks) != len(vectors):
        raise ValueError("one vector per chunk is needed")
    if headings is not None and len(headings) != len(chunks):
        raise ValueError("one heading per chunk is needed")
    return list(headings) if headings is not None else [None] * len(chunks)


class NumpyVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []
        self.headings: list[str | None] = []

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, headings=None) -> None:
        heads = _check(chunks, vectors, headings)
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, dtype=np.float32)])
        self.chunks.extend(chunks)
        self.headings.extend(heads)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, vector: np.ndarray, k: int = 5, chapter: str | None = None) -> list[Hit]:
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)  # cosine similarity
        if chapter is not None:  # metadata filter
            mask = np.array([h is not None and h[:2] == chapter for h in self.headings])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(self.chunks[i].chunk_id, self.chunks[i].doc_id, self.chunks[i].text,
                float(scores[i]), self.headings[i])
            for i in order
            if np.isfinite(scores[i])
        ]


class SQLiteVectorStore:
    """Chunks, headings and embeddings in one SQLite table; cosine similarity computed in NumPy.

    `path=":memory:"` keeps the table in memory; a file path (in an ignored folder, never in the repository)
    keeps it on disk, so that the collection is embedded only once.
    """

    def __init__(self, dim: int, path: str = ":memory:") -> None:
        self.dim = dim
        self.con = sqlite3.connect(path)
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT,
               heading TEXT, text TEXT, embedding BLOB)"""
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, headings=None) -> None:
        heads = _check(chunks, vectors, headings)
        rows = [
            (c.chunk_id, c.doc_id, h, c.text, np.asarray(v, dtype=np.float32).tobytes())
            for c, h, v in zip(chunks, heads, vectors, strict=True)
        ]
        with self.con:  # one transaction
            self.con.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?, ?)", rows)

    def __len__(self) -> int:
        return self.con.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(self, vector: np.ndarray, k: int = 5, chapter: str | None = None) -> list[Hit]:
        if chapter is not None:
            # TODO (exercise 1): add the metadata filter in SQL. Extend the query below with
            #   WHERE substr(heading, 1, 2) = ?
            # and pass chapter as a parameter, so that only the rows of that chapter are loaded and ranked.
            raise NotImplementedError("exercise 1: filter by chapter in SQL")
        rows = self.con.execute("SELECT chunk_id, doc_id, text, heading, embedding FROM chunks").fetchall()
        return _rank(rows, vector, k, self.dim)


def _rank(rows: list[tuple], vector: np.ndarray, k: int, dim: int) -> list[Hit]:
    """Rank rows (chunk_id, doc_id, text, heading, embedding BLOB) by cosine similarity to the query."""
    if not rows:
        return []
    matrix = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), dim)
    scores = matrix @ np.asarray(vector, dtype=np.float32)  # cosine similarity: all vectors have length 1
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(rows[i][0], rows[i][1], rows[i][2], float(scores[i]), rows[i][3]) for i in order]
