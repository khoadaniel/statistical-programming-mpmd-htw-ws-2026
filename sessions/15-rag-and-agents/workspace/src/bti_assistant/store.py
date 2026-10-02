"""Vector stores with one interface: `add(chunks, vectors, headings)` and `search(vector, k, chapter)`.

- `NumpyVectorStore`: everything in memory; exact search with one matrix product.
- `DuckDBVectorStore`: an in-process SQL database; cosine similarity in SQL, no server needed.
- `PgVectorStore`: PostgreSQL with the pgvector extension, the course path (see README).
Every chunk carries the heading of its decision as metadata; `chapter="64"` restricts a search to the
decisions of one HS chapter (the first two digits of the heading).
All vectors must have length 1 (use `embeddings.normalise`), so that cosine similarity is a dot product.
"""

from __future__ import annotations

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


class DuckDBVectorStore:
    """Chunks and embeddings in a DuckDB table; `array_cosine_similarity` ranks them."""

    def __init__(self, dim: int, path: str = ":memory:") -> None:
        import duckdb

        self.dim = dim
        self.con = duckdb.connect(path)
        self.con.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (chunk_id VARCHAR PRIMARY KEY, doc_id VARCHAR,
                heading VARCHAR, text VARCHAR, embedding FLOAT[{dim}])"""
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, headings=None) -> None:
        heads = _check(chunks, vectors, headings)
        rows = [
            (c.chunk_id, c.doc_id, h, c.text, [float(x) for x in v])
            for c, h, v in zip(chunks, heads, vectors, strict=True)
        ]
        self.con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)

    def __len__(self) -> int:
        return self.con.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(self, vector: np.ndarray, k: int = 5, chapter: str | None = None) -> list[Hit]:
        sql = f"""
            SELECT chunk_id, doc_id, text, array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS score,
                   heading
            FROM chunks
            WHERE ?::VARCHAR IS NULL OR left(heading, 2) = ?
            ORDER BY score DESC
            LIMIT ?"""
        q = [float(x) for x in vector]
        rows = self.con.execute(sql, [q, chapter, chapter, k]).fetchall()
        return [Hit(r[0], r[1], r[2], float(r[3]), r[4]) for r in rows]


class PgVectorStore:
    """PostgreSQL with pgvector. Start the database with Docker (see README):

        docker run --name pgvector -e POSTGRES_PASSWORD=course -p 5432:5432 -d pgvector/pgvector:pg17

    and connect with DATABASE_URL=postgresql://postgres:course@localhost:5432/postgres.
    Requires the optional dependencies: uv sync --extra postgres
    """

    def __init__(self, dim: int, url: str, table: str = "decision_chunks") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim, self.table = dim, table
        self.con = psycopg.connect(url, autocommit=True)
        self.con.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.con)  # numpy arrays <-> the vector type
        self.con.execute(
            f"""CREATE TABLE IF NOT EXISTS {table} (chunk_id text PRIMARY KEY, doc_id text,
                heading text, text text, embedding vector({dim}))"""
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, headings=None) -> None:
        heads = _check(chunks, vectors, headings)
        with self.con.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} VALUES (%s, %s, %s, %s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                [(c.chunk_id, c.doc_id, h, c.text, np.asarray(v, dtype=np.float32))
                 for c, h, v in zip(chunks, heads, vectors, strict=True)],
            )

    def create_index(self) -> None:
        """Approximate nearest-neighbour index (HNSW) for cosine distance, the operator <=>."""
        self.con.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
            "USING hnsw (embedding vector_cosine_ops)"
        )

    def search(self, vector: np.ndarray, k: int = 5, chapter: str | None = None) -> list[Hit]:
        if chapter is not None:
            # TODO (exercise 1): add the metadata filter. Extend the SQL below with
            #   WHERE left(heading, 2) = %s
            # before ORDER BY, and pass chapter as an extra parameter in the right position.
            raise NotImplementedError("exercise 1: filter by chapter in SQL")
        q = np.asarray(vector, dtype=np.float32)
        rows = self.con.execute(
            f"""SELECT chunk_id, doc_id, text, 1 - (embedding <=> %s) AS score, heading
                FROM {self.table}
                ORDER BY embedding <=> %s   -- <=> is the cosine distance, 1 - cosine similarity
                LIMIT %s""",
            (q, q, k),
        ).fetchall()
        return [Hit(r[0], r[1], r[2], float(r[3]), r[4]) for r in rows]
