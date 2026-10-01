"""Retrievers: embed the question, search the store, keep the best chunk per document."""

from __future__ import annotations

from collections.abc import Sequence

from .chunking import Chunk
from .embeddings import Embedder
from .hybrid import reciprocal_rank_fusion
from .keyword import BM25
from .store import Hit, NumpyVectorStore


class VectorRetriever:
    def __init__(self, embedder: Embedder, store) -> None:
        self.embedder, self.store = embedder, store

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], embedder: Embedder, parent_asins=None, store=None):
        store = store or NumpyVectorStore(embedder.dim)
        store.add(chunks, embedder.encode([c.text for c in chunks]), parent_asins)
        return cls(embedder, store)

    def search(self, query: str, k: int = 5, parent_asin: str | None = None) -> list[Hit]:
        vector = self.embedder.encode([query])[0]
        hits = self.store.search(vector, k=4 * k, parent_asin=parent_asin)  # extra, then dedupe
        best: dict[str, Hit] = {}
        for h in hits:
            best.setdefault(h.doc_id, h)  # hits are sorted, so the first one per document is its best
        return list(best.values())[:k]


class HybridRetriever:
    """BM25 keyword search and vector search, merged with reciprocal rank fusion."""

    def __init__(self, vector: VectorRetriever, chunks: Sequence[Chunk], parent_asins=None,
                 depth: int = 50) -> None:
        self.vector, self.depth = vector, depth
        self.by_doc = {c.doc_id: c for c in reversed(list(chunks))}  # first chunk per document
        asins = parent_asins if parent_asins is not None else [None] * len(chunks)
        self.asin_of = {c.doc_id: a for c, a in zip(chunks, asins, strict=True)}
        self.bm25 = BM25([c.doc_id for c in chunks], [c.text for c in chunks])

    def search(self, query: str, k: int = 5, parent_asin: str | None = None) -> list[Hit]:
        dense = self.vector.search(query, k=self.depth, parent_asin=parent_asin)
        keyword = [doc_id for doc_id, _ in self.bm25.search(query, k=self.depth)]
        if parent_asin is not None:  # the same metadata filter for the keyword ranking
            keyword = [d for d in keyword if self.asin_of.get(d) == parent_asin]
        fused = reciprocal_rank_fusion([[h.doc_id for h in dense], list(dict.fromkeys(keyword))])
        dense_by_doc = {h.doc_id: h for h in dense}
        out = []
        for doc_id in fused[:k]:
            hit = dense_by_doc.get(doc_id)
            if hit is None:
                c = self.by_doc[doc_id]
                hit = Hit(c.chunk_id, c.doc_id, c.text, 0.0, self.asin_of.get(doc_id))
            out.append(hit)
        return out
