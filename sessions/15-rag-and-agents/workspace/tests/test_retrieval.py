import numpy as np
import pytest

from review_assistant.embeddings import HashingEmbedder, normalise
from review_assistant.keyword import BM25
from review_assistant.retrieval import HybridRetriever, VectorRetriever
from review_assistant.store import DuckDBVectorStore, NumpyVectorStore


def test_embeddings_have_unit_length_and_are_deterministic():
    emb = HashingEmbedder(dim=64)
    a, b = emb.encode(["fishy burps", "fishy burps"]), emb.encode([""])
    assert a.shape == (2, 64)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a[0], a[1])
    assert np.allclose(b, 0)  # an empty text gives the zero vector, not NaN


def test_normalise():
    v = normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(v, [[0.6, 0.8], [0.0, 0.0]])


def test_vector_search_finds_the_relevant_review(retriever):
    hits = retriever.search("fishy burps", k=2)
    assert {h.doc_id for h in hits} == {"r4", "r5"}
    assert hits[0].score >= hits[1].score


def test_one_hit_per_document(retriever):
    ids = [h.doc_id for h in retriever.search("scale wifi", k=5)]
    assert len(ids) == len(set(ids))


def test_metadata_filter(retriever):
    hits = retriever.search("stopped working", k=5, parent_asin="PAD")
    assert hits and all(h.parent_asin == "PAD" for h in hits)


@pytest.mark.parametrize("store_cls", [NumpyVectorStore, DuckDBVectorStore])
def test_stores_return_the_same_ranking(store_cls, chunks, asins):
    emb = HashingEmbedder(dim=128)
    store = store_cls(emb.dim)
    store.add(chunks, emb.encode([c.text for c in chunks]), asins)
    q = emb.encode(["heating pad stopped working"])[0]
    reference = NumpyVectorStore(emb.dim)
    reference.add(chunks, emb.encode([c.text for c in chunks]), asins)
    got, want = store.search(q, k=3), reference.search(q, k=3)
    assert [h.chunk_id for h in got] == [h.chunk_id for h in want]
    assert np.allclose([h.score for h in got], [h.score for h in want], atol=1e-5)
    assert all(h.parent_asin == "SCALE" for h in store.search(q, k=3, parent_asin="SCALE"))
    assert len(store) == len(chunks)


def test_store_rejects_mismatched_input(chunks):
    with pytest.raises(ValueError):
        NumpyVectorStore(4).add(chunks, np.zeros((1, 4)))


def test_bm25_prefers_documents_with_rare_query_words():
    bm25 = BM25(["a", "b", "c"], ["the pad is hot", "the scale is the best", "the pad stopped working"])
    ranked = [i for i, _ in bm25.search("pad stopped")]
    assert ranked[0] == "c" and "b" not in ranked


def test_hybrid_combines_both_rankings(chunks, asins):
    vector = VectorRetriever.from_chunks(chunks, HashingEmbedder(dim=256), parent_asins=asins)
    hybrid = HybridRetriever(vector, chunks, parent_asins=asins)
    ids = [h.doc_id for h in hybrid.search("fishy burps", k=3)]
    assert set(ids[:2]) == {"r4", "r5"}
    assert all(h.parent_asin == "PAD" for h in hybrid.search("hot", k=3, parent_asin="PAD"))
