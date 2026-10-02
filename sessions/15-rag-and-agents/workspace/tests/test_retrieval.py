import numpy as np
import pytest

from bti_assistant.embeddings import HashingEmbedder, normalise
from bti_assistant.keyword import BM25
from bti_assistant.retrieval import HybridRetriever, VectorRetriever
from bti_assistant.store import DuckDBVectorStore, NumpyVectorStore


def test_embeddings_have_unit_length_and_are_deterministic():
    emb = HashingEmbedder(dim=64)
    a, b = emb.encode(["Laufsohle aus Gummi", "Laufsohle aus Gummi"]), emb.encode([""])
    assert a.shape == (2, 64)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a[0], a[1])
    assert np.allclose(b, 0)  # an empty text gives the zero vector, not NaN


def test_hashing_tokens_keep_umlauts_and_accents():
    emb = HashingEmbedder(dim=4096)
    same = emb.encode(["Rädern", "rädern"])
    assert np.allclose(same[0], same[1])
    assert not np.allclose(emb.encode(["Rädern"])[0], emb.encode(["Radern"])[0])


def test_normalise():
    v = normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(v, [[0.6, 0.8], [0.0, 0.0]])


def test_vector_search_finds_the_relevant_decisions(retriever):
    hits = retriever.search("Oberteil Laufsohle", k=2)
    assert {h.doc_id for h in hits} == {"DE-001/21", "DE-002/21"}
    assert hits[0].score >= hits[1].score
    assert {h.heading for h in hits} == {"6403", "6404"}


def test_one_hit_per_decision(retriever):
    ids = [h.doc_id for h in retriever.search("mask nonwoven fabric", k=5)]
    assert len(ids) == len(set(ids))


def test_chapter_filter(retriever):
    hits = retriever.search("kunststoff", k=5, chapter="64")
    assert hits and all(h.heading.startswith("64") for h in hits)


@pytest.mark.parametrize("store_cls", [NumpyVectorStore, DuckDBVectorStore])
def test_stores_return_the_same_ranking(store_cls, chunks, headings):
    emb = HashingEmbedder(dim=128)
    store = store_cls(emb.dim)
    store.add(chunks, emb.encode([c.text for c in chunks]), headings)
    q = emb.encode(["Spielzeug Kunststoff kinderen"])[0]
    reference = NumpyVectorStore(emb.dim)
    reference.add(chunks, emb.encode([c.text for c in chunks]), headings)
    got, want = store.search(q, k=3), reference.search(q, k=3)
    assert [h.chunk_id for h in got] == [h.chunk_id for h in want]
    assert np.allclose([h.score for h in got], [h.score for h in want], atol=1e-5)
    assert all(h.heading.startswith("95") for h in store.search(q, k=3, chapter="95"))
    assert len(store) == len(chunks)


def test_store_rejects_mismatched_input(chunks):
    with pytest.raises(ValueError):
        NumpyVectorStore(4).add(chunks, np.zeros((1, 4)))


def test_bm25_prefers_documents_with_rare_query_words():
    bm25 = BM25(["a", "b", "c"], ["Schuhe aus Leder", "Spielzeug aus Kunststoff", "Stiefel aus Leder mit Futter"])
    ranked = [i for i, _ in bm25.search("Stiefel Leder")]
    assert ranked[0] == "c" and "b" not in ranked


def test_hybrid_combines_both_rankings(chunks, headings):
    vector = VectorRetriever.from_chunks(chunks, HashingEmbedder(dim=256), headings=headings)
    hybrid = HybridRetriever(vector, chunks, headings=headings)
    ids = [h.doc_id for h in hybrid.search("Oberteil Laufsohle", k=3)]
    assert set(ids[:2]) == {"DE-001/21", "DE-002/21"}
    assert all(h.heading.startswith("63") for h in hybrid.search("mask", k=3, chapter="63"))
