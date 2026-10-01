"""Command-line demo on the case-study data (needs the extras: uv sync --extra data --extra embeddings).

    uv run python -m review_assistant.demo "stopped connecting to wifi"
    uv run python -m review_assistant.demo "fishy burps" --store duckdb
    DATABASE_URL=postgresql://postgres:course@localhost:5432/postgres \
        uv run python -m review_assistant.demo "fishy burps" --store pgvector
    uv run python -m review_assistant.demo "Does the fish oil cause fishy burps?" --ask   # RAG answer via LLM_* env

`--embedder hashing` avoids the model download, but its results are poor: it only matches shared
words, and hash collisions add noise. It exists for the offline tests.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .retrieval import VectorRetriever
from .store import DuckDBVectorStore, NumpyVectorStore, PgVectorStore

DEFAULT_DATA = Path(__file__).resolve().parents[5] / "case-study" / "data"


def load_collection(data_dir: Path, n_products: int = 300, per_product: int = 20) -> dict[str, tuple[str, str]]:
    """The course collection: reviews of the most-reviewed products, {review_id: (parent_asin, text)}."""
    import pandas as pd

    reviews = pd.read_parquet(data_dir / "train_sample.parquet")
    top = reviews["parent_asin"].value_counts().index[:n_products]
    docs = reviews[reviews["parent_asin"].isin(top) & (reviews["text"].str.len() > 0)]
    docs = docs.groupby("parent_asin").head(per_product)
    return {r.review_id: (r.parent_asin, f"{r.title}. {r.text}") for r in docs.itertuples()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="minilm")
    parser.add_argument("--store", choices=["numpy", "duckdb", "pgvector"], default="numpy")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--ask", action="store_true", help="generate an answer with the configured LLM")
    args = parser.parse_args()

    collection = load_collection(args.data)
    chunks = chunk_documents({rid: text for rid, (_, text) in collection.items()})
    asins = [collection[c.doc_id][0] for c in chunks]
    embedder = SentenceTransformerEmbedder() if args.embedder == "minilm" else HashingEmbedder()
    store = {"numpy": lambda: NumpyVectorStore(embedder.dim),
             "duckdb": lambda: DuckDBVectorStore(embedder.dim),
             "pgvector": lambda: PgVectorStore(embedder.dim, os.environ["DATABASE_URL"])}[args.store]()
    retriever = VectorRetriever.from_chunks(chunks, embedder, parent_asins=asins, store=store)
    print(f"{len(collection)} reviews, {len(chunks)} chunks, store={args.store}, embedder={args.embedder}\n")

    if args.ask:
        from .llm import make_client
        from .rag import answer_question

        client, model = make_client()
        result = answer_question(args.query, retriever, client, model, k=args.k)
        print(result.answer, "\n\ncited:", result.cited_ids, "citation precision:", result.citation_precision)
        return
    for h in retriever.search(args.query, k=args.k):
        print(f"{h.score:.3f}  {h.doc_id}  {h.parent_asin}  {h.text[:90]}")


if __name__ == "__main__":
    main()
