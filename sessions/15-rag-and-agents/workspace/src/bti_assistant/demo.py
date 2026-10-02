"""Command-line demo on the case-study data (needs the extras: uv sync --extra data --extra embeddings).

    uv run python -m bti_assistant.demo "Damenstiefel mit Oberteil aus Leder"
    uv run python -m bti_assistant.demo "plastic toy car" --store duckdb
    DATABASE_URL=postgresql://postgres:course@localhost:5432/postgres \\
        uv run python -m bti_assistant.demo "plastic toy car" --store pgvector
    uv run python -m bti_assistant.demo "Damenstiefel mit Oberteil aus Leder" --ask   # heading suggestion via LLM_* env

`--embedder hashing` avoids the model download, but its results are poor: it only matches shared
words (so it cannot search across languages), and hash collisions add noise. It exists for the tests.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .rag import vote_headings
from .retrieval import VectorRetriever
from .store import DuckDBVectorStore, NumpyVectorStore, PgVectorStore

DEFAULT_DATA = Path(__file__).resolve().parents[5] / "case-study" / "data"


def load_collection(data_dir: Path, n: int = 5000, seed: int = 0) -> dict[str, tuple[str, str]]:
    """The course collection: a random sample of training decisions, {bti_reference: (heading, description)}."""
    import pandas as pd

    d = pd.read_parquet(data_dir / "train_sample.parquet", columns=["bti_reference", "heading", "description"])
    d = d.drop_duplicates("bti_reference").sample(n, random_state=seed)
    return {r.bti_reference: (r.heading, r.description) for r in d.itertuples()}


def load_heading_texts(data_dir: Path) -> dict[str, str]:
    import pandas as pd

    nom = pd.read_parquet(data_dir / "nomenclature.parquet")
    return dict(zip(nom["heading"], nom["heading_description"], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n", type=int, default=5000, help="number of decisions to index")
    parser.add_argument("--embedder", choices=["hashing", "e5"], default="e5")
    parser.add_argument("--store", choices=["numpy", "duckdb", "pgvector"], default="numpy")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--ask", action="store_true", help="suggest a heading with the configured LLM")
    args = parser.parse_args()

    collection = load_collection(args.data, n=args.n)
    chunks = chunk_documents({ref: text for ref, (_, text) in collection.items()})
    headings = [collection[c.doc_id][0] for c in chunks]
    embedder = SentenceTransformerEmbedder() if args.embedder == "e5" else HashingEmbedder()
    store = {"numpy": lambda: NumpyVectorStore(embedder.dim),
             "duckdb": lambda: DuckDBVectorStore(embedder.dim),
             "pgvector": lambda: PgVectorStore(embedder.dim, os.environ["DATABASE_URL"])}[args.store]()
    retriever = VectorRetriever.from_chunks(chunks, embedder, headings=headings, store=store)
    print(f"{len(collection)} decisions, {len(chunks)} chunks, store={args.store}, embedder={args.embedder}\n")

    if args.ask:
        from .llm import make_client
        from .rag import suggest_heading

        client, model = make_client()
        s = suggest_heading(args.query, retriever, client, model, load_heading_texts(args.data), k=10)
        print("suggested heading:", s.heading, "| error:", s.error, "\nreason:", s.reason,
              "\ncited:", s.cited_ids, "citation precision:", s.citation_precision, "\ncandidates:", s.candidates)
        return
    hits = retriever.search(args.query, k=args.k)
    for h in hits:
        print(f"{h.score:.3f}  {h.heading}  {h.doc_id:22s}  {' '.join(h.text.split())[:80]}")
    print("\nheadings by similarity-weighted vote:", vote_headings(hits))


if __name__ == "__main__":
    main()
