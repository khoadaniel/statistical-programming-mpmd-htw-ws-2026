"""Retrieval-augmented generation and a small tool-calling agent over product reviews.

Modules
-------
chunking    split documents into overlapping word windows
embeddings  turn text into vectors (offline hashing embedder or sentence-transformers)
store       vector stores: in-memory numpy, DuckDB, PostgreSQL with pgvector
keyword     BM25 keyword search
hybrid      reciprocal rank fusion of several rankings
metrics     recall@k, reciprocal rank, citation checks (pure functions)
prompts     prompt assembly with labelled sources
llm         OpenAI-compatible client from environment variables, and a fake client for tests
rag         retrieve -> assemble -> generate -> cite
tools       function schemas, the two agent tools and their guardrails
agent       the plan-act-observe loop
"""

__version__ = "0.1.0"
