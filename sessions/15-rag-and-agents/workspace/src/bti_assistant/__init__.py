"""A classification assistant for EU customs decisions: retrieval-augmented generation and a small agent.

Modules
-------
chunking    split long descriptions into overlapping word windows
embeddings  turn text into vectors (offline hashing embedder or a multilingual sentence-transformers model)
store       vector stores: in-memory numpy, DuckDB, PostgreSQL with pgvector
keyword     BM25 keyword search
hybrid      reciprocal rank fusion of several rankings
metrics     heading hit@k, recall@k, reciprocal rank, citation checks (pure functions)
prompts     prompt assembly with labelled sources (past decisions and nomenclature texts)
llm         OpenAI-compatible client from environment variables, and a fake client for tests
rag         retrieve -> assemble -> generate -> cite; heading suggestion with cited BTI references
tools       function schemas, the three agent tools and their guardrails
agent       the plan-act-observe loop
"""

__version__ = "0.2.0"
