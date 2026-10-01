# Workspace: review assistant (RAG and a small agent)

A small Python package that does what the three notebooks of Session 15 do, as tested software:
semantic search over reviews, a RAG answer with citations, retrieval metrics, and an agent with two tools.
Copy the folder into your team repository and extend it.

```
src/review_assistant/
  chunking.py    word windows with overlap; sentence chunks (exercise 2)
  embeddings.py  HashingEmbedder (offline, for tests) and SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  store.py       NumpyVectorStore, DuckDBVectorStore, PgVectorStore (the course path; exercise 1)
  keyword.py     BM25
  hybrid.py      reciprocal rank fusion
  retrieval.py   VectorRetriever, HybridRetriever
  metrics.py     recall@k, precision@k, hit@k, reciprocal rank, citation precision, Cohen's kappa
  prompts.py     prompt assembly with labelled sources
  llm.py         OpenAI-compatible client from environment variables; FakeChatClient for tests
  rag.py         retrieve -> assemble -> generate -> cite
  tools.py       function schemas, sql_query and search_reviews, guardrails
  agent.py       the plan-act-observe loop (token budget: exercise 3), task scoring
  demo.py        command-line demo on the case-study data
tests/           pytest; offline, with a fake LLM, numpy/DuckDB stores and an sqlite database
solutions/       reference solutions of the exercises, for self-checking only
```

## Setup

```bash
cd sessions/15-rag-and-agents/workspace
uv sync                                    # core: numpy, openai, duckdb
uv run pytest -q                           # 41 passed, 2 xfailed (the open exercises)
uvx ruff check .
uv sync --extra data --extra embeddings    # pandas and sentence-transformers for the demo
uv run python -m review_assistant.demo "stopped connecting to wifi"
```

### LLM access

All LLM calls go through an OpenAI-compatible endpoint, configured with environment variables. The default
is a local [Ollama](https://ollama.com/) server; nothing leaves your machine.

```bash
ollama pull llama3.2                                     # about 2 GB; supports tool calling
export LLM_BASE_URL=http://localhost:11434/v1            # default
export LLM_MODEL=llama3.2                                # default
# another provider: export LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=gpt-4.1-mini LLM_API_KEY=...
uv run python -m review_assistant.demo "Does the fish oil cause fishy burps?" --ask
```

Never write an API key into code, notebooks or a commit. Keep it in the environment or in a `.env` file that
is listed in `.gitignore`.

### PostgreSQL with pgvector (the course path)

```bash
docker run --name pgvector -e POSTGRES_PASSWORD=course -p 5432:5432 -d pgvector/pgvector:pg17
export DATABASE_URL=postgresql://postgres:course@localhost:5432/postgres
uv sync --extra data --extra embeddings --extra postgres
uv run python -m review_assistant.demo "fishy burps" --store pgvector
docker stop pgvector                                      # docker start pgvector to continue later
```

The image `pgvector/pgvector:pg17` is PostgreSQL 17 with the extension installed; `PgVectorStore` runs
`CREATE EXTENSION IF NOT EXISTS vector` itself. The tests do not need the database: they use the numpy and
DuckDB stores, which return the same ranking.

## Exercises

The exercises follow the practice tasks of the session. Tests for exercises 2 and 3 are in
`tests/test_exercises.py`; they are marked `xfail` and turn into `XPASS` when your solution works.

1. **Semantic search in PostgreSQL (block 1).** Start the pgvector container and run the demo with
   `--store pgvector`. Then implement the metadata filter in `PgVectorStore.search` (TODO in `store.py`):
   `WHERE parent_asin = %s`. Check that `--store pgvector` and `--store numpy` return the same reviews.
   Add `store.create_index()` and compare the query plan with `EXPLAIN`.
2. **Chunk size (block 2).** Implement `chunk_sentences` (TODO in `chunking.py`). Write a small script that
   evaluates recall@5 on the ten labelled questions of notebook `04-case-study-rag-evaluation.ipynb` with
   `metrics.evaluate_retrieval`, for word windows of 60, 120 and 250 words and for sentence chunks.
3. **Cost guardrail (block 3).** Implement the token budget in `run_agent` (TODO in `agent.py`): stop with
   `stop_reason="budget"` once the total tokens exceed `max_total_tokens`.
4. **A third tool (block 3).** Add a tool `product_info(parent_asin)` that returns title, store and price of
   one product. Write its schema, add it to the `ToolBox`, and write tests with `FakeChatClient` for a
   correct call and for an unknown product id.
5. **Evaluate the agent (block 3).** Write ten `AgentTask`s (as in notebook `10-case-study-agent.ipynb`), run
   them with a real model and `score_task`, and report the share of correct answers, tool errors and tokens
   per task. Add one task whose data contain a prompt injection.

## Design notes

- **Pure functions for metrics.** Every metric in `metrics.py` takes lists and sets and returns a number,
  so it can be checked with a worked example by hand (see `tests/test_metrics.py`).
- **The model proposes, the code decides.** `ToolBox.call` validates the arguments against the schema, runs
  only allow-listed tools and returns errors as observations instead of raising them.
- **Read-only SQL is checked twice in practice.** `check_read_only_sql` is a coarse first line of defence. In
  a real deployment, the agent's database user must only have `SELECT` rights.
- **Any OpenAI-compatible endpoint.** The code uses only `client.chat.completions.create` with `messages`,
  `tools` and `temperature`, which Ollama, vLLM, LM Studio, OpenAI and many other providers support.
