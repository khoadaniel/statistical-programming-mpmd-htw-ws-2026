# Workspace: a classification assistant for customs decisions (RAG and a small agent)

A small Python package that does what the three notebooks of Session 15 do, as tested software:
semantic search over past Binding Tariff Information (BTI) decisions in any language, a heading suggestion
that cites the decisions it is based on, retrieval metrics, and an agent with three tools.
Copy the folder into your team repository and extend it.

```
src/bti_assistant/
  chunking.py    word windows with overlap; sentence chunks (exercise 2)
  embeddings.py  HashingEmbedder (offline, for tests) and SentenceTransformerEmbedder (multilingual-e5-small)
  store.py       NumpyVectorStore, DuckDBVectorStore, PgVectorStore (the course path; exercise 1)
  keyword.py     BM25
  hybrid.py      reciprocal rank fusion
  retrieval.py   VectorRetriever, HybridRetriever (filter by HS chapter)
  metrics.py     heading hit@k, recall@k, precision@k, reciprocal rank, citation precision, Cohen's kappa
  prompts.py     prompt assembly with labelled sources (BTI references, headings, nomenclature texts)
  llm.py         OpenAI-compatible client from environment variables; FakeChatClient for tests
  rag.py         answer_question, vote_headings (retrieval-only baseline), suggest_heading (checked JSON answer)
  tools.py       function schemas; sql_query, search_decisions, lookup_heading; guardrails
  agent.py       the plan-act-observe loop (token budget: exercise 3), task scoring
  demo.py        command-line demo on the case-study data
tests/           pytest; offline, with a fake LLM, numpy/DuckDB stores and an sqlite database
solutions/       reference solutions of the exercises, for self-checking only
```

## Setup

```bash
cd sessions/15-rag-and-agents/workspace
uv sync                                    # core: numpy, openai, duckdb
uv run pytest -q                           # 48 passed, 2 xfailed (the open exercises)
uvx ruff check .
uv sync --extra data --extra embeddings    # pandas and sentence-transformers for the demo (model: about 470 MB)
uv run python -m bti_assistant.demo "Damenstiefel mit Oberteil aus Rindleder und Laufsohle aus Gummi"
```

The demo indexes 5,000 decisions of the training sample and prints the most similar ones with their
headings and a similarity-weighted vote. For the query above, four of the five neighbours are boots
(headings 6403, 6401, 6404) and the vote puts 6403 first; an English query such as "plastic toy car for
children" finds French and English toy decisions (9503).

### LLM access

All LLM calls go through an OpenAI-compatible endpoint, configured with environment variables. The default
is a local [Ollama](https://ollama.com/) server; nothing leaves your machine.

```bash
ollama pull llama3.2                                     # about 2 GB; supports tool calling
export LLM_BASE_URL=http://localhost:11434/v1            # default
export LLM_MODEL=llama3.2                                # default
# another provider: export LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=gpt-4.1-mini LLM_API_KEY=...
uv run python -m bti_assistant.demo "Damenstiefel mit Oberteil aus Rindleder" --ask
```

Never write an API key into code, notebooks or a commit. Keep it in the environment or in a `.env` file that
is listed in `.gitignore`. New BTI requests contain confidential business information: before you send real
requests to a hosted provider, clarify the legal basis.

### PostgreSQL with pgvector (the course path)

```bash
docker run --name pgvector -e POSTGRES_PASSWORD=course -p 5432:5432 -d pgvector/pgvector:pg17
export DATABASE_URL=postgresql://postgres:course@localhost:5432/postgres
uv sync --extra data --extra embeddings --extra postgres
uv run python -m bti_assistant.demo "plastic toy car" --store pgvector
docker stop pgvector                                      # docker start pgvector to continue later
```

The image `pgvector/pgvector:pg17` is PostgreSQL 17 with the extension installed; `PgVectorStore` runs
`CREATE EXTENSION IF NOT EXISTS vector` itself. The tests do not need the database: they use the numpy and
DuckDB stores, which return the same ranking.

## Exercises

The exercises follow the practice tasks of the session. Tests for exercises 2 and 3 are in
`tests/test_exercises.py`; they are marked `xfail` and turn into `XPASS` when your solution works.

1. **Semantic search in PostgreSQL (block 1).** Start the pgvector container and run the demo with
   `--store pgvector`. Then implement the chapter filter in `PgVectorStore.search` (TODO in `store.py`):
   `WHERE left(heading, 2) = %s`. Check that `--store pgvector` and `--store numpy` return the same decisions.
   Add `store.create_index()` and compare the query plan with `EXPLAIN`.
2. **Chunk size (block 2).** Implement `chunk_sentences` (TODO in `chunking.py`). Write a small script that
   evaluates heading hit@5 (`metrics.evaluate_heading_retrieval`) on 500 decisions of 2023 used as queries,
   for word windows of 40, 120 and 250 words and for sentence chunks.
3. **Cost guardrail (block 3).** Implement the token budget in `run_agent` (TODO in `agent.py`): stop with
   `stop_reason="budget"` once the total tokens exceed `max_total_tokens`.
4. **A fourth tool (block 3).** Add a tool `heading_history(heading)` that returns the number of decisions
   per year and issuing country for one heading (hint: a fixed, parameterised SQL query, not free SQL).
   Write its schema, add it to the `ToolBox`, and write tests with `FakeChatClient` for a correct call and
   for a heading that is not four digits.
5. **Evaluate the agent (block 3).** Write ten `AgentTask`s (as in notebook `10-case-study-agent.ipynb`), run
   them with a real model and `score_task`, and report the share of correct answers, tool errors and tokens
   per task. Add one task whose data contain a prompt injection.

## Design notes

- **Pure functions for metrics.** Every metric in `metrics.py` takes lists and sets and returns a number,
  so it can be checked with a worked example by hand (see `tests/test_metrics.py`).
- **The model proposes, the code decides.** `suggest_heading` accepts only a heading that one of the
  retrieved decisions carries and reports invented references through `citation_precision`;
  `ToolBox.call` validates arguments against the schema, runs only allow-listed tools and returns errors as
  observations instead of raising them.
- **Read-only SQL is checked twice in practice.** `check_read_only_sql` is a coarse first line of defence. In
  a real deployment, the agent's database user must only have `SELECT` rights.
- **Any OpenAI-compatible endpoint.** The code uses only `client.chat.completions.create` with `messages`,
  `tools`, `temperature` and `response_format={"type": "json_object"}`, which Ollama, vLLM, LM Studio,
  OpenAI and many other providers support.
