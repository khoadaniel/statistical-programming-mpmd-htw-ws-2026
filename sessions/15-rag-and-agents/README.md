# Session 15 · Large language models II: retrieval-augmented generation and agents

> [!NOTE]
> **Guiding question.** How do we let a language model answer from our own data and use tools, and how do we evaluate the result?

**Learning outcomes.** Students are able to

- build a retrieval-augmented generation pipeline over a document collection
- evaluate retrieval and answers with a labelled test set
- build a simple agent with tool calling and assess its risks

## Session plan

**0:00–0:45 · Retrieval-augmented generation** ([theory](theory/01-retrieval-augmented-generation.md))

- [Why retrieval: knowledge cut-off, hallucination, sources](theory/01-retrieval-augmented-generation.md#why-retrieval-knowledge-cut-off-hallucination-sources)
- The components of RAG: [chunking](theory/01-retrieval-augmented-generation.md#chunking), [embedding](theory/01-retrieval-augmented-generation.md#embedding-the-chunks), [vector search](theory/01-retrieval-augmented-generation.md#vector-search-with-cosine-similarity) [in PostgreSQL with pgvector](theory/01-retrieval-augmented-generation.md#vector-search-in-postgresql-with-pgvector), [prompt assembly](theory/01-retrieval-augmented-generation.md#prompt-assembly), [answers with citations](theory/01-retrieval-augmented-generation.md#answers-with-citations)

*Practice:* Build semantic search over past customs decisions (22 languages) in PostgreSQL → [`01-case-study-semantic-search-postgres.ipynb`](workbooks/01-case-study-semantic-search-postgres.ipynb)

**1:00–1:45 · Evaluating and improving RAG** ([theory](theory/02-evaluating-and-improving-rag.md))

- Evaluating RAG: [a test set of questions](theory/02-evaluating-and-improving-rag.md#a-test-set-of-questions), [recall@k](theory/02-evaluating-and-improving-rag.md#recallk-and-related-retrieval-metrics), [groundedness](theory/02-evaluating-and-improving-rag.md#groundedness) and [human ratings](theory/02-evaluating-and-improving-rag.md#human-ratings)
- [Improving retrieval (chunk size, hybrid search, re-ranking)](theory/02-evaluating-and-improving-rag.md#improving-retrieval-chunk-size-hybrid-search-re-ranking)

*Practice:* Case study: a RAG classification assistant that proposes a heading from similar past decisions with cited BTI references, evaluated with heading hit@5 on ten labelled requests (and 500 more) → [`04-case-study-rag-evaluation.ipynb`](workbooks/04-case-study-rag-evaluation.ipynb)

**2:00–2:45 · Agents and guardrails** ([theory](theory/03-agents-and-guardrails.md))

- Agents: [tool calling and function schemas](theory/03-agents-and-guardrails.md#tool-calling-and-function-schemas), [the loop of planning, acting and observing](theory/03-agents-and-guardrails.md#the-loop-of-planning-acting-and-observing)
- [Risks (error propagation, costs, prompt injection)](theory/03-agents-and-guardrails.md#risks-error-propagation-cost-and-prompt-injection) and [guardrails](theory/03-agents-and-guardrails.md#guardrails)

*Practice:* Build a small agent with three tools (an SQL query over the decisions, the similar-decision search, a nomenclature look-up) and evaluate it on ten tasks, including a prompt injection inside a description → [`10-case-study-agent.ipynb`](workbooks/10-case-study-agent.ipynb), then as tested software in [`workspace/`](workspace/README.md)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-retrieval-augmented-generation.md](theory/01-retrieval-augmented-generation.md) | Why retrieval; chunking, embedding, vector search, pgvector, prompts, citations | 1 | core |
| [theory/02-evaluating-and-improving-rag.md](theory/02-evaluating-and-improving-rag.md) | Test set, recall@k, groundedness, human ratings; chunk size, hybrid search, re-ranking | 2 | core |
| [theory/03-agents-and-guardrails.md](theory/03-agents-and-guardrails.md) | Tool calling, function schemas, agent loop, risks, guardrails | 3 | core |
| [workbooks/01-case-study-semantic-search-postgres.ipynb](workbooks/01-case-study-semantic-search-postgres.ipynb) | Own: semantic search over decisions in PostgreSQL/pgvector (DuckDB fallback), multilingual queries, chapter filter | 1 | core |
| [workbooks/02-question-answering-using-embeddings.ipynb](workbooks/02-question-answering-using-embeddings.ipynb) | OpenAI Cookbook: search-then-ask question answering (needs an OpenAI key) | 1 | optional |
| [workbooks/03-advanced-rag.ipynb](workbooks/03-advanced-rag.ipynb) | Hugging Face Cookbook: chunking, vector index, re-ranking with LangChain | 1–2 | optional |
| [workbooks/pgvector-examples/](workbooks/pgvector-examples/) | pgvector-python scripts: sentence-transformers, hybrid search (RRF, cross-encoder), RAG with Ollama | 1–2 | optional |
| [workbooks/04-case-study-rag-evaluation.ipynb](workbooks/04-case-study-rag-evaluation.ipynb) | Own: ten labelled requests and 500 more, heading hit@5, variants, similarity-vote baseline, RAG prototype, ratings | 2 | core |
| [workbooks/05-rag-evaluation.ipynb](workbooks/05-rag-evaluation.ipynb) | Hugging Face Cookbook: synthetic test set and LLM-as-judge evaluation | 2 | optional |
| [workbooks/06-search-reranking-with-cross-encoders.ipynb](workbooks/06-search-reranking-with-cross-encoders.ipynb) | OpenAI Cookbook: re-ranking search results (needs an OpenAI key) | 2 | optional |
| [workbooks/07-function-calling-with-chat-models.ipynb](workbooks/07-function-calling-with-chat-models.ipynb) | OpenAI Cookbook: function calling, including an SQL tool | 3 | optional |
| [workbooks/08-agent-text-to-sql.ipynb](workbooks/08-agent-text-to-sql.ipynb) | Hugging Face Cookbook: text-to-SQL agent with error correction (smolagents) | 3 | optional |
| [workbooks/09-how-to-use-guardrails.ipynb](workbooks/09-how-to-use-guardrails.ipynb) | OpenAI Cookbook: input and output guardrails | 3 | optional |
| [workbooks/10-case-study-agent.ipynb](workbooks/10-case-study-agent.ipynb) | Own: agent with SQL, decision search and nomenclature look-up, ten tasks, prompt-injection demo | 3 | core |
| [workspace/](workspace/README.md) | Package `bti_assistant`: retrieval, heading metrics, heading suggestion, three tools, agent loop; offline tests; exercises | 1–3 | core |

Sources and licences: [source.md](source.md).

## Before and after the session

**Preparation.** Re-read the embeddings part of Session 14. Start the database once so that the image is downloaded before class:
`docker run --name pgvector -e POSTGRES_PASSWORD=course -p 5432:5432 -d pgvector/pgvector:pg17`.
Download the multilingual embedding model once (run the first cells of workbook 01; `intfloat/multilingual-e5-small`, about 470 MB). Optionally install [Ollama](https://ollama.com/) and run `ollama pull llama3.2` (about 2 GB) for a local LLM; without it, the notebooks skip the LLM cells or use a scripted stand-in.

**Team project until the next session.** Optional language-model component (for example a classification assistant with retrieval), with its evaluation.

**Further reading (optional, free)**

- Anthropic (2024): [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- Hamel Husain (2024): [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- DataTalksClub: [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) (RAG, vector search and evaluation, with exercises)
- OWASP: [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)
- Hugging Face: [Agents Course](https://huggingface.co/learn/agents-course)

## Setup

Packages beyond the course environment: `sentence-transformers`, `openai`, `duckdb`, and for PostgreSQL `psycopg[binary]` and `pgvector`. From the repository root:

```bash
uv run --with jupyterlab --with pandas --with pyarrow --with scikit-learn --with sentence-transformers \
       --with openai --with duckdb --with "psycopg[binary]" --with pgvector jupyter lab
```

LLM access (all notebooks and the workspace): `LLM_BASE_URL` (default `http://localhost:11434/v1`, Ollama), `LLM_MODEL` (default `llama3.2`), `LLM_API_KEY` (only for commercial providers; never commit it). The embedding model `intfloat/multilingual-e5-small` (about 470 MB) and the English cross-encoder `ms-marco-MiniLM-L-6-v2` (about 90 MB, theory page 2) are downloaded on first use. The case-study workbooks import the workspace package from `workspace/src`.

Workspace: `cd workspace && uv run pytest -q` (see [workspace/README.md](workspace/README.md)). The third-party notebooks list their own dependencies in their first cells; several need an OpenAI key or a Hugging Face token.
