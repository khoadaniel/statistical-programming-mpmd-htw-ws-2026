# Session 15 · Retrieval-augmented generation and agents

> **Guiding question.** How do we let a language model answer from our own data and use tools, and how do we evaluate the result?

## 1 · Retrieval-augmented generation and its evaluation

[Open notebook](workbooks/01-lecture-retrieval-augmented-generation-and-evaluation.ipynb)

- Why retrieval: knowledge cut-off, private data, hallucination, sources; search versus fine-tuning.
- Chunking: word windows, tokens versus words, recursive chunking, duplicates.
- Embedding and semantic search; cosine, dot product and Euclidean distance; languages.
- Vector search over stored embeddings: SQLite BLOBs, SQL filters, NumPy ranking, exact and approximate search.
- Prompt assembly with labelled sources and a token budget; answers with citations and their automatic checks.
- Evaluating RAG: test sets, precision, recall@k, hit@k, MRR, standard errors, synthetic questions.
- Groundedness, human ratings, an NLI judge and Cohen's kappa; end-to-end accuracy with an LLM judge.
- Improving retrieval, measured on 500 requests: chunk size, metadata filters, BM25 and hybrid search, weighted fusion, contextual enrichment, re-ranking (BM25 and the cross-encoder were introduced in Session 14).
- Worked example: a classification assistant that proposes a heading with cited BTI references, evaluated with heading hit@5.

## 2 · Agents and guardrails

[Open notebook](workbooks/02-lecture-agents-and-guardrails.ipynb)

- Tool calling and function schemas (JSON Schema, `tool_choice`, parallel calls, OpenAI and Anthropic formats).
- The database schema in the tool description, schemas from type hints, argument validation.
- The loop of planning, acting and observing; the message protocol; a scripted model.
- Self-correction, silent failures, and when a workflow is enough.
- Risks: error propagation, cost per step, prompt injection inside a description.
- Guardrails: allow-list, schema validation, read-only SQLite, step and token limits, human confirmation, input and output checks.
- Worked example: an agent with three tools evaluated on ten tasks, including a prompt injection.

## Team project until the next session

Optional language-model component (for example a classification assistant with retrieval), with its evaluation.

## Setup

- Models are downloaded from the Hugging Face Hub on first use: `intfloat/multilingual-e5-small`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. Embeddings are cached in the system's temporary folder.
- Language model (optional): any OpenAI-compatible endpoint with tool calling, set by `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`, or `OPENAI_API_KEY`, or a local [Ollama](https://ollama.com/) server (`ollama pull llama3.2`). Without one, the model cells print a message and show a saved example, and the agent runs with a scripted model. Never write a key into a notebook.
- [`workspace/`](workspace/README.md) holds the same assistant as tested software (package `bti_assistant`): `cd workspace && uv run pytest -q`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
