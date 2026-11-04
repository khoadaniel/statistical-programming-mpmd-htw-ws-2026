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

## 3 · Agents with LangChain

[Open notebook](workbooks/03-lecture-agents-with-langchain.ipynb)

- Setup: a Claude API key in a `.env` file, Claude Haiku 5.5, costs.
- The chat model `ChatAnthropic` with `max_tokens`; `invoke`, `stream`, `batch`; messages and `.text`.
- Few-shot prompting: zero-shot and few-shot messages, dynamic selection of examples by similarity.
- The concept of a tool, `@tool`, the tool definition as the instruction Claude reads, the system prompt, `bind_tools` and the manual loop.
- `create_agent` as a wrapper around a model; `ToolRuntime` and context; tool errors.
- Structured output, streaming, short-term memory and checkpointers.
- Graphs on a café inbox: `StateGraph`, state, nodes (a function, a model call, an agent), edges, conditional edges, `compile`; reducers on a shared shopping list; stream modes.
- Worked example: the same agent rebuilt by hand as a graph.

## 4 · Agent architectures with LangGraph

[Open notebook](workbooks/04-lecture-agent-architectures-with-langgraph.ipynb)

- The five patterns of the docs and their comparison: subagents with a supervisor, handoffs, skills, router, custom workflow.
- `Command` and `goto`, `Send` and map-reduce, subgraphs, orchestrator-workers.
- The handoff mechanism and what context passes; swarm-style peer handoffs with `create_agent` and tools.
- Human in the loop with `interrupt`, custom and built-in middleware, context engineering.
- Long-term memory and stores, time travel, durable execution, guardrails.
- Model Context Protocol, Deep Agents, Functional API, unit testing, tracing.
- Worked example: an EBTI agent with three tools and ten tasks, including a prompt injection.
- Evaluating an agent: golden dataset, outcome and trajectory graders, pass@k and pass^k, an LLM judge, TPR and TNR, the Rogan-Gladen correction.

## 5 · Practice: a weekend trip planner agent

[Open notebook](workbooks/05-practice-trip-planner-agent.ipynb)

- What an agent adds to a plain model call, then one agent built in steps: no tools, a first tool, a web service (Open-Meteo), a knowledge base, a calculator with rules, and memory.
- Part 2: workflows of agents, all built with `create_agent`: two agents in a line (prompt chaining) as a LangGraph graph, then an orchestrator agent, one city agent per city in parallel with `Send`, and a synthesiser agent, timed against the chain and the single agent.
- [Trip planner app](workbooks/trip_planner_app/README.md): the team of Step 8 as a live web page that draws the graph, the parallel city agents and the reducer while they run; it asks you when the request is unclear.

## 6 · Practice: an MCP server and an agent

[Open notebook](workbooks/06-practice-mcp-server-and-agent.ipynb)

- Build an MCP server with FastMCP, run it over stdio and HTTP, connect a LangChain agent, and spot tool poisoning.

## 7 · Practice: evaluating an agent

[Open notebook](workbooks/07-practice-evaluating-an-agent.ipynb)

- Code graders, pass@k and pass^k, and an LLM judge checked against hand labels.

## Team project until the next session

Optional language-model component (for example a classification assistant with retrieval), with its evaluation.

## Setup

- Models are downloaded from the Hugging Face Hub on first use: `intfloat/multilingual-e5-small`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. Embeddings are cached in the system's temporary folder.
- Language model for notebooks 01 and 02 (optional): any OpenAI-compatible endpoint with tool calling, set by `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`, or `OPENAI_API_KEY`, or a local [Ollama](https://ollama.com/) server (`ollama pull llama3.2`). Without one, the model cells print a message and show a saved example, and the agent runs with a scripted model. Never write a key into a notebook.
- Notebooks 03 and 04 call Claude Haiku 5.5 (`claude-haiku-5-5`) through the Claude API, so you need your own API key:
  1. Create a key on the [API keys](https://platform.claude.com/settings/keys) page of the Claude Console.
  2. In `workbooks/`, copy `.env.example` to `.env` and paste the key after `ANTHROPIC_API_KEY=`. Git ignores `.env`; never type the key into a notebook.
  3. In the Console, set a monthly spending limit under **Settings > Billing**.
- Cost of notebooks 03 and 04: Haiku 5.5 costs 0.10 USD per million input tokens and 0.50 USD per million output tokens ([pricing](https://platform.claude.com/docs/en/about-claude/pricing), 3 November 2026). A full run of notebook 03 costs a few cents, and notebook 04 well under one euro. Cells that call Claude are saved without output and show a result only when you run them. The setup section of notebook 03 explains the details.
- [`workspace/`](workspace/README.md) holds the same assistant as tested software (package `bti_assistant`): `cd workspace && uv run pytest -q`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
