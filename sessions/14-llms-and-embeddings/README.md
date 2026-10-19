# Session 14 · Large language models I: transformer models, embeddings and their applications

> **Guiding question.** How do language models represent and generate text, and what can we use them for?

## 1 · Embeddings: from words to vectors, and what they are good for

[Open notebook](workbooks/01-lecture-embeddings.ipynb)

- Tokens and token counts per language, byte-pair encoding, `tiktoken` encodings, embeddings and cosine similarity, contextual embeddings, attention and pre-training.
- Encoder, decoder and encoder–decoder models, with one model of each family at work.
- The Hugging Face pattern: Hub and model cards, tokeniser → model → outputs, a classification head, `pipeline()`, reproducibility.
- Sentence-embedding models with sentence-transformers: prefixes, input limits, choosing a model.
- Semantic search across languages, re-ranking and BM25.
- Clustering decisions on embeddings, with a t-SNE map.
- Embedding features versus TF-IDF, and a learning curve.
- Worked example: finding the heading among similar decisions in other languages.

## 2 · Generative models through an API

[Open notebook](workbooks/02-lecture-generative-models.ipynb)

- Next-token prediction, temperature, prompts, chat templates and the chat API.
- Context window, the cost of a run over the validation decisions, and counting the tokens of a chat request.
- Zero-shot and few-shot classification among candidate headings with structured output (`pydantic`, nested schemas, refusals).
- Comparison with a trained model on quality, cost, latency and data protection; worked example: an LLM against the trained classifier on 200 decisions.
- Abstention: routing unsure requests to a customs officer, and an LLM only for the unsure cases.
- Checkpoint 2: TF-IDF plus embeddings against checkpoint 1 with a paired bootstrap, and the post for the class board.

## Team project until the next session

Decide with evidence whether embeddings or a language model improve the model; checkpoint 2 on the class board.

## Setup

- Both notebooks download small models from the Hugging Face Hub on their first run (about 3.5 GB in total, cached in `~/.cache/huggingface/hub`); run them once before class.
- The checkpoint-2 section encodes 64,808 descriptions (a training sample of 50,000 and the 14,808 validation decisions) once, which is slow on a laptop CPU: run it before class; later runs load the vectors from `workbooks/embeddings_cache/` (ignored by git).
- For a real API model in Block 2, set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` (any OpenAI-compatible provider), or `OPENAI_API_KEY`, or run [Ollama](https://ollama.com) with `ollama pull qwen2.5:3b`. Without any of them the notebook uses the small local model `Qwen/Qwen2.5-0.5B-Instruct`. Never write a key into a notebook.
- To regenerate the figures: `uv run python sessions/14-llms-and-embeddings/figures/make_figures.py`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
