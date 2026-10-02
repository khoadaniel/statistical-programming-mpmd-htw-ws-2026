# Session 14 · Large language models I: transformer models, embeddings and their applications

> [!NOTE]
> **Guiding question:** How do language models represent and generate text, and what can we use them for?

**Learning outcomes.** Students are able to

- explain the transformer idea and the difference between encoder, decoder and encoder–decoder models
- use embedding models for semantic search, clustering and as features for classification
- use a generative model through an API for classification with structured output and compare it with a trained model

Neural-network theory stays conceptual here; module 3.4 covers it in depth.

## Session plan

**0:00–0:45 · From words to vectors** ([theory/01-from-words-to-vectors.md](theory/01-from-words-to-vectors.md))

- From words to vectors: [tokens](theory/01-from-words-to-vectors.md#tokens-how-a-language-model-splits-text), [embeddings](theory/01-from-words-to-vectors.md#embeddings-words-and-texts-as-vectors), [attention and pre-training](theory/01-from-words-to-vectors.md#attention-and-pre-training-conceptual) at a conceptual level
- [Encoder models (BERT type), decoder models (GPT type) and encoder–decoder models](theory/01-from-words-to-vectors.md#encoder-decoder-and-encoderdecoder-models)
- [Embedding models (sentence-transformers)](theory/01-from-words-to-vectors.md#embedding-models-sentence-transformers)
- *Practice:* tokenise descriptions of goods and compare token counts across languages; compute semantic similarity between descriptions of the same goods in different languages → [workbooks/01-case-study-tokens-and-similarity.ipynb](workbooks/01-case-study-tokens-and-similarity.ipynb)

**1:00–1:45 · Applications of embedding models** ([theory/02-applications-of-embeddings.md](theory/02-applications-of-embeddings.md))

- [Semantic search](theory/02-applications-of-embeddings.md#semantic-search)
- [Clustering (with the methods of Session 11)](theory/02-applications-of-embeddings.md#clustering-decisions)
- [Classification features compared with TF-IDF](theory/02-applications-of-embeddings.md#embedding-features-versus-tf-idf)
- *Practice:* embedding features versus TF-IDF on a subsample; nearest-neighbour search over decisions in several languages → [workbooks/05-case-study-embeddings-vs-tfidf.ipynb](workbooks/05-case-study-embeddings-vs-tfidf.ipynb)

**2:00–2:45 · Generative models through an API** ([theory/03-generative-models-through-an-api.md](theory/03-generative-models-through-an-api.md))

- [Prompts, temperature, context window and costs](theory/03-generative-models-through-an-api.md#how-a-generative-model-produces-text-temperature-context-window-and-costs) · [the chat API](theory/03-generative-models-through-an-api.md#prompts-and-the-chat-api)
- [Zero-shot and few-shot classification with structured output](theory/03-generative-models-through-an-api.md#zero-shot-and-few-shot-classification-with-structured-output)
- [Comparison with a trained model on quality, cost, latency and data protection](theory/03-generative-models-through-an-api.md#comparison-with-a-trained-model-quality-cost-latency-and-data-protection)
- *Practice:* case study: compare an LLM (choosing among ten candidate headings) with the trained classifier on 200 decisions → [workbooks/09-case-study-llm-vs-trained-classifier.ipynb](workbooks/09-case-study-llm-vs-trained-classifier.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-from-words-to-vectors.md](theory/01-from-words-to-vectors.md) | Subword tokens in many languages, embeddings, attention, pre-training, model families, multilingual sentence-transformers | 1 | core |
| [theory/02-applications-of-embeddings.md](theory/02-applications-of-embeddings.md) | Cross-lingual semantic search, clustering, embedding features vs TF-IDF | 2 | core |
| [theory/03-generative-models-through-an-api.md](theory/03-generative-models-through-an-api.md) | Temperature, context, costs, chat API, candidate headings and structured output, comparison with a trained model | 3 | core |
| [workbooks/01-case-study-tokens-and-similarity.ipynb](workbooks/01-case-study-tokens-and-similarity.ipynb) | Case study: token counts per language with SentencePiece and BPE; similarity of descriptions across languages | 1 | core |
| [workbooks/02-hf-course-tokenizers.ipynb](workbooks/02-hf-course-tokenizers.ipynb) | Hugging Face course: loading tokenisers, encoding and decoding | 1 | optional |
| [workbooks/03-hf-course-bpe-tokenization.ipynb](workbooks/03-hf-course-bpe-tokenization.ipynb) | Hugging Face course: byte-pair encoding implemented step by step | 1 | optional |
| [workbooks/04-hf-course-transformer-pipelines.ipynb](workbooks/04-hf-course-transformer-pipelines.ipynb) | Hugging Face course: encoder, decoder and encoder–decoder models via `pipeline()` | 1 | optional (Colab; some models > 1 GB) |
| [workbooks/05-case-study-embeddings-vs-tfidf.ipynb](workbooks/05-case-study-embeddings-vs-tfidf.ipynb) | Case study: embeddings vs TF-IDF vs both, learning curve, cross-lingual nearest-neighbour search | 2 | core |
| [workbooks/06-sbert-retrieve-rerank-wikipedia.ipynb](workbooks/06-sbert-retrieve-rerank-wikipedia.ipynb) | sentence-transformers: semantic search with bi-encoder and cross-encoder re-ranking (Simple Wikipedia) | 2 | optional |
| [workbooks/07-openai-cookbook-clustering.ipynb](workbooks/07-openai-cookbook-clustering.ipynb) | OpenAI Cookbook: k-means on precomputed embeddings of food reviews (third-party data), t-SNE plot | 2 | optional (last part needs an OpenAI key) |
| [workbooks/08-openai-cookbook-structured-outputs.ipynb](workbooks/08-openai-cookbook-structured-outputs.ipynb) | OpenAI Cookbook: structured outputs with JSON schema and pydantic | 3 | optional (needs an OpenAI key, or adapt `base_url`) |
| [workbooks/09-case-study-llm-vs-trained-classifier.ipynb](workbooks/09-case-study-llm-vs-trained-classifier.ipynb) | Case study: zero/few-shot LLM with ten candidate headings vs TF-IDF on 200 decisions; quality, cost, latency table | 3 | core |

Sources and licences of third-party files: [source.md](source.md).

## Before and after the session

**Preparation.**

- Install the embedding dependencies once (`uv sync --group embeddings` at the repository root, about 1 GB with PyTorch) and run the second cell of workbook 01 so that the multilingual model `intfloat/multilingual-e5-small` (about 470 MB) is downloaded before class.
- For Block 3, install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:3b` (about 2 GB; `llama3.2:3b` also works). Without it, the workbook runs but skips the LLM calls.
- Read Alammar's *The Illustrated Transformer* (link below), sections on self-attention.
- Revise TF-IDF and the error analysis of Session 13, k-means and UMAP of Session 11, and the bootstrap of Session 7.

**Team project until the next session.** Decide with evidence whether a language-model component improves the project.

**Further reading (optional).**

- Alammar, J. *The Illustrated Transformer*: https://jalammar.github.io/illustrated-transformer/
- Hugging Face *LLM Course*, chapters 1, 2 and 6: https://huggingface.co/learn/llm-course
- Sentence-Transformers documentation (semantic search, clustering, multilingual models): https://sbert.net/
- 3Blue1Brown, *Transformers, the tech behind LLMs* (video lessons on embeddings and attention): https://www.3blue1brown.com/lessons/gpt
- Jurafsky and Martin, *Speech and Language Processing*, 3rd ed. draft, chapters on transformers and large language models: https://web.stanford.edu/~jurafsky/slp3/

## Setup

Packages beyond the course environment:

- `sentence-transformers` (pulls in PyTorch and `transformers`): Blocks 1 and 2; in the root project as the dependency group `embeddings`
- `tiktoken`: token counts (theory pages 1 and 3, workbook 01)
- `openai`, `pydantic`, `httpx`: already in the course environment (Block 3)
- a local model server for Block 3: [Ollama](https://ollama.com) with `qwen2.5:3b` or `llama3.2:3b` (optional; any OpenAI-compatible provider works via `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`)

```bash
# from the repository root
uv sync --group embeddings
uv run --with tiktoken jupyter lab
# or without changing the project environment:
uv run --with sentence-transformers --with tiktoken --with umap-learn jupyter lab

# Block 3 with a hosted provider instead of Ollama (example; the key stays in the shell)
export LLM_BASE_URL="https://api.openai.com/v1" LLM_MODEL="gpt-4o-mini" LLM_API_KEY="..."
```

The Hugging Face and sentence-transformers workbooks (02, 03, 04, 06) start with `pip install` cells written for Google Colab; in the course environment skip those cells. Workbook 04 downloads several models, some larger than 1 GB: run it on Colab or pick the small models named in its cells. The figures of the theory pages are made by [theory/figures/make_figures.py](theory/figures/make_figures.py).
