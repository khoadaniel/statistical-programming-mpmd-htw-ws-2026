# Session 14 · Extras and notebook provenance

These are the downloaded source notebooks used to build the two [block notebooks](../../README.md); their content is curated into the block notebooks, and the originals stay here unchanged as reference copies.

## Downloaded notebooks

| File | Content | Block | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|---|
| [01-hf-course-tokenizers.ipynb](01-hf-course-tokenizers.ipynb) | Loading a tokeniser, tokens, ids, decoding, saving | 1 | [huggingface/notebooks, `course/en/chapter2/section4_pt.ipynb`](https://raw.githubusercontent.com/huggingface/notebooks/main/course/en/chapter2/section4_pt.ipynb) | [Apache-2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE) | 2026-10-01 | Renamed from `section4_pt.ipynb`; content unchanged. Curated into block notebook 01, section "The Hugging Face pattern" (inside the tokeniser) |
| [02-hf-course-bpe-tokenization.ipynb](02-hf-course-bpe-tokenization.ipynb) | Byte-pair encoding implemented from scratch | 1 | [huggingface/notebooks, `course/en/chapter6/section5.ipynb`](https://raw.githubusercontent.com/huggingface/notebooks/main/course/en/chapter6/section5.ipynb) | [Apache-2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE) | 2026-10-01 | Renamed from `section5.ipynb`; content unchanged. Curated into block notebook 01, section "From words to vectors" (byte-pair encoding from scratch, on EBTI descriptions) |
| [03-openai-cookbook-count-tokens.ipynb](03-openai-cookbook-count-tokens.ipynb) | tiktoken encodings compared; counting the tokens of chat requests and tool definitions | 1, 2 | [openai/openai-cookbook, `examples/How_to_count_tokens_with_tiktoken.ipynb`](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_count_tokens_with_tiktoken.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-14 | Renamed from `How_to_count_tokens_with_tiktoken.ipynb`; content unchanged. Curated into block notebook 01, section "From words to vectors" (encodings compared), and block notebook 02, section "Applications of generative models" (counting the tokens of a chat request). The last parts need an OpenAI key |
| [04-hf-course-transformer-pipelines.ipynb](04-hf-course-transformer-pipelines.ipynb) | Encoder, decoder and encoder–decoder tasks with `pipeline()` | 1 | [huggingface/notebooks, `course/en/chapter1/section3.ipynb`](https://raw.githubusercontent.com/huggingface/notebooks/main/course/en/chapter1/section3.ipynb) | [Apache-2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE) | 2026-10-01 | Renamed from `section3.ipynb`; content unchanged. Curated into block notebook 01, section "Encoder, decoder and encoder–decoder models" (one model of each family), with small models. Without `model=` it downloads large default models, some over 1 GB |
| [05-sbert-retrieve-rerank-wikipedia.ipynb](05-sbert-retrieve-rerank-wikipedia.ipynb) | Bi-encoder retrieval, cross-encoder re-ranking and BM25 over Simple Wikipedia | 1 | [huggingface/sentence-transformers, `examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb`](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) | [Apache-2.0](https://github.com/huggingface/sentence-transformers/blob/main/LICENSE) | 2026-10-01 | Renamed from `retrieve_rerank_simple_wikipedia.ipynb`; content unchanged. Curated into block notebook 01, section "Applications" (retrieve and re-rank, on English EBTI decisions). Upstream bug: the search function moves the query to the GPU when *no* GPU is available; delete those two lines before running it on a laptop |
| [06-openai-cookbook-clustering.ipynb](06-openai-cookbook-clustering.ipynb) | k-means and t-SNE on precomputed embeddings of food reviews; naming clusters with GPT-4 | 1 | [openai/openai-cookbook, `examples/Clustering.ipynb`](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Clustering.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `Clustering.ipynb`. Patched one line: the data path `./data/fine_food_reviews_with_embeddings_1k.csv` was replaced by the raw GitHub URL of the same file in the cookbook repository (35 MB, not copied here). Curated into block notebook 01, section "Applications" (the t-SNE map). The last part needs an OpenAI key |
| [07-openai-cookbook-structured-outputs.ipynb](07-openai-cookbook-structured-outputs.ipynb) | Structured outputs with JSON schema and pydantic, refusals, function calling | 2 | [openai/openai-cookbook, `examples/Structured_Outputs_Intro.ipynb`](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Structured_Outputs_Intro.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `Structured_Outputs_Intro.ipynb`; content unchanged. Curated into block notebook 02, section "Zero-shot and few-shot classification with structured output" (structured output, JSON mode, nested schemas, refusals). Needs an OpenAI key |

The Hugging Face and sentence-transformers extras (01, 02, 04, 05) start with `pip install` cells written for Google Colab; skip those cells in the course environment. All extras download their models and data from the internet.

## Authors and licences

- **Hugging Face, LLM Course notebooks:** extras 01, 02 and 04, Apache-2.0. In block notebook 01, the subsections "Byte-pair encoding from scratch, on descriptions of goods", "One model of each family at work" and "Step 1 in detail: inside the tokeniser" adapt their code and follow the course chapters (also Apache-2.0).
- **UKP Lab and Hugging Face, sentence-transformers examples:** extra 05, Apache-2.0. The subsection on retrieve and re-rank in block notebook 01 adapts its code.
- **OpenAI, OpenAI Cookbook:** extras 03, 06 and 07, MIT. Their code and explanations are adapted in block notebook 01, subsections "Encodings compared with tiktoken" and "A 2-D map of the clusters with t-SNE", and in block notebook 02, subsections "Counting the tokens of a chat request" and "JSON mode and structured outputs".

Keep the original attribution and licence notices when reusing these sources. The block notebooks are course material under CC BY 4.0; each adapted section names its source and licence. The product description in exercise 1.3 is translated by the course team.

## Figures

[`figures/make_figures.py`](../../figures/make_figures.py) and its PNGs, course team, CC BY 4.0: `tokenisation.png` (SentencePiece and BPE tokens of an English and a German phrase), `attention.png` (illustrative attention weights, not from a trained model), `embedding_space.png` (2-D UMAP projection of 1,500 EBTI decisions of five chapters, embedded with multilingual-e5-small), `coverage_accuracy.png` (coverage–accuracy curve of a TF-IDF classifier on the EBTI validation set).

## Models

Downloaded at run time into the Hugging Face cache, not stored in the repository. Check the licence on the model card before using a model in a project.

| Model | Used in | Licence |
|---|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | both blocks | MIT |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | block 1, embedding models | Apache-2.0 |
| [MoritzLaurer/mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | block 1, Hugging Face pattern (zero-shot classification) | MIT |
| [google-bert/bert-base-german-cased](https://huggingface.co/google-bert/bert-base-german-cased) | block 1, one model of each family | MIT |
| [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | block 1, Hugging Face pattern (classification head) | Apache-2.0 |
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) (tokeniser only) | block 1, byte-pair encoding | MIT |
| [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) | block 1, one model of each family | Apache-2.0 |
| [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | block 1, retrieve and re-rank | Apache-2.0 |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | block 1, one model of each family; block 2 (local stand-in) | Apache-2.0 |
| Optional via Ollama: `qwen2.5:3b` ([Qwen research licence](https://ollama.com/library/qwen2.5)) or `llama3.2:3b` ([Llama 3.2 Community License](https://ollama.com/library/llama3.2)) | block 2, if a server runs | see link |

## Data

- European Commission, *European Binding Tariff Information (EBTI)* database, full export of 15 October 2026, prepared by the lecturer and downloaded as `ebti-v1.zip` from the course drive (see [case-study/README.md](../../../../case-study/README.md)); reuse with acknowledgement under Commission Decision 2011/833/EU.
- European Commission, TARIC monthly extraction of 1 October 2026 (`Nomenclature EN.xlsx`), same reuse conditions.
- The price example in block notebook 02 (0.15 and 0.60 USD per million input and output tokens) is OpenAI's list price for `gpt-4o-mini` at its release on 18 July 2024.
