# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks
| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/02-question-answering-using-embeddings.ipynb](workbooks/02-question-answering-using-embeddings.ipynb) | Search-then-ask question answering with embeddings; costs | [openai/openai-cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Question_answering_using_embeddings.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `Question_answering_using_embeddings.ipynb`; content unchanged |
| [workbooks/03-advanced-rag.ipynb](workbooks/03-advanced-rag.ipynb) | Chunking, vector index, retrieval, re-ranking, generation (LangChain) | [huggingface/cookbook](https://raw.githubusercontent.com/huggingface/cookbook/main/notebooks/en/advanced_rag.ipynb) | [Apache-2.0](https://github.com/huggingface/cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `advanced_rag.ipynb`; content unchanged |
| [workbooks/05-rag-evaluation.ipynb](workbooks/05-rag-evaluation.ipynb) | Synthetic evaluation set, critique agents, LLM-as-judge | [huggingface/cookbook](https://raw.githubusercontent.com/huggingface/cookbook/main/notebooks/en/rag_evaluation.ipynb) | [Apache-2.0](https://github.com/huggingface/cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `rag_evaluation.ipynb`; content unchanged |
| [workbooks/06-search-reranking-with-cross-encoders.ipynb](workbooks/06-search-reranking-with-cross-encoders.ipynb) | Re-ranking search results | [openai/openai-cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Search_reranking_with_cross-encoders.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `Search_reranking_with_cross-encoders.ipynb`; content unchanged |
| [workbooks/07-function-calling-with-chat-models.ipynb](workbooks/07-function-calling-with-chat-models.ipynb) | Function schemas, tool calls, an SQL tool | [openai/openai-cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_call_functions_with_chat_models.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `How_to_call_functions_with_chat_models.ipynb`; content unchanged |
| [workbooks/08-agent-text-to-sql.ipynb](workbooks/08-agent-text-to-sql.ipynb) | Text-to-SQL agent with automatic error correction (smolagents) | [huggingface/cookbook](https://raw.githubusercontent.com/huggingface/cookbook/main/notebooks/en/agent_text_to_sql.ipynb) | [Apache-2.0](https://github.com/huggingface/cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `agent_text_to_sql.ipynb`; content unchanged |
| [workbooks/09-how-to-use-guardrails.ipynb](workbooks/09-how-to-use-guardrails.ipynb) | Input and output guardrails | [openai/openai-cookbook](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_use_guardrails.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `How_to_use_guardrails.ipynb`; content unchanged |
| [workbooks/pgvector-examples/sentence_transformers_example.py](workbooks/pgvector-examples/sentence_transformers_example.py) | Embeddings into pgvector, nearest-neighbour query | [pgvector/pgvector-python](https://raw.githubusercontent.com/pgvector/pgvector-python/master/examples/sentence_transformers/example.py) | [MIT](workbooks/pgvector-examples/LICENSE.txt) | 2026-10-01 | Renamed from `examples/sentence_transformers/example.py`; content unchanged |
| [workbooks/pgvector-examples/hybrid_search_rrf.py](workbooks/pgvector-examples/hybrid_search_rrf.py) | Hybrid search: full-text + vectors with reciprocal rank fusion in SQL | [pgvector/pgvector-python](https://raw.githubusercontent.com/pgvector/pgvector-python/master/examples/hybrid_search/rrf.py) | [MIT](workbooks/pgvector-examples/LICENSE.txt) | 2026-10-01 | Renamed from `examples/hybrid_search/rrf.py`; content unchanged |
| [workbooks/pgvector-examples/hybrid_search_cross_encoder.py](workbooks/pgvector-examples/hybrid_search_cross_encoder.py) | Hybrid search with cross-encoder re-ranking | [pgvector/pgvector-python](https://raw.githubusercontent.com/pgvector/pgvector-python/master/examples/hybrid_search/cross_encoder.py) | [MIT](workbooks/pgvector-examples/LICENSE.txt) | 2026-10-01 | Renamed from `examples/hybrid_search/cross_encoder.py`; content unchanged |
| [workbooks/pgvector-examples/rag_example.py](workbooks/pgvector-examples/rag_example.py) | Minimal RAG with pgvector and Ollama | [pgvector/pgvector-python](https://raw.githubusercontent.com/pgvector/pgvector-python/master/examples/rag/example.py) | [MIT](workbooks/pgvector-examples/LICENSE.txt) | 2026-10-01 | Renamed from `examples/rag/example.py`; content unchanged |
| [workbooks/pgvector-examples/LICENSE.txt](workbooks/pgvector-examples/LICENSE.txt) | Licence of the four scripts above | [pgvector/pgvector-python](https://raw.githubusercontent.com/pgvector/pgvector-python/master/LICENSE.txt) | MIT | 2026-10-01 | unchanged |

The pgvector extension itself is released under the [PostgreSQL Licence](https://github.com/pgvector/pgvector/blob/master/LICENSE); its README is cited, not copied. LlamaIndex and LangChain are referenced only as links.

## Own material
| File | Covers | Licence |
|---|---|---|
| [theory/01-retrieval-augmented-generation.md](theory/01-retrieval-augmented-generation.md) | Block 1 theory | CC-BY-4.0, course team |
| [theory/02-evaluating-and-improving-rag.md](theory/02-evaluating-and-improving-rag.md) | Block 2 theory | CC-BY-4.0, course team |
| [theory/03-agents-and-guardrails.md](theory/03-agents-and-guardrails.md) | Block 3 theory | CC-BY-4.0, course team |
| [theory/figures/make_figures.py](theory/figures/make_figures.py), `recall-at-k.png`, `embedding-map.png` | Figures (case-study data, all-MiniLM-L6-v2) | CC-BY-4.0, course team |
| [workbooks/01-case-study-semantic-search-postgres.ipynb](workbooks/01-case-study-semantic-search-postgres.ipynb) | Practice block 1 | Author: course team, licence CC-BY-4.0 |
| [workbooks/04-case-study-rag-evaluation.ipynb](workbooks/04-case-study-rag-evaluation.ipynb) | Practice block 2; ten labelled questions (relevance judgements by the course team) | Author: course team, licence CC-BY-4.0 |
| [workbooks/10-case-study-agent.ipynb](workbooks/10-case-study-agent.ipynb) | Practice block 3 | Author: course team, licence CC-BY-4.0 |
| [workspace/](workspace/README.md) | Package `review_assistant` with tests, exercises and solutions | Code: MIT; text: CC-BY-4.0, course team |

Parts of the theory pages and the first five labelled questions reuse the earlier course notes (course team).

## Citations
- Anthropic (2024). *Building effective agents*. https://www.anthropic.com/engineering/building-effective-agents
- Anthropic (2024). *Introducing Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- Cormack, G. V., Clarke, C. L. A. and Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR 2009*. https://doi.org/10.1145/1571941.1572114
- Es, S., James, J., Espinosa-Anke, L. and Schockaert, S. (2024). RAGAs: Automated evaluation of retrieval augmented generation. *EACL 2024 System Demonstrations*. https://arxiv.org/abs/2309.15217
- Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T. and Fritz, M. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *AISec 2023*. https://arxiv.org/abs/2302.12173
- Hou, Y., Li, J., He, Z., Yan, A., Chen, X. and McAuley, J. (2024). Bridging language and items for retrieval and recommendation. https://arxiv.org/abs/2403.03952 (case-study data)
- Husain, H. (2024). *Your AI Product Needs Evals*. https://hamel.dev/blog/posts/evals/
- Kane, A. and contributors. *pgvector* (README). https://github.com/pgvector/pgvector
- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*. https://arxiv.org/abs/2005.11401
- Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D. and Ho, D. E. (2024). Hallucination-free? Assessing the reliability of leading AI legal research tools. https://arxiv.org/abs/2405.20362
- Manning, C. D., Raghavan, P. and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
- Muennighoff, N., Tazi, N., Magne, L. and Reimers, N. (2023). MTEB: Massive text embedding benchmark. *EACL 2023*. https://arxiv.org/abs/2210.07316
- Nigam, P. et al. (2019). Semantic product search. *KDD 2019*. https://arxiv.org/abs/1907.00937
- OWASP Foundation (2025). *OWASP Top 10 for LLM Applications*. https://genai.owasp.org/llm-top-10/
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*. https://arxiv.org/abs/1908.10084
- Robertson, S. and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval* 3(4). https://doi.org/10.1561/1500000019
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2104.08663
- Yao, S. et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. https://arxiv.org/abs/2210.03629
- Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*. https://arxiv.org/abs/2306.05685
