# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/02-hf-course-tokenizers.ipynb](workbooks/02-hf-course-tokenizers.ipynb) | Loading tokenisers, tokenising, encoding and decoding (Hugging Face LLM course, chapter 2, section 4, PyTorch) | [huggingface/notebooks, `course/en/chapter2/section4_pt.ipynb`](https://raw.githubusercontent.com/huggingface/notebooks/main/course/en/chapter2/section4_pt.ipynb) | [Apache-2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE) | 2026-10-01 | Renamed from `section4_pt.ipynb`; content unchanged |
| [workbooks/03-hf-course-bpe-tokenization.ipynb](workbooks/03-hf-course-bpe-tokenization.ipynb) | Byte-pair encoding implemented from scratch (chapter 6, section 5) | [huggingface/notebooks, `course/en/chapter6/section5.ipynb`](https://raw.githubusercontent.com/huggingface/notebooks/main/course/en/chapter6/section5.ipynb) | [Apache-2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE) | 2026-10-01 | Renamed from `section5.ipynb`; content unchanged |
| [workbooks/04-hf-course-transformer-pipelines.ipynb](workbooks/04-hf-course-transformer-pipelines.ipynb) | `pipeline()` for classification, zero-shot, generation, fill-mask, NER, QA, summarisation, translation (chapter 1, section 3) | [huggingface/notebooks, `course/en/chapter1/section3.ipynb`](https://raw.githubusercontent.com/huggingface/notebooks/main/course/en/chapter1/section3.ipynb) | [Apache-2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE) | 2026-10-01 | Renamed from `section3.ipynb`; content unchanged |
| [workbooks/06-sbert-retrieve-rerank-wikipedia.ipynb](workbooks/06-sbert-retrieve-rerank-wikipedia.ipynb) | Semantic search with a bi-encoder, re-ranking with a cross-encoder, comparison with BM25 | [huggingface/sentence-transformers, `examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb`](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) | [Apache-2.0](https://github.com/huggingface/sentence-transformers/blob/main/LICENSE) | 2026-10-01 | Renamed from `retrieve_rerank_simple_wikipedia.ipynb`; content unchanged |
| [workbooks/07-openai-cookbook-clustering.ipynb](workbooks/07-openai-cookbook-clustering.ipynb) | k-means clustering of precomputed review embeddings, t-SNE visualisation, cluster naming with an LLM | [openai/openai-cookbook, `examples/Clustering.ipynb`](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Clustering.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `Clustering.ipynb`. Patched one line: the data path `./data/fine_food_reviews_with_embeddings_1k.csv` was replaced by the raw GitHub URL of the same file in the cookbook repository (35 MB, not copied into this repository) |
| [workbooks/08-openai-cookbook-structured-outputs.ipynb](workbooks/08-openai-cookbook-structured-outputs.ipynb) | Structured outputs with JSON schema and pydantic (`response_format`) | [openai/openai-cookbook, `examples/Structured_Outputs_Intro.ipynb`](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Structured_Outputs_Intro.ipynb) | [MIT](https://github.com/openai/openai-cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `Structured_Outputs_Intro.ipynb`; content unchanged |

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-from-words-to-vectors.md](theory/01-from-words-to-vectors.md), [theory/02-applications-of-embeddings.md](theory/02-applications-of-embeddings.md), [theory/03-generative-models-through-an-api.md](theory/03-generative-models-through-an-api.md) | Theory pages for the three blocks; partly based on the course's earlier lecture notes; the abstention section reuses the course's earlier page on decision thresholds | CC-BY-4.0, author: course team |
| [theory/figures/make_figures.py](theory/figures/make_figures.py) and the four PNG figures | Tokenisation example (SentencePiece and BPE, English and German), illustrative attention weights, 2-D UMAP projection of 1,500 EBTI decisions embedded with multilingual-e5-small, coverage–accuracy curve of a TF-IDF classifier on the EBTI decisions (abstention) | CC-BY-4.0, author: course team |
| [workbooks/01-case-study-tokens-and-similarity.ipynb](workbooks/01-case-study-tokens-and-similarity.ipynb) | Token counts per language and cross-lingual semantic similarity on the EBTI decision sample | CC-BY-4.0, author: course team |
| [workbooks/05-case-study-embeddings-vs-tfidf.ipynb](workbooks/05-case-study-embeddings-vs-tfidf.ipynb) | Embedding features vs TF-IDF, learning curve, cross-lingual nearest-neighbour search | CC-BY-4.0, author: course team |
| [workbooks/09-case-study-llm-vs-trained-classifier.ipynb](workbooks/09-case-study-llm-vs-trained-classifier.ipynb) | Zero/few-shot LLM choosing among ten candidate headings with structured output vs trained classifier on 200 decisions | CC-BY-4.0, author: course team |
| [workbooks/10-case-study-leaderboard-l2.ipynb](workbooks/10-case-study-leaderboard-l2.ipynb) | Leaderboard round L2: TF-IDF plus multilingual embeddings with a paired bootstrap against round L1; optional LLM choice for unsure requests with the trained model as fallback | CC-BY-4.0, author: course team |

Models used (downloaded at run time, not stored here): `intfloat/multilingual-e5-small` (MIT, https://huggingface.co/intfloat/multilingual-e5-small); optionally `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0); via Ollama `qwen2.5:3b` (Qwen research licence, see https://ollama.com/library/qwen2.5) or `llama3.2:3b` (Llama 3.2 Community License, see https://ollama.com/library/llama3.2). Check the model licence before using a model in a project.

## Citations

- Alammar, J. (2018). *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/ (linked only)
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V. and Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS 2016*.
- Brown, T. et al. (2020). Language models are few-shot learners. *NeurIPS 2020*. https://arxiv.org/abs/2005.14165
- Conneau, A. et al. (2020). Unsupervised cross-lingual representation learning at scale (XLM-RoBERTa). *Proceedings of ACL 2020*. https://arxiv.org/abs/1911.02116
- Chan, B., Schweter, S. and Möller, T. (2020). German's next language model. *Proceedings of COLING 2020*, 6788–6796.
- Cohan, A., Feldman, S., Beltagy, I., Downey, D. and Weld, D. S. (2020). SPECTER: document-level representation learning using citation-informed transformers. *Proceedings of ACL 2020*.
- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL 2019*. https://arxiv.org/abs/1810.04805
- European Data Protection Board (2025). *AI Privacy Risks & Mitigations: Large Language Models*. https://www.edpb.europa.eu/our-work-tools/our-documents/support-pool-experts-projects/ai-privacy-risks-mitigations-large_en
- European Commission. *European Binding Tariff Information (EBTI)* database, full export (case-study data). Reuse with acknowledgement under Commission Decision 2011/833/EU. https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en
- datasets/harmonized-system: *Harmonized System nomenclature (HS 2022)*, ODC-PDDL. https://github.com/datasets/harmonized-system
- Firth, J. R. (1957). A synopsis of linguistic theory 1930–1955. In *Studies in Linguistic Analysis*. Blackwell.
- Gilardi, F., Alizadeh, M. and Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30), e2305016120.
- Grbovic, M. and Cheng, H. (2018). Real-time personalization using embeddings for search ranking at Airbnb. *Proceedings of KDD 2018*.
- Grootendorst, M. (2022). BERTopic: neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- Hugging Face (2025). *LLM Course*. https://huggingface.co/learn/llm-course (course notebooks: https://github.com/huggingface/notebooks, Apache-2.0)
- Jain, S. and Wallace, B. C. (2019). Attention is not explanation. *Proceedings of NAACL 2019*.
- Jurafsky, D. and Martin, J. H. (2026). *Speech and Language Processing*, 3rd ed. draft. https://web.stanford.edu/~jurafsky/slp3/ (linked only)
- Liu, N. F. et al. (2024). Lost in the middle: how language models use long contexts. *Transactions of the ACL*, 12, 157–173.
- Mikolov, T., Chen, K., Corrado, G. and Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv:1301.3781.
- Muennighoff, N., Tazi, N., Magne, L. and Reimers, N. (2023). MTEB: massive text embedding benchmark. *Proceedings of EACL 2023*.
- OpenAI. *OpenAI Cookbook*. https://cookbook.openai.com/ (MIT)
- Petrov, A., La Malfa, E., Torr, P. H. S. and Bibi, A. (2023). Language model tokenizers introduce unfairness between languages. *NeurIPS 2023*.
- Radford, A., Narasimhan, K., Salimans, T. and Sutskever, I. (2018). *Improving language understanding by generative pre-training*. OpenAI.
- Raffel, C. et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.
- Reimers, N. and Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. *Proceedings of EMNLP 2020*. https://arxiv.org/abs/2004.09813
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*. https://sbert.net/
- Sennrich, R., Haddow, B. and Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of ACL 2016*.
- Strubell, E., Ganesh, A. and McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. *Proceedings of ACL 2019*.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I. (2021). BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS 2021 Datasets and Benchmarks*.
- Tshitoyan, V. et al. (2019). Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571, 95–98.
- Tunstall, L. et al. (2022). Efficient few-shot learning without prompts (SetFit). arXiv:2209.11055.
- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R. and Wei, F. (2024). Multilingual E5 text embeddings: a technical report. arXiv:2402.05672.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
- Court and regulatory cases cited on theory page 3: *Mata v. Avianca, Inc.*, No. 22-cv-1461 (S.D.N.Y. 2023); *Moffatt v. Air Canada*, 2024 BCCRT 149; Garante per la protezione dei dati personali, provision of 30 March 2023 on ChatGPT.
- Xiong, M. et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR 2024*. https://arxiv.org/abs/2306.13063
