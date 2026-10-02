# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/02-data100-text-wrangling-regex.ipynb](workbooks/02-data100-text-wrangling-regex.ipynb) | String canonicalisation, regular expressions, text extraction (Polars) | [DS-100/course-notes, `content/regex/regex.ipynb`](https://raw.githubusercontent.com/DS-100/course-notes/main/content/regex/regex.ipynb) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-10-01 | Renamed from `regex.ipynb`; content unchanged |
| [workbooks/data/county_and_state.csv](workbooks/data/county_and_state.csv), [county_and_population.csv](workbooks/data/county_and_population.csv), [log.txt](workbooks/data/log.txt) | Small data files used by workbook 02 | [DS-100/course-notes, `content/regex/data/`](https://github.com/DS-100/course-notes/tree/main/content/regex/data) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-10-01 | Unchanged |
| [workbooks/03-sklearn-hashing-vs-dict-vectorizer.ipynb](workbooks/03-sklearn-hashing-vs-dict-vectorizer.ipynb) | DictVectorizer, FeatureHasher, HashingVectorizer, CountVectorizer, TfidfVectorizer compared | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/06cfc926acb27652fb2aa5bfc583e7cb/plot_hashing_vs_dict_vectorizer.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_hashing_vs_dict_vectorizer.ipynb`; content unchanged |
| [workbooks/04-sklearn-text-classification-sparse-features.ipynb](workbooks/04-sklearn-text-classification-sparse-features.ipynb) | Linear classifiers on TF-IDF features (20 Newsgroups), confusion matrix, top features, benchmark | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/7ff1697c60d48929305821f39296dbb9/plot_document_classification_20newsgroups.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_document_classification_20newsgroups.ipynb`; content unchanged |
| [workbooks/06-sklearn-document-clustering-lsa.ipynb](workbooks/06-sklearn-document-clustering-lsa.ipynb) | Latent semantic analysis (TruncatedSVD) and k-means on news texts | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/751db3d5e6b909ff00972495eaae53df/plot_document_clustering.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_document_clustering.ipynb`; content unchanged |

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-text-as-data.md](theory/01-text-as-data.md), [theory/02-tfidf-and-text-classification.md](theory/02-tfidf-and-text-classification.md), [theory/03-error-analysis-and-limits.md](theory/03-error-analysis-and-limits.md) | Theory pages for the three blocks; partly based on the course's earlier lecture notes | CC-BY-4.0, author: course team |
| [theory/figures/make_figures.py](theory/figures/make_figures.py) and the three PNG figures | Multilingual document-term matrix sketch, top words per heading (6403, 6404, 9503), footwear confusion matrix; EBTI case-study data | CC-BY-4.0, author: course team |
| [workbooks/01-case-study-vocabulary-and-dtm.ipynb](workbooks/01-case-study-vocabulary-and-dtm.ipynb) | Vocabulary, document-term matrix, Zipf's law, preprocessing on the EBTI decision sample | CC-BY-4.0, author: course team |
| [workbooks/05-case-study-tfidf-leaderboard.ipynb](workbooks/05-case-study-tfidf-leaderboard.ipynb) | Word and character TF-IDF classifiers, per-heading metrics, leaderboard submission (round L2) | CC-BY-4.0, author: course team |
| [workbooks/07-case-study-error-analysis.ipynb](workbooks/07-case-study-error-analysis.ipynb) | Error analysis of 20 misclassified decisions, quoted heading numbers, improvements | CC-BY-4.0, author: course team |

## Citations

- Blei, D. M., Ng, A. Y. and Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Cavnar, W. B. and Trenkle, J. M. (1994). N-gram-based text categorization. *Proceedings of SDAIR-94*, 161–175.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. and Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391–407.
- Joulin, A., Grave, E., Bojanowski, P. and Mikolov, T. (2017). Bag of tricks for efficient text classification. *Proceedings of EACL 2017*, 427–431.
- Graham, P. (2002). *A Plan for Spam*. https://paulgraham.com/spam.html
- European Commission. *European Binding Tariff Information (EBTI)* database, full export (case-study data). Reuse with acknowledgement under Commission Decision 2011/833/EU. https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en
- datasets/harmonized-system: *Harmonized System nomenclature (HS 2022)*, ODC Public Domain Dedication and Licence. https://github.com/datasets/harmonized-system
- Jurafsky, D. and Martin, J. H. (2026). *Speech and Language Processing*, 3rd ed. draft. https://web.stanford.edu/~jurafsky/slp3/ (linked only)
- Landauer, T. K. and Dumais, S. T. (1997). A solution to Plato's problem: the latent semantic analysis theory of acquisition, induction, and representation of knowledge. *Psychological Review*, 104(2), 211–240.
- Manning, C. D., Raghavan, P. and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
- Mosteller, F. and Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*. Addison-Wesley.
- Ng, A. (2018). *Machine Learning Yearning*. deeplearning.ai.
- Northcutt, C. G., Athalye, A. and Mueller, J. (2021). Pervasive label errors in test sets destabilize machine learning benchmarks. *NeurIPS 2021 Datasets and Benchmarks Track*. https://arxiv.org/abs/2103.14749
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.
- Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). "Why should I trust you?": explaining the predictions of any classifier. *Proceedings of KDD 2016*, 1135–1144.
- Slapin, J. B. and Proksch, S.-O. (2008). A scaling model for estimating time-series party positions from texts. *American Journal of Political Science*, 52(3), 705–722.
- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
- Wang, S. and Manning, C. D. (2012). Baselines and bigrams: simple, good sentiment and topic classification. *Proceedings of ACL 2012*, 90–94.
- Weinberger, K., Dasgupta, A., Langford, J., Smola, A. and Attenberg, J. (2009). Feature hashing for large scale multitask learning. *Proceedings of ICML 2009*, 1113–1120.
- World Customs Organization. *What is the Harmonized System (HS)?* https://www.wcoomd.org/en/topics/nomenclature/overview/what-is-the-harmonized-system.aspx (linked only)
- Zech, J. R. et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine*, 15(11), e1002683.
- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
- Course documentation: spaCy (https://spacy.io/usage/linguistic-features) and NLTK (https://www.nltk.org/, Snowball stemmers) are linked, not copied.
