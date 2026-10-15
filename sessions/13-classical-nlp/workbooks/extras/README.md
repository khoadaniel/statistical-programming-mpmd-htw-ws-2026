# Session 13 · Extras and notebook provenance

Downloaded third-party notebooks, numbered in block order; parts of them are curated into the two [block notebooks](../../README.md).

## Downloaded notebooks

| File | Content | Block | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|---|
| [01-data100-text-wrangling-regex.ipynb](01-data100-text-wrangling-regex.ipynb) | Data 100 course notes: string canonicalisation, regular expressions, capture groups and extraction (Polars); reads the files in [data/](data/) | 1 | [DS-100/course-notes, `content/regex/regex.ipynb`](https://raw.githubusercontent.com/DS-100/course-notes/main/content/regex/regex.ipynb) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-10-01 | Renamed from `regex.ipynb`; content unchanged (an earlier change of the data paths to `../data/` was undone on 2026-10-14, when the data moved to `extras/data/`). Curated into the Block 1 notebook, section "Preprocessing" (cleaning text with string methods and regular expressions), rewritten for pandas and EBTI data |
| [02-sklearn-hashing-vs-dict-vectorizer.ipynb](02-sklearn-hashing-vs-dict-vectorizer.ipynb) | scikit-learn: `DictVectorizer`, `FeatureHasher`, `HashingVectorizer`, `CountVectorizer` and `TfidfVectorizer` compared; the hashing trick | 1 | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/06cfc926acb27652fb2aa5bfc583e7cb/plot_hashing_vs_dict_vectorizer.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_hashing_vs_dict_vectorizer.ipynb`; content unchanged. Curated into the Block 1 notebook, sections "Documents, tokens and vocabulary" (frequency dictionaries and `DictVectorizer`), "TF-IDF and n-grams" (`TfidfTransformer`) and "Sparse, high-dimensional matrices" (the hashing trick) |
| [03-sklearn-column-transformer-text-stats.ipynb](03-sklearn-column-transformer-text-stats.ipynb) | scikit-learn: `ColumnTransformer` with TF-IDF on subject and body plus text statistics via `FunctionTransformer` and `DictVectorizer` | 1 | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/15dc6d7a809edf988a7328336a25faec/plot_column_transformer.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-14 | Renamed from `plot_column_transformer.ipynb`; content unchanged. Curated into the Block 1 notebook, section "Simple text statistics as features" (text statistics and TF-IDF in one pipeline) |
| [04-sklearn-text-classification-sparse-features.ipynb](04-sklearn-text-classification-sparse-features.ipynb) | scikit-learn: linear classifiers on TF-IDF features (20 Newsgroups), confusion matrix, feature effects, metadata leakage, benchmark of eight classifiers | 2 | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/7ff1697c60d48929305821f39296dbb9/plot_document_classification_20newsgroups.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_document_classification_20newsgroups.ipynb` (was `extras/03-...` until 2026-10-14); content unchanged. Curated into the Block 2 notebook, sections "Linear models for text classification" (benchmark), "Error analysis and the most informative n-grams" (confusion matrix, average feature effects) and "Leakage through the customs' justification" (the metadata leak) |
| [05-sklearn-document-clustering-lsa.ipynb](05-sklearn-document-clustering-lsa.ipynb) | scikit-learn: latent semantic analysis (truncated SVD) and k-means on news texts, clustering scores, top terms per cluster | 2 | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/751db3d5e6b909ff00972495eaae53df/plot_document_clustering.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_document_clustering.ipynb` (was `extras/04-...` until 2026-10-14); content unchanged. Curated into the Block 2 notebook, section "Dimensionality reduction with truncated SVD" (clustering descriptions with LSA and k-means) |

Extras 02–05 download the 20 Newsgroups dataset (about 14 MB) on first use; extra 01 uses Polars, which the course environment provides for this notebook only (course-written material uses pandas).

## Authors and licences

- **Data 100 (UC Berkeley), course notes:** extra 01 and its data files, BSD-3-Clause.
- **scikit-learn developers, example gallery:** extras 02–05, BSD-3-Clause.
- The block notebooks are course material under CC BY 4.0; their sections marked "Adapted from" curate BSD-3-Clause code and text from these sources, with attribution at each section.

## Data files

| File | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|
| [data/county_and_state.csv](data/county_and_state.csv), [data/county_and_population.csv](data/county_and_population.csv), [data/log.txt](data/log.txt) | [DS-100/course-notes, `content/regex/data/`](https://github.com/DS-100/course-notes/tree/main/content/regex/data) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-10-01 | Unchanged; used by extra 01. Moved from `workbooks/data/` to `workbooks/extras/data/` on 2026-10-14 |

**Case-study data** (read by the block notebooks from `case-study/data/ebti/`): European Commission, *European Binding Tariff Information (EBTI)* database, full export, reuse with acknowledgement under Commission Decision 2011/833/EU; datasets/harmonized-system, *Harmonized System nomenclature (HS 2022)*, ODC Public Domain Dedication and Licence. Prepared by the lecturer and downloaded as `ebti.zip` from the course drive; see [case-study/README.md](../../../../case-study/README.md).

## Figures

| File | Content | Licence |
|---|---|---|
| [../../figures/dtm_sketch.png](../../figures/dtm_sketch.png) | multilingual document-term matrix sketch, used in the Block 1 notebook, section "Bag-of-words computed by hand" | CC BY 4.0, course team |
| [../../figures/top_ngrams.png](../../figures/top_ngrams.png), [../../figures/confusion_matrix.png](../../figures/confusion_matrix.png) | top words per heading (6403, 6404, 9503) and footwear confusion matrix; no longer used by the notebooks, which now compute both inline | CC BY 4.0, course team |

All three are generated from EBTI case-study data by [../../figures/make_figures.py](../../figures/make_figures.py) (`uv run python sessions/13-classical-nlp/figures/make_figures.py`).

## Archived course notebooks

The course-written notebooks `01-lecture-text-as-data.ipynb` (earlier version), `02-case-study-vocabulary-and-dtm.ipynb`, `03-lecture-text-classification-and-error-analysis.ipynb`, `04-case-study-tfidf-leaderboard.ipynb`, `05-case-study-error-analysis.ipynb` and `session-13-exercises.ipynb` (CC BY 4.0) were folded into the two block notebooks on 2026-10-14 and deleted; git history keeps them. The section on text statistics adapts the course's earlier Session 9 page on text statistics to the EBTI descriptions.

## Linked, not copied

Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft); Manning, Raghavan and Schütze, *Introduction to Information Retrieval*; spaCy and NLTK documentation (Snowball stemmers, used through the `nltk` package without data downloads); the WCO page on the Harmonized System.
