# Session 13 · Classical NLP: bag-of-words, TF-IDF and text classification

> [!NOTE]
> **Guiding question:** What can a model learn from the words themselves?

**Learning outcomes.** Students are able to

- represent text as word counts and TF-IDF vectors
- train and evaluate a text classifier on high-dimensional data
- analyse the errors of a text model

**The case study from here on.** Sessions 13–16 work with the EU's Binding Tariff Information decisions (EBTI): a trader describes a product, customs states its code, and the Commission publishes the decision. The task is to suggest the four-digit HS heading of a new request from its description of goods, written in one of 23 languages. Models are trained on 309,529 decisions of 2017–2023 (experiments use a 50,000-decision sample) and tested on 113,188 later decisions: 2024 forms the public leaderboard, 2025–2026 the private one. The leaderboard has three rounds: L1 in this session (TF-IDF), L2 in Session 14 (embeddings or a language model) and L3 in Session 16 (after retraining with the released 2024 labels). Details: [theory page 02](theory/02-tfidf-and-text-classification.md#the-ebti-task-and-the-leaderboard) and [case-study/README.md](../../case-study/README.md).

## Session plan

**0:00–0:45 · Text as data** ([theory/01-text-as-data.md](theory/01-text-as-data.md))

- [Documents, tokens and vocabulary](theory/01-text-as-data.md#documents-tokens-and-vocabulary)
- [Preprocessing (lower-casing, stop words, stemming and lemmatisation)](theory/01-text-as-data.md#preprocessing-lower-casing-stop-words-stemming-and-lemmatisation)
- [Bag-of-words computed by hand](theory/01-text-as-data.md#bag-of-words-computed-by-hand)
- [Sparse, high-dimensional matrices](theory/01-text-as-data.md#sparse-high-dimensional-matrices)
- *Practice:* build the vocabulary of the descriptions of goods across languages and inspect the document-term matrix → [workbooks/01-case-study-vocabulary-and-dtm.ipynb](workbooks/01-case-study-vocabulary-and-dtm.ipynb)

**1:00–1:45 · TF-IDF and text classification** ([theory/02-tfidf-and-text-classification.md](theory/02-tfidf-and-text-classification.md))

- [The EBTI task: predict the customs heading of a product from its description, in 23 languages](theory/02-tfidf-and-text-classification.md#the-ebti-task-and-the-leaderboard)
- [TF-IDF](theory/02-tfidf-and-text-classification.md#tf-idf-weighting-words-by-how-informative-they-are) and [n-grams](theory/02-tfidf-and-text-classification.md#n-grams-keeping-short-phrases); [linear models for text classification](theory/02-tfidf-and-text-classification.md#linear-models-for-text-classification)
- Metrics for more than 1,000 classes: [accuracy and macro-F1](theory/02-tfidf-and-text-classification.md#accuracy-and-macro-f1-with-more-than-1000-classes), [rare headings](theory/02-tfidf-and-text-classification.md#per-heading-results-and-rare-headings), [top-k accuracy](theory/02-tfidf-and-text-classification.md#top-k-accuracy-a-short-list-for-the-officer), [validation by time](theory/02-tfidf-and-text-classification.md#validation-by-time-versus-a-random-split)
- *Practice:* case study: first leaderboard submission (L1) with a TF-IDF classifier → [workbooks/05-case-study-tfidf-leaderboard.ipynb](workbooks/05-case-study-tfidf-leaderboard.ipynb)

**2:00–2:45 · Error analysis and the limits of word counts** ([theory/03-error-analysis-and-limits.md](theory/03-error-analysis-and-limits.md))

- [Error analysis of text models](theory/03-error-analysis-and-limits.md#error-analysis-of-text-models)
- [Leakage through the customs' justification, a text that exists only after the decision](theory/03-error-analysis-and-limits.md#leakage-through-the-customs-justification)
- [The most informative n-grams per class](theory/03-error-analysis-and-limits.md#the-most-informative-n-grams-per-class)
- [Dimensionality reduction with truncated SVD](theory/03-error-analysis-and-limits.md#dimensionality-reduction-with-truncated-svd)
- [Limits of word counts (word order, synonyms) as the motivation for language models](theory/03-error-analysis-and-limits.md#the-limits-of-word-counts)
- *Practice:* analyse 20 misclassified decisions with their English keywords and heading texts, and improve the classifier → [workbooks/07-case-study-error-analysis.ipynb](workbooks/07-case-study-error-analysis.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-text-as-data.md](theory/01-text-as-data.md) | Tokens, vocabulary, preprocessing in a multilingual corpus (stemming, compounds), bag-of-words, sparse matrices | 1 | core |
| [theory/02-tfidf-and-text-classification.md](theory/02-tfidf-and-text-classification.md) | The EBTI task and the leaderboard, TF-IDF, word and character n-grams, linear SVM with 1,000 headings, accuracy and macro-F1, per-heading metrics, top-k accuracy, validation by time versus a random split, leaderboard round L1 | 2 | core |
| [theory/03-error-analysis-and-limits.md](theory/03-error-analysis-and-limits.md) | Footwear confusion matrix, error categories, leakage through the justification, top n-grams and quoted codes, truncated SVD, limits of counts | 3 | core |
| [workbooks/01-case-study-vocabulary-and-dtm.ipynb](workbooks/01-case-study-vocabulary-and-dtm.ipynb) | Case study: reading decisions with keywords, tokens per language, document-term matrix, Zipf's law, preprocessing, compounds | 1 | core |
| [workbooks/02-data100-text-wrangling-regex.ipynb](workbooks/02-data100-text-wrangling-regex.ipynb) | Data 100: string canonicalisation and regular expressions (Polars) | 1 | optional |
| [workbooks/03-sklearn-hashing-vs-dict-vectorizer.ipynb](workbooks/03-sklearn-hashing-vs-dict-vectorizer.ipynb) | scikit-learn: vectorisers compared, hashing trick, sparse output | 1 | optional |
| [workbooks/04-sklearn-text-classification-sparse-features.ipynb](workbooks/04-sklearn-text-classification-sparse-features.ipynb) | scikit-learn: linear classifiers on TF-IDF (20 Newsgroups), confusion matrix, top features | 2 | optional |
| [workbooks/05-case-study-tfidf-leaderboard.ipynb](workbooks/05-case-study-tfidf-leaderboard.ipynb) | Case study: always-3926 baseline, word and character TF-IDF classifiers, per-heading report, top-k accuracy, random versus time split, `submission-L1-tfidf.csv` for round L1 | 2 | core |
| [workbooks/06-sklearn-document-clustering-lsa.ipynb](workbooks/06-sklearn-document-clustering-lsa.ipynb) | scikit-learn: LSA (truncated SVD) and k-means on news texts | 3 | optional |
| [workbooks/07-case-study-error-analysis.ipynb](workbooks/07-case-study-error-analysis.ipynb) | Case study: 20 misclassified decisions, top n-grams, quoted heading numbers, the justification leak, improvements | 3 | core |

Sources and licences of third-party files: [source.md](source.md).

## Before and after the session

**Preparation.** Make sure `case-study/data/` exists (run `case-study/prepare_data.py`, see [case-study/README.md](../../case-study/README.md)). Revise logistic regression (Session 6), validation and leakage (Sessions 7 and 9), precision, recall, F1 and the confusion matrix (Session 8), backtesting by time (Session 12) and PCA (Session 11). Read the case-study description of the EBTI data in [case-study/README.md](../../case-study/README.md). Read the WCO page *What is the Harmonized System?* (link below) to know what a heading is.

**Team project until the next session.** Text features where the project uses text; otherwise model improvement.

**Further reading (optional).**

- Jurafsky, D. and Martin, J. H. *Speech and Language Processing*, 3rd ed. draft, chapters on words and tokens, naive Bayes and sentiment, logistic regression: https://web.stanford.edu/~jurafsky/slp3/
- World Customs Organization, *What is the Harmonized System (HS)?*: https://www.wcoomd.org/en/topics/nomenclature/overview/what-is-the-harmonized-system.aspx
- scikit-learn tutorial *Working with text data*: https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html (documentation of version 1.4; the tutorial is not part of later versions)
- Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, chapters 2 and 6 (tokens, TF-IDF): https://nlp.stanford.edu/IR-book/
- spaCy 101 (tokenisation and lemmatisation in many languages; outlook to linguistic methods in module 3.4): https://spacy.io/usage/spacy-101

## Setup

Everything runs in the course environment (`uv sync` at the repository root, then `uv run jupyter lab`); `nltk` (Snowball stemmers for German and French, no data download needed) is part of it.

```bash
# from the repository root
uv run jupyter lab
```

The models are trained on the 50,000-decision sample; the case-study workbooks 05 and 07 take about 9 and 6 minutes on a laptop (the character n-gram model is the slow part). Workbooks 03, 04 and 06 download the 20 Newsgroups dataset (about 14 MB) on first use. Workbook 02 reads the small files in [workbooks/data/](workbooks/data/). The figures of the theory pages are made by [theory/figures/make_figures.py](theory/figures/make_figures.py).
