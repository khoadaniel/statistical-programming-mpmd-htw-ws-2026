# Session 13 · Classical NLP: bag-of-words, TF-IDF and text classification

> [!NOTE]
> **Guiding question:** What can a model learn from the words themselves?

**Learning outcomes.** Students are able to

- represent text as word counts and TF-IDF vectors
- train and evaluate a text classifier on high-dimensional data
- analyse the errors of a text model

## Session plan

**0:00–0:45 · Text as data** ([theory/01-text-as-data.md](theory/01-text-as-data.md))

- [Documents, tokens and vocabulary](theory/01-text-as-data.md#documents-tokens-and-vocabulary)
- [Preprocessing (lower-casing, stop words, stemming and lemmatisation)](theory/01-text-as-data.md#preprocessing-lower-casing-stop-words-stemming-and-lemmatisation)
- [Bag-of-words computed by hand](theory/01-text-as-data.md#bag-of-words-computed-by-hand)
- [Sparse, high-dimensional matrices](theory/01-text-as-data.md#sparse-high-dimensional-matrices)
- *Practice:* build the vocabulary of the reviews and inspect the document-term matrix → [workbooks/01-case-study-vocabulary-and-dtm.ipynb](workbooks/01-case-study-vocabulary-and-dtm.ipynb)

**1:00–1:45 · TF-IDF and text classification** ([theory/02-tfidf-and-text-classification.md](theory/02-tfidf-and-text-classification.md))

- [TF-IDF](theory/02-tfidf-and-text-classification.md#tf-idf-weighting-words-by-how-informative-they-are) and [n-grams](theory/02-tfidf-and-text-classification.md#n-grams-keeping-short-phrases)
- [Linear models for text classification](theory/02-tfidf-and-text-classification.md#linear-models-for-text-classification)
- [Per-class metrics](theory/02-tfidf-and-text-classification.md#per-class-metrics)
- *Practice:* case study: leaderboard submission (round L3) with a TF-IDF classifier → [workbooks/05-case-study-tfidf-leaderboard.ipynb](workbooks/05-case-study-tfidf-leaderboard.ipynb)

**2:00–2:45 · Error analysis and the limits of word counts** ([theory/03-error-analysis-and-limits.md](theory/03-error-analysis-and-limits.md))

- [Error analysis of text models](theory/03-error-analysis-and-limits.md#error-analysis-of-text-models)
- [The most informative n-grams per class](theory/03-error-analysis-and-limits.md#the-most-informative-n-grams-per-class)
- [Dimensionality reduction with truncated SVD](theory/03-error-analysis-and-limits.md#dimensionality-reduction-with-truncated-svd)
- [Limits of word counts (word order, synonyms) as the motivation for language models](theory/03-error-analysis-and-limits.md#the-limits-of-word-counts)
- *Practice:* analyse 20 misclassified reviews and improve the classifier → [workbooks/07-case-study-error-analysis.ipynb](workbooks/07-case-study-error-analysis.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-text-as-data.md](theory/01-text-as-data.md) | Tokens, vocabulary, preprocessing, bag-of-words, sparse matrices | 1 | core |
| [theory/02-tfidf-and-text-classification.md](theory/02-tfidf-and-text-classification.md) | TF-IDF, n-grams, linear classifiers, per-class metrics, leaderboard round L3 | 2 | core |
| [theory/03-error-analysis-and-limits.md](theory/03-error-analysis-and-limits.md) | Confusion matrix, error categories, top n-grams and shortcuts, truncated SVD, limits of counts | 3 | core |
| [workbooks/01-case-study-vocabulary-and-dtm.ipynb](workbooks/01-case-study-vocabulary-and-dtm.ipynb) | Case study: tokens, vocabulary, document-term matrix, Zipf's law, preprocessing | 1 | core |
| [workbooks/02-data100-text-wrangling-regex.ipynb](workbooks/02-data100-text-wrangling-regex.ipynb) | Data 100: string canonicalisation and regular expressions (Polars) | 1 | optional |
| [workbooks/03-sklearn-hashing-vs-dict-vectorizer.ipynb](workbooks/03-sklearn-hashing-vs-dict-vectorizer.ipynb) | scikit-learn: vectorisers compared, hashing trick, sparse output | 1 | optional |
| [workbooks/04-sklearn-text-classification-sparse-features.ipynb](workbooks/04-sklearn-text-classification-sparse-features.ipynb) | scikit-learn: linear classifiers on TF-IDF (20 Newsgroups), confusion matrix, top features | 2 | optional |
| [workbooks/05-case-study-tfidf-leaderboard.ipynb](workbooks/05-case-study-tfidf-leaderboard.ipynb) | Case study: TF-IDF classifier, per-class report, `submission.csv` for round L3 | 2 | core |
| [workbooks/06-sklearn-document-clustering-lsa.ipynb](workbooks/06-sklearn-document-clustering-lsa.ipynb) | scikit-learn: LSA (truncated SVD) and k-means on news texts | 3 | optional |
| [workbooks/07-case-study-error-analysis.ipynb](workbooks/07-case-study-error-analysis.ipynb) | Case study: 20 misclassified reviews, top n-grams, star-title shortcut, improvements | 3 | core |

Sources and licences of third-party files: [source.md](source.md).

## Before and after the session

**Preparation.** Make sure `case-study/data/` exists (run `case-study/prepare_data.py`, see [case-study/README.md](../../case-study/README.md)). Revise logistic regression (Session 6), macro-F1 and the confusion matrix (Session 8), and PCA (Session 11). Read the first section of the scikit-learn tutorial *Working with text data* (link below).

**Team project until the next session.** Text features where the project uses text; otherwise model improvement.

**Further reading (optional).**

- Jurafsky, D. and Martin, J. H. *Speech and Language Processing*, 3rd ed. draft, chapters on words and tokens, naive Bayes and sentiment, logistic regression: https://web.stanford.edu/~jurafsky/slp3/
- scikit-learn tutorial *Working with text data*: https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html (documentation of version 1.4; the tutorial is not part of later versions)
- Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, chapters 2 and 6 (tokens, TF-IDF): https://nlp.stanford.edu/IR-book/
- spaCy 101 (tokenisation, lemmatisation; outlook to linguistic methods in module 3.4): https://spacy.io/usage/spacy-101
- NLTK book, chapter 3 *Processing raw text*: https://www.nltk.org/book/ch03.html

## Setup

Everything except stemming runs in the course environment (`uv sync` at the repository root, then `uv run jupyter lab`). Extra packages:

- `nltk` (Porter stemmer in theory page 1 and workbook 01; no data download needed)
- optional: `spacy` with the model `en_core_web_sm` for lemmatisation (theory page 1)

```bash
# from the repository root
uv run --with nltk jupyter lab
# optional lemmatisation example
uv run --with nltk --with spacy \
  --with "en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
  jupyter lab
```

Workbooks 03, 04 and 06 download the 20 Newsgroups dataset (about 14 MB) on first use. Workbook 02 reads the small files in [workbooks/data/](workbooks/data/).
