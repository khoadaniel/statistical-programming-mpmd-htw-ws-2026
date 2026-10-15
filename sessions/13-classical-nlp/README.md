# Session 13 · Classical NLP: bag-of-words, TF-IDF and text classification

> **Guiding question.** What can a model learn from the words themselves?

## 1 · Text as data

[Open notebook](workbooks/01-lecture-text-as-data.ipynb)

- Documents, tokens, types and the vocabulary; out-of-vocabulary words; frequency dictionaries and `DictVectorizer`.
- Preprocessing in 23 languages: lower-casing, stop words, stemming and stemmers compared, lemmatisation, compounds; cleaning with regular expressions.
- Bag-of-words and the document-term matrix computed by hand.
- TF-IDF by hand and in scikit-learn, the parameters of `TfidfVectorizer`; word and character n-grams.
- Sparse matrices (CSR), Zipf's law, `min_df` and `max_df`, the hashing trick.
- Simple text statistics as features, alone and next to TF-IDF in one pipeline.
- Worked example: the vocabulary of the EBTI descriptions across languages.

## 2 · Text classification and error analysis

[Open notebook](workbooks/02-lecture-text-classification-and-error-analysis.ipynb)

- The EBTI task, the random split stratified by heading, the leaderboard, the long tail of headings and the allowed inputs.
- Linear models for text classification; a benchmark of eight classifiers.
- Accuracy and macro-F1 with almost 1,000 classes, rare headings, top-k accuracy, a split by date against the random split.
- Error analysis: confusion matrices, 20 misclassified decisions, the most informative n-grams, average feature effects, quoted codes.
- Leakage through the customs' justification.
- Truncated SVD (latent semantic analysis) and clustering of descriptions.
- The limits of word counts as the motivation for language models.
- Worked example: leaderboard round L1.

## Team project until the next session

TF-IDF classifier on the team's validation scheme, error analysis, leaderboard round L1.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
