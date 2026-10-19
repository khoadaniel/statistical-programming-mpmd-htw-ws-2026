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

- The EBTI task: the labelled decisions, what the lecturer holds back, the long tail of headings (the many headings with only a few decisions each) and the allowed inputs.
- Designing the validation split: five designs, stratifying by heading with rare headings, a split stratified by heading and language against a split by date, which score to report.
- Linear models for text classification; a benchmark of eight classifiers.
- Accuracy and macro-F1 with almost 1,000 classes, rare headings, top-k accuracy.
- Error analysis: confusion matrices, 20 misclassified decisions, the most informative n-grams, average feature effects.
- Leakage through the customs' justification.
- Truncated SVD (latent semantic analysis) and clustering of descriptions.
- The limits of word counts as the motivation for language models.
- Worked example: checkpoint 1, the validation design and score with bootstrap intervals for the class board.

## Team project until the next session

A TF-IDF classifier on the team's own validation design, error analysis, and checkpoint 1 on the class board: split design, model, accuracy and macro-F1 with intervals.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
