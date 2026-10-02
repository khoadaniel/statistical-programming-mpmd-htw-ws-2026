# TF-IDF, n-grams and linear text classifiers

Raw counts treat every word alike: "mit" (with) counts as much as "Spielzeug" (toy). This page introduces TF-IDF, a weighting that makes informative words count more, and n-grams, which keep short phrases such as "leather upper" or pieces of words such as `zeug`. It then shows why linear models are the standard classifiers for this kind of sparse, high-dimensional input, even with more than 1,000 classes, and how to read per-class metrics when most headings are rare. The page ends with the practice task of this block: the third leaderboard round (L3) with a TF-IDF classifier.

The code blocks on this page build on each other; run them in order from the repository root. The model comparison takes about one and a half minutes, the character n-gram model about two minutes.

```mermaid
flowchart LR
    T["Training decisions<br/>2017-2021"] --> V["TfidfVectorizer<br/>(words or char_wb 3-5)"]
    V --> C["Linear SVM<br/>(SGDClassifier, hinge)"]
    C --> E["Validation 2022-2023:<br/>accuracy, macro-F1,<br/>per-heading report"]
    E -->|"choose settings"| F["Refit on all<br/>training decisions"]
    F --> P["Predict test<br/>decisions 2024-2026"]
    P --> S["submission.csv<br/>id,heading"]
```

## TF-IDF: weighting words by how informative they are

### Concept

**Term frequency** tf(t, d) is the count of term t in document d. **Document frequency** df(t) is the number of documents that contain t at least once. With N documents, the **inverse document frequency** is

  idf(t) = log(N / df(t)).

It is large for rare terms and zero for a term that occurs in every document. **TF-IDF** is the product tf(t, d) × idf(t): a term gets a high weight in a document if it is frequent there and rare elsewhere.

Worked example with N = 3 documents: d1 = "plastic toy plastic box", d2 = "plastic bag", d3 = "wooden toy".

| Term | tf in d1 | df | idf = ln(3 / df) | tf × idf |
|---|---|---|---|---|
| plastic | 2 | 2 | ln(1.5) = 0.405 | 0.811 |
| toy | 1 | 2 | 0.405 | 0.405 |
| box | 1 | 1 | ln(3) = 1.099 | 1.099 |

"box" occurs once in d1, but it gets the highest weight because no other document uses it.

scikit-learn's `TfidfVectorizer` makes three changes to this textbook formula:

1. **Smoothed idf**: idf(t) = ln((1 + N) / (1 + df(t))) + 1. It acts as if one extra document contained every term, so no division by zero occurs, and the "+ 1" keeps terms that appear everywhere from being removed entirely. For the example: plastic and toy get ln(4/3) + 1 = 1.288, box gets ln(4/2) + 1 = 1.693.
2. **L2 normalisation**: each row is divided by its Euclidean length, so that every document vector has length 1. Long German and short French descriptions become comparable, and the dot product of two rows is their cosine similarity. For d1: the raw weights (2 × 1.288, 1.288, 1.693) = (2.576, 1.288, 1.693) have length 3.34, giving (0.771, 0.386, 0.507).
3. **Sublinear tf** (optional, `sublinear_tf=True`): tf is replaced by 1 + ln(tf). The tenth "Kunststoff" in a description then adds much less than the second.

### Why it matters

Raw counts are dominated by frequent words that carry little information, and in a multilingual corpus there is a separate set of such words for every language. TF-IDF down-weights all of them automatically, without a stop-word list per language, and the row normalisation removes the effect of description length (German descriptions are more than twice as long as French ones). TF-IDF is the standard input for linear text classifiers and was the core of classic keyword search.

### How it works in Python

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["plastic toy plastic box", "plastic bag", "wooden toy"]
# by hand for document 1, with scikit-learn's smoothed idf (N = 3 documents)
tf = {"plastic": 2, "toy": 1, "box": 1}             # counts in document 1
df = {"plastic": 2, "toy": 2, "box": 1}             # documents that contain the word
idf = {w: np.log((1 + 3) / (1 + df[w])) + 1 for w in tf}
print({w: round(float(v), 3) for w, v in idf.items()})   # plastic 1.288, toy 1.288, box 1.693
w = np.array([tf[t] * idf[t] for t in tf])
print((w / np.linalg.norm(w)).round(3))             # [0.771 0.386 0.507] after L2 normalisation

tfidf = TfidfVectorizer()
T = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())                # ['bag' 'box' 'plastic' 'toy' 'wooden']
print(T.toarray().round(3)[0])                      # [0.    0.507 0.771 0.386 0.   ]
print(np.linalg.norm(T.toarray(), axis=1))          # [1. 1. 1.]: every row has length 1

# on the case-study sample: which words get the lowest and highest idf?
decisions = pd.read_parquet("case-study/data/train_sample.parquet")
texts = decisions["description"]
vec = TfidfVectorizer(min_df=3).fit(texts)
idf_s = pd.Series(vec.idf_, index=vec.get_feature_names_out()).sort_values()
print(idf_s.index[:8].tolist())     # ['in', 'mit', 'aus', 'und', 'der', 'als', 'einem', 'von']
print(idf_s.round(2).iloc[[0, -1]].to_dict())       # {'in': 1.54, 'blaubeere': 10.43}
```

The lowest idf values belong to German function words, because German is the most frequent language; the highest go to words that occur in exactly three descriptions, such as "Blaubeere" (blueberry).

### In practice

- TF-IDF ranking was the core of classic search engines. Apache Lucene, on which Elasticsearch and Solr are built, used a TF-IDF similarity by default until version 6 (2016), when it switched to BM25, a refinement of the same idea with saturating term frequency and length normalisation (Session 15 uses BM25).
- Keyword extraction: the terms with the highest TF-IDF weight in a document summarise what distinguishes it from the rest of the corpus. Digital libraries and newsrooms use this to tag documents.
- Spärck Jones (1972) introduced the idea of weighting terms by their inverse document frequency in a paper on information retrieval; it is still the default weighting in many retrieval systems.

> [!NOTE]
> TF-IDF is still a bag-of-words representation: it changes the *values* in the document-term matrix, not its columns. The matrix keeps the same shape and the same sparsity.

## n-grams: keeping short phrases

### Concept

An **n-gram** is a sequence of n consecutive tokens. Unigrams are single words, **bigrams** are pairs ("leather upper"), **trigrams** are triples. `ngram_range=(1, 2)` adds all bigrams to the vocabulary next to the unigrams.

Worked example from the footwear chapter: heading 6403 covers shoes with uppers of leather, heading 6404 shoes with uppers of textile. "leather upper, textile sole" and "textile upper, leather sole" contain the same four words. With unigrams their vectors are identical. With bigrams the first contains `leather upper` and `textile sole`, the second `textile upper` and `leather sole`, so a model can learn that the material of the *upper* decides the heading.

The cost is size. The number of distinct bigrams grows much faster than the number of words. In the case-study sample with `min_df=3` there are about 67,000 unigrams, 236,000 features with bigrams and 420,000 with trigrams.

**Character n-grams** use sequences of letters instead of words. With `analyzer="char_wb"` scikit-learn builds them inside word boundaries: "zeug" gives ` zeu`, `zeug`, `eug `. They are robust to misspellings and inflections and, most importantly here, they let a German compound such as "Kunststoffspielzeug" share features with "Spielzeug" and "Kunststoff". The price is many more features per document and longer training.

### Why it matters

For tariff classification, short phrases ("aus Leder", "en matière plastique", "for children") and word pieces (material and product names inside compounds) carry the decisive information. On the case-study sample, word bigrams do *not* help (validation accuracy 0.769 against 0.772 for unigrams, see below), while character n-grams do (0.788). This is an empirical result for this data, not a general rule: Wang and Manning (2012) found bigrams useful for English sentiment, where negations matter. Always measure.

### How it works in Python

```python
pair = ["leather upper, textile sole", "textile upper, leather sole"]
uni = CountVectorizer().fit(pair)
print(uni.transform(pair).toarray())       # [[1 1 1 1] [1 1 1 1]]: identical vectors
bi = CountVectorizer(ngram_range=(1, 2)).fit(pair)
print(bi.get_feature_names_out())
# ['leather' 'leather sole' 'leather upper' 'sole' 'textile' 'textile sole'
#  'textile upper' 'upper' 'upper leather' 'upper textile']
print(bi.transform(pair).toarray())
# [[1 0 1 1 1 1 0 1 0 1]
#  [1 1 0 1 1 0 1 1 1 0]]: now different

for rng in [(1, 1), (1, 2), (1, 3)]:
    v = TfidfVectorizer(ngram_range=rng, min_df=3).fit(texts)
    print(rng, len(v.vocabulary_))         # (1, 1) 66997 / (1, 2) 236112 / (1, 3) 419615

# character n-grams link a compound to its parts
ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(["Kunststoffspielzeug"])
print(ch.transform(["Spielzeugauto"]).nnz)  # 18 of the 36 character n-grams of 'Spielzeugauto' are known
```

### In practice

- Language identification uses character n-grams; Cavnar and Trenkle (1994) described the classic n-gram frequency method that many language detectors still follow.
- The fastText classifier of Facebook AI Research represents words by their character n-grams, so that it handles morphologically rich languages and words it has never seen; its authors later published pre-trained vectors for 157 languages.
- The Google Books Ngram Viewer charts how often words and phrases (up to five-grams) appeared in millions of digitised books over time.

> [!WARNING]
> Every additional n-gram range multiplies the number of features and the memory needed. Always combine n-grams with `min_df` (for example 2–5) or `max_features`, and check `len(vectorizer.vocabulary_)` before fitting a model.

## Linear models for text classification

### Concept

A **linear classifier** computes one score per class as a weighted sum of the features, plus an intercept: score_k(d) = b_k + Σ_j w_kj x_dj. It predicts the class with the highest score. For text, each feature is one n-gram, so each weight says how much that n-gram pushes a decision towards a heading.

Worked example: suppose the model has the weights below for heading 6404 (textile uppers) and a description has TF-IDF values `textile` = 0.5, `leather` = 0.4, `textile upper` = 0.6.

| n-gram | weight (6404) | value | product |
|---|---|---|---|
| textile | 2.0 | 0.5 | 1.0 |
| leather | −3.0 | 0.4 | −1.2 |
| textile upper | 4.0 | 0.6 | 2.4 |

With intercept −1.0 the score for 6404 is −1.0 + 1.0 − 1.2 + 2.4 = 1.2. The bigram outweighs the word "leather" (which here refers to the sole). The model computes such a score for each of the 890 headings in the training data and picks the largest.

Three linear classifiers are common for text:

| Model | Idea | Output | Notes |
|---|---|---|---|
| Multinomial or complement naive Bayes | word probabilities per class, assumed independent | probabilities (poorly calibrated) | very fast; works on counts |
| Logistic regression | minimises log-loss; softmax turns scores into probabilities (Session 6) | probabilities | slow with 1,000 classes and 300,000 features |
| Linear support vector machine | maximises the margin between classes | scores only | here trained with `SGDClassifier(loss="hinge")` |

With many classes, scikit-learn trains one binary model per class ("one versus rest"). `SGDClassifier` fits these models by **stochastic gradient descent**: it passes over the data a fixed number of times (`max_iter`) and updates the weights after each example, which is fast enough for 300,000 decisions and 1,100 headings. Its parameter `alpha` is the strength of the **regularisation** penalty (larger = stronger); `LogisticRegression` and `LinearSVC` use `C`, the inverse.

The vectoriser and the classifier belong into one **pipeline**. Then `fit` learns the vocabulary and idf weights only from the training part.

**Validation by time.** The leaderboard asks for decisions of 2024–2026 given decisions of 2017–2023. A random split mixes years and lets renewed decisions with near-identical descriptions appear on both sides. We therefore validate like the leaderboard: train on 2017–2021, validate on 2022–2023 (Session 7).

### Why it matters

Linear models are fast on sparse input, need little tuning, and are easy to inspect, because each weight belongs to a readable n-gram. A TF-IDF linear model is the baseline every more complex text model has to beat; on the full training set it reaches 0.87–0.88 public accuracy (case-study README). Tree ensembles (Session 10) are a poor fit here: they split on one feature at a time and struggle with hundreds of thousands of sparse columns and 1,000 classes.

### How it works in Python

```python
import time

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

year = decisions["start_date"].dt.year
train, valid = decisions[year <= 2021], decisions[year >= 2022]      # validation by time
print(len(train), len(valid), train["heading"].nunique())            # 36801 13199 890


def svm():                                       # a linear SVM fitted by stochastic gradient descent
    return SGDClassifier(loss="hinge", alpha=1e-5, max_iter=20, tol=None, random_state=0, n_jobs=-1)


models = {
    "naive Bayes, word counts 1-2": make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2),
                                                  ComplementNB()),
    "linear SVM, word TF-IDF 1": make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), svm()),
    "linear SVM, word TF-IDF 1-2": make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), svm()),
}
for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(train["description"], train["heading"])
    pred = model.predict(valid["description"])
    print(f"{name:30s} acc {accuracy_score(valid['heading'], pred):.3f}  "
          f"macro-F1 {f1_score(valid['heading'], pred, average='macro'):.3f}  "
          f"{time.perf_counter() - t0:.0f} s")
# naive Bayes, word counts 1-2   acc 0.673  macro-F1 0.392  26 s
# linear SVM, word TF-IDF 1      acc 0.772  macro-F1 0.511  24 s
# linear SVM, word TF-IDF 1-2    acc 0.769  macro-F1 0.507  37 s

# character n-grams inside word boundaries (about two minutes)
char = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True, max_features=300_000), svm())
char.fit(train["description"], train["heading"])
pred_char = char.predict(valid["description"])
print(round(accuracy_score(valid["heading"], pred_char), 3),
      round(f1_score(valid["heading"], pred_char, average="macro"), 3))   # 0.788 0.529
```

Results of further runs (same split, documented in the course team's experiments): no lower-casing 0.758; `LinearSVC` on word unigrams 0.769 in 96 seconds; a **random** split instead of the time split gives 0.814, five points more optimistic than the honest time-based estimate.

### In practice

- Routing e-mails and support tickets to the right team is often done with TF-IDF and a linear model, because it is cheap to retrain when categories change.
- Automatic coding of occupations, industries and products into official classifications (ISCO, NACE, HS) is a long-standing application of linear text classifiers in official statistics, often with a human checking low-confidence cases (Session 8).
- Joulin et al. (2017) showed with fastText that a linear classifier on word and character n-grams reaches accuracy close to deep neural networks on standard text-classification benchmarks while training in seconds.

> [!WARNING]
> **Do not compare models on the test set.** Choose the analyzer, n-gram range, `min_df` and `alpha` on the validation years. The leaderboard is used once per round, for the final choice.

> [!TIP]
> Grid search over a text pipeline multiplies the fitting time. With 1,000 classes, start with one setting per idea (word vs character, unigrams vs bigrams), keep the best, and only then tune `alpha` on a small grid.

## Per-class metrics

### Concept

For each class k, compare the predictions with the true labels:

- **Precision** = correct predictions of k / all predictions of k. "When the model says 9503 (toys), how often is it right?"
- **Recall** = correct predictions of k / all true members of k. "Of all toy decisions, how many does the model find?"
- **F1** = 2 × precision × recall / (precision + recall), the harmonic mean; it is low if either of the two is low.
- **Support** = the number of true members of k in the evaluation data.

Three averages summarise the per-class values:

| Average | Definition | Effect with rare classes |
|---|---|---|
| macro | plain mean of the per-class F1 values | each of the ~800 headings counts equally, also those with one decision |
| weighted | mean weighted by support | dominated by frequent headings |
| micro | computed from all predictions pooled; equals accuracy for single-label tasks | dominated by frequent headings |

Worked example: the validation years contain 523 decisions of heading 3926 (other articles of plastics). The model predicts 3926 for 655 decisions, of which 442 are correct. Precision = 442 / 655 = 0.67, recall = 442 / 523 = 0.85, F1 = 2 × 0.67 × 0.85 / 1.52 = 0.75. The model uses 3926 as a fallback: many decisions about other plastic goods (3923 containers, 3924 tableware) are pushed into it, which lowers its precision.

### Why it matters

With about 1,100 headings and a long tail, accuracy and macro-F1 tell different stories. Accuracy is dominated by the frequent headings and answers "what share of decisions is right?". Macro-F1 gives every heading the same weight and answers "how well does the model know the tariff as a whole?". In the validation years, 415 of the 778 headings that occur have fewer than five decisions, and 237 headings get F1 = 0. The table below shows why: F1 depends strongly on how many training examples a heading has.

### How it works in Python

```python
from sklearn.metrics import classification_report

pred = models["linear SVM, word TF-IDF 1"].predict(valid["description"])
report = pd.DataFrame(classification_report(valid["heading"], pred, output_dict=True,
                                            zero_division=0)).T.iloc[:-3]   # drop the averages
print(report.sort_values("support", ascending=False).head(5).round(2))
#       precision  recall  f1-score  support
# 3926       0.67    0.85      0.75    523.0
# 9503       0.77    0.96      0.85    372.0
# 6307       0.74    0.90      0.81    360.0
# 2106       0.73    0.90      0.81    347.0
# 4202       0.89    0.95      0.92    325.0
print(len(report), int((report["f1-score"] == 0).sum()), int((report["support"] < 5).sum()))
# 778 237 415: headings in the validation years, with F1 = 0, with fewer than 5 decisions

for avg in ["macro", "weighted", "micro"]:
    print(avg, round(f1_score(valid["heading"], pred, average=avg, zero_division=0), 3))
# macro 0.511 / weighted 0.76 / micro 0.772 (= accuracy)

# F1 by the number of training decisions of a heading
n_train = train["heading"].value_counts().reindex(report.index, fill_value=0)
bins = pd.cut(n_train, [-1, 0, 10, 50, 200, np.inf], labels=["0", "1-10", "11-50", "51-200", ">200"])
print(report["f1-score"].groupby(bins, observed=True).agg(["mean", "count"]).round(2))
#         mean  count
# 0       0.00     44      <- headings never seen in training cannot be predicted
# 1-10    0.36    328
# 11-50   0.64    250
# 51-200  0.77    120
# >200    0.79     36
```

### In practice

- Content-moderation systems report recall per category (for example spam, hate speech, self-harm), because a high overall accuracy can hide a category that is almost never detected.
- Medical screening tests are described by sensitivity (recall of the "ill" class) and specificity (recall of the "healthy" class), the same per-class view under different names.
- Extreme multi-label classification, for example assigning codes from large medical or product taxonomies, reports macro-averaged and "tail" metrics separately, because a model can look good on frequent codes while ignoring thousands of rare ones.

> [!IMPORTANT]
> A model trained on 50,000 decisions sees on average about 55 examples per heading, but the median heading has far fewer. More training data (the full 309,529 decisions) helps the tail most: on the public leaderboard, macro-F1 rises from 0.52 to 0.68 while accuracy rises from 0.81 to 0.87 (case-study README).

## Practice: leaderboard round L3 with a TF-IDF classifier

The task is to train a TF-IDF classifier on the 50,000-decision sample, predict the 113,188 test decisions, and submit a file with one row per test decision and the columns `id,heading`. Workbook [05-case-study-tfidf-leaderboard.ipynb](../workbooks/05-case-study-tfidf-leaderboard.ipynb) contains the full exercise, including a character n-gram model and an optional run on the full training set.

```python
test = pd.read_parquet("case-study/data/test.parquet")
final = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), svm())
final.fit(decisions["description"], decisions["heading"])        # all 50,000 sample decisions
submission = pd.DataFrame({"id": test["id"], "heading": final.predict(test["description"])})
print(submission.shape, submission["heading"].value_counts(normalize=True).head(3).round(3).to_dict())
# (113188, 2) {'3926': 0.052, '9503': 0.037, '2106': 0.035}
submission.to_csv("submission.csv", index=False)
```

Score it locally (where the solution file is available) or upload it to the course leaderboard:

```bash
uv run python case-study/score.py submission.csv
# private_accuracy     0.7762
# private_macro_f1     0.4863
# public_accuracy      0.8033
# public_macro_f1      0.5177
```

With character n-grams (3–5, `char_wb`) on the same sample the scores are 0.817 public and 0.796 private accuracy (about four minutes). The public score is higher than the validation score (0.772) because the final model also learns from 2022–2023, the years closest to the test period. The private years 2025–2026 are about three points lower: the further the test period is from the training period, the more the data have changed (Session 16).

## Check your understanding

1. Compute the textbook idf = ln(N / df) of a word that occurs in 500 of 50,000 descriptions, and of a word that occurs in all of them. What does scikit-learn's smoothed idf give for the second word?
2. Why does L2 normalisation make a short French and a long German description with the same word proportions get the same weight pattern?
3. Give two descriptions that have identical unigram vectors but belong to different headings, and show which bigrams separate them.
4. A model predicts 9503 for 40 decisions, 30 of them correctly; the validation set has 120 decisions of 9503. Compute precision, recall and F1 for 9503.
5. Why is a random train-validation split more optimistic than a split by year on this data set? Name two reasons.

## Further reading

- Jurafsky, D. and Martin, J. H. (2026). *Speech and Language Processing*, 3rd ed. draft, chapters "Naive Bayes, Text Classification and Sentiment" and "Logistic Regression". https://web.stanford.edu/~jurafsky/slp3/
- scikit-learn developers. *Classification of text documents using sparse features* (example). https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
- Joulin, A., Grave, E., Bojanowski, P. and Mikolov, T. (2017). Bag of tricks for efficient text classification. *Proceedings of EACL 2017*, 427–431. https://aclanthology.org/E17-2068/
- European Commission. *European Binding Tariff Information (EBTI)*: what a BTI decision is and how to search the database. https://taxation-customs.ec.europa.eu/online-services/online-services-and-databases-customs/european-binding-tariff-information-ebti_en
