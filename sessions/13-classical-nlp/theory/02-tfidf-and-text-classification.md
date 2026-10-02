# TF-IDF, n-grams and linear text classifiers

A customs officer who receives a new request wants a suggestion: which heading is this product most likely in? This page sets up that task, the EBTI case study of Sessions 13–16, and builds the first model for it. Raw counts treat every word alike: "mit" (with) counts as much as "Spielzeug" (toy). TF-IDF is a weighting that makes informative words count more, and n-grams keep short phrases such as "leather upper" or pieces of words such as `zeug`. Linear models are the standard classifiers for this kind of sparse, high-dimensional input, even with more than 1,000 classes. With so many classes, most of them rare, a single accuracy figure is not enough, so the page shows how to evaluate such a model: accuracy and macro-F1, per-heading results by frequency, top-k accuracy, and validation by time instead of a random split. The page ends with the practice task of this block: the first leaderboard round (L1) with a TF-IDF classifier.

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

## The EBTI task and the leaderboard

### Concept

**The decision to support.** A trader asks a customs authority how a product is classified in the EU tariff; customs answers with a Binding Tariff Information (BTI) decision that states the code, and the European Commission publishes all decisions in the EBTI database. Classifying a product takes an expert's time, and the tariff has more than 1,000 four-digit **headings**. A model that reads the description of goods and proposes the most likely headings can save that time, provided we know how often it is right and where it fails.

**The data** ([case-study/README.md](../../../case-study/README.md)):

| Table | Rows | What it contains |
|---|---|---|
| `train` | 309,529 decisions, 2017–2023 | description, language, issuing country, start date, the label `heading`, and columns written by customs together with the decision (keywords, justification, full code) |
| `train_sample` | 50,000 | a random sample of `train`, used for all experiments in this session |
| `test` | 113,188 decisions, 2024–2026 | only what a request contains: `id`, issuing country, language, start date, description |
| `nomenclature` | 1,229 headings | the English text of every heading, chapter and section |

Descriptions are written in the language of the issuing country (57 % German, 16 % French, 5 % English, 23 languages in all); every training decision also has English keywords that help us read it.

**The split is by time.** The model is trained on decisions up to 2023 and predicts later ones, as it would in use. The **public leaderboard** scores the decisions of 2024, the **private leaderboard** those of 2025–2026, which stay hidden until the end of the course. Test decisions whose description repeats a training description are removed, so the task cannot be solved by looking up old decisions. A submission is a CSV file with one row per test decision and the columns `id,heading`. The metric is **accuracy**, with **macro-F1** reported alongside (both explained below). There are three rounds: **L1** with a TF-IDF classifier (this session), **L2** with embeddings or a language model (Session 14), and **L3** after retraining with the released 2024 labels (Session 16). The leaderboard is not graded, because decisions can be looked up in the public database.

### Why it matters

Before choosing a method we fix what a good answer is. The test set contains only what a request contains, so a model may use only those columns; the time split makes the validation score a forecast of performance on future requests; and the leaderboard gives every team the same hidden test, so results can be compared honestly across the three rounds.

### How it works in Python

```python
import pandas as pd

decisions = pd.read_parquet("case-study/data/train_sample.parquet")
test = pd.read_parquet("case-study/data/test.parquet")
print(len(decisions), decisions["heading"].nunique(), len(test))   # 50000 934 113188
print(test.columns.tolist())   # ['id', 'issuing_country', 'language', 'start_date', 'description']
print(test["start_date"].dt.year.value_counts().sort_index().to_dict())
# {2024: 40369, 2025: 41126, 2026: 31693}: 2024 public, 2025-2026 private
print(decisions["language"].value_counts(normalize=True).head(3).round(2).to_dict())
# {'de': 0.57, 'fr': 0.16, 'en': 0.05}
print(decisions["heading"].value_counts().head(3).to_dict())       # {'3926': 2009, '9503': 1424, '6307': 1374}
```

### In practice

- Customs administrations and the World Customs Organization have built tools that suggest HS codes from the free-text goods descriptions of declarations; an officer confirms or corrects the suggestion.
- Kaggle and Codabench competitions follow the protocol of the course leaderboard: a hidden test set, a fixed submission format, a public leaderboard during the competition and a private one for the final ranking.

> [!WARNING]
> `keywords`, `classification_justification`, `cn_code`, `chapter`, `status` and the end dates are written by customs together with or after the decision. They are not in the test set and must not be model inputs. Block 3 measures what happens when this rule is broken.

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

**Validation by time.** The leaderboard asks for decisions of 2024–2026 given decisions of 2017–2023, so we validate the same way: train on 2017–2021, validate on 2022–2023. The section on validation below measures how much a random split would overstate the result.

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

Results of further runs (same split, documented in the course team's experiments): no lower-casing 0.758; `LinearSVC` on word unigrams 0.769 in 96 seconds.

### In practice

- Routing e-mails and support tickets to the right team is often done with TF-IDF and a linear model, because it is cheap to retrain when categories change.
- Automatic coding of occupations, industries and products into official classifications (ISCO, NACE, HS) is a long-standing application of linear text classifiers in official statistics, often with a human checking low-confidence cases (Session 14, abstention).
- Joulin et al. (2017) showed with fastText that a linear classifier on word and character n-grams reaches accuracy close to deep neural networks on standard text-classification benchmarks while training in seconds.

> [!WARNING]
> **Do not compare models on the test set.** Choose the analyzer, n-gram range, `min_df` and `alpha` on the validation years. The leaderboard is used once per round, for the final choice.

> [!TIP]
> Grid search over a text pipeline multiplies the fitting time. With 1,000 classes, start with one setting per idea (word vs character, unigrams vs bigrams), keep the best, and only then tune `alpha` on a small grid.

## Accuracy and macro-F1 with more than 1,000 classes

### Concept

With two classes, Session 8 computed precision, recall and F1 for the positive class. With many classes the same measures are computed **per class**, treating that class as positive and all others as negative (one-vs-rest):

- **Precision** of heading k = correct predictions of k / all predictions of k. "When the model says 9503 (toys), how often is it right?"
- **Recall** of heading k = correct predictions of k / all true members of k. "Of all toy decisions, how many does the model find?"
- **F1** = 2 × precision × recall / (precision + recall), the harmonic mean; it is low if either of the two is low.
- **Support** = the number of true members of k in the evaluation data.

Three averages summarise the per-class values:

| Average | Definition | Effect with rare classes |
|---|---|---|
| macro | plain mean of the per-class F1 values | each of the ~800 headings counts equally, also those with one decision |
| weighted | mean weighted by support | dominated by frequent headings |
| micro | computed from all predictions pooled; equals accuracy for single-label tasks | dominated by frequent headings |

Worked example with ten decisions: six of heading 3926 (plastic articles), two of 6403 (leather footwear) and two of 6404 (textile footwear). The model gets all six 3926 right, one of two 6403 and one of two 6404; the two errors are predicted as 3926. Per-class F1: 3926 0.86 (precision 6/8, recall 6/6), 6403 0.67, 6404 0.67. Macro-F1 = (0.86 + 0.67 + 0.67) / 3 = 0.73; accuracy = 8 / 10 = 0.80.

### Why it matters

The headings have a **long tail**: the most frequent heading (3926) covers 4 % of the decisions, and 466 of the 890 headings in the training years 2017–2021 of the sample have fewer than ten decisions. Accuracy is dominated by the few hundred frequent headings and answers "what share of requests gets the right suggestion?". Macro-F1 gives a heading with 3 decisions the same weight as one with 2,000 and answers "how well does the model know the tariff as a whole?". Always predicting 3926 gives accuracy 0.04 and a macro-F1 close to 0. A model that is good on frequent headings and poor on rare ones reaches a high accuracy and a modest macro-F1, which is what the reference models of the leaderboard show (word TF-IDF on the full training set: accuracy 0.872, macro-F1 0.682 on 2024). The leaderboard ranks by accuracy, because an officer's workload depends on the share of requests; macro-F1 is reported next to it so that the tail stays visible.

### How it works in Python

```python
y_true = ["3926"] * 6 + ["6403", "6403", "6404", "6404"]
y_pred = ["3926"] * 6 + ["3926", "6403", "3926", "6404"]
print(f1_score(y_true, y_pred, average=None, labels=["3926", "6403", "6404"]).round(2))   # [0.86 0.67 0.67]
print(round(f1_score(y_true, y_pred, average="macro"), 2), accuracy_score(y_true, y_pred))   # 0.73 0.8

always_3926 = np.repeat("3926", len(valid))                        # the simplest baseline
print(round(accuracy_score(valid["heading"], always_3926), 3),
      round(f1_score(valid["heading"], always_3926, average="macro"), 4))   # 0.04 0.0001

pred = models["linear SVM, word TF-IDF 1"].predict(valid["description"])
for avg in ["macro", "weighted", "micro"]:
    print(avg, round(f1_score(valid["heading"], pred, average=avg, zero_division=0), 3))
# macro 0.511 / weighted 0.76 / micro 0.772 (= accuracy)
```

### In practice

- Shared tasks in text classification, such as SemEval-2017 Task 4 (Rosenthal et al., 2017), use macro-averaged measures so that rare classes count.
- Automatic coding of occupations, causes of death or economic activities by statistical offices deals with hundreds of codes and a long tail; evaluations report accuracy together with per-class or macro-averaged results.

> [!WARNING]
> `f1_score` on more than two classes needs `average=`; always say which average you report. Macro-F1 also depends on which classes appear in the evaluation set at all, so compare macro-F1 values only on the same set.

## Per-heading results and rare headings

### Concept

A list of 1,000 per-heading F1 values cannot be read, but two views can: the per-heading report sorted by support, and the mean F1 of headings **grouped by how many training decisions they have**. The second view answers directly whether the errors concentrate in the tail.

Worked example: the validation years contain 523 decisions of heading 3926 (other articles of plastics). The model predicts 3926 for 655 decisions, of which 442 are correct. Precision = 442 / 655 = 0.67, recall = 442 / 523 = 0.85, F1 = 2 × 0.67 × 0.85 / 1.52 = 0.75. The model uses 3926 as a fallback: many decisions about other plastic goods (3923 containers, 3924 tableware) are pushed into it, which lowers its precision.

### Why it matters

In the validation years, 415 of the 778 headings that occur have fewer than five decisions, and 237 headings get F1 = 0. The grouped table below shows why: F1 depends strongly on how many training examples a heading has. This tells us what to improve: more data for rare headings helps more than tuning the frequent ones.

### How it works in Python

```python
from sklearn.metrics import classification_report

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

## Top-k accuracy: a short list for the officer

### Concept

**Top-k accuracy** counts a prediction as correct if the true heading is among the k headings with the highest scores. It measures a model that proposes candidates to a person, which is how a classification tool for customs officers or traders is used: the officer reads three suggestions with their heading texts and picks one. A linear model gives a score for every heading (`decision_function`), so the top k are the k largest scores of each row.

### Why it matters

An officer can check three suggestions in seconds, so a model whose second or third suggestion is often right is more useful than its accuracy says. Top-k recall is also the upper bound for any method that chooses among the model's candidates, such as the language model of Session 14 that picks one of ten candidate headings.

### How it works in Python

```python
svm_word = models["linear SVM, word TF-IDF 1"]
scores = svm_word.decision_function(valid["description"])        # one score per heading
order = np.argsort(-scores, axis=1)
y_val = valid["heading"].to_numpy()
for k in [1, 3, 5, 10]:
    top_k = svm_word.classes_[order[:, :k]]
    print(k, round((top_k == y_val[:, None]).any(axis=1).mean(), 3))
# 1 0.772
# 3 0.838
# 5 0.856
# 10 0.876
print(round((~valid["heading"].isin(train["heading"])).mean(), 3))   # 0.01: headings never seen in training
```

With three suggestions the true heading is in the list for 84 % of the decisions instead of 77 %. `sklearn.metrics.top_k_accuracy_score` computes the same, but only when every validation heading also occurs in training; here 1 % of the validation decisions have a heading the model has never seen.

### In practice

- Search engines and recommender systems are evaluated by top-k measures (precision at k, recall at k), because users look at a short list.
- Tools for automatic coding of free text into classifications (occupations, economic activities, products) usually propose a few candidate codes for a human coder to confirm.

## Validation by time versus a random split

### Concept

A **random split** draws the validation decisions from all years; a **split by time** validates on the most recent years only. Both use the same number of decisions, but they answer different questions. A random split asks "how well does the model classify a decision like the ones it was trained on?"; a split by time asks "how well will it classify next year's requests?", which is the leaderboard's question and the officer's.

### Why it matters

Three things make the random split too optimistic here. Renewed decisions with near-identical descriptions from the same trader end up on both sides; the product mix and the wording change over the years (Session 16 measures this drift); and headings that were changed in the HS 2022 revision appear in training with their new meaning only in the later years. On the sample, the random split reports 0.816 accuracy and 0.571 macro-F1, the split by time 0.772 and 0.511. The leaderboard sides with the time split: the same model, trained on all 50,000 decisions, scores 0.803 on 2024 and 0.776 on 2025–2026 (practice section below).

### How it works in Python

```python
from sklearn.model_selection import train_test_split

rand_train, rand_valid = train_test_split(decisions, test_size=len(valid), random_state=0)
random_model = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), svm())
random_model.fit(rand_train["description"], rand_train["heading"])
pred_r = random_model.predict(rand_valid["description"])
print(round(accuracy_score(rand_valid["heading"], pred_r), 3),
      round(f1_score(rand_valid["heading"], pred_r, average="macro"), 3))   # 0.816 0.571
# time split with the same sizes (36,801 / 13,199): 0.772 0.511
print(round(rand_valid["description"].isin(rand_train["description"]).mean(), 3),
      round(valid["description"].isin(train["description"]).mean(), 3))   # 0.016 0.004: exact repeats
```

Exact repeats explain only a small part of the gap (1.6 % against 0.4 % of the validation descriptions); most of it comes from near-repeats and from change over time.

### In practice

- Forecasting competitions and credit-scoring teams validate "out of time": the model is fitted on older periods and tested on the most recent one, because that is how it will be used.
- Kapoor and Narayanan (2023) list random splits of temporal data among the common forms of leakage in published machine-learning studies.

> [!CAUTION]
> Choose models and settings on the validation years, never on the leaderboard. Every submission chosen by its public score is a small step of tuning on the test set; use the leaderboard to check, not to search.

## Practice: leaderboard round L1 with a TF-IDF classifier

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
2. A model predicts 9503 for 40 decisions, 30 of them correctly; the validation set has 120 decisions of 9503. Compute precision, recall and F1 for 9503.
3. Always predicting 3926 gives accuracy 0.04 and macro-F1 0.0001. Why does macro-F1 punish this rule so much more than accuracy, and which of the two would a head of a classification unit care about first?
4. Why is top-3 accuracy a sensible measure for a tool that assists customs officers? What does a top-10 accuracy of 0.876 mean for a method that chooses among the model's ten best headings?
5. Why is a random train-validation split more optimistic than a split by year on this data set? Name two reasons.

## Further reading

- Jurafsky, D. and Martin, J. H. (2026). *Speech and Language Processing*, 3rd ed. draft, chapters "Naive Bayes, Text Classification and Sentiment" and "Logistic Regression". https://web.stanford.edu/~jurafsky/slp3/
- scikit-learn developers. *Classification of text documents using sparse features* (example). https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
- Joulin, A., Grave, E., Bojanowski, P. and Mikolov, T. (2017). Bag of tricks for efficient text classification. *Proceedings of EACL 2017*, 427–431. https://aclanthology.org/E17-2068/
- European Commission. *European Binding Tariff Information (EBTI)*: what a BTI decision is and how to search the database. https://taxation-customs.ec.europa.eu/online-services/online-services-and-databases-customs/european-binding-tariff-information-ebti_en
