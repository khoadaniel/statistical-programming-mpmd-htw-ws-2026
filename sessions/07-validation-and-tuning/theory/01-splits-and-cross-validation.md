# Data splits, cross-validation and the bootstrap

This page covers the first block of Session 7. In Session 6 you split the data once into a training and a test set and reported one test score. That score answers the question "how well will the model do on new data?" only roughly: another split gives another number. Here we build the tools that make the answer reliable: a clear role for each part of the data, cross-validation that uses every row for validation once, splitters that respect groups and time, and a bootstrap interval that says how uncertain a score is.

The code blocks on this page build on each other: run them in order from the repository root. The Telco churn data (7,043 customers) are downloaded from IBM's GitHub repository and serve as a small warm-up. The main example is the Inside Airbnb listings for Berlin in `case-study/data/airbnb/` (`uv run python case-study/prepare_airbnb.py`, see [case-study/README.md](../../../case-study/README.md)).

```mermaid
flowchart LR
    A["All labelled data"] --> B["Development data"]
    A --> T["Test set<br/>(locked away)"]
    B --> CV["Cross-validation:<br/>train and validate k times"]
    CV --> C["Choose model and<br/>hyperparameters"]
    C --> F["Refit on all<br/>development data"]
    F --> E["Evaluate once<br/>on the test set"]
    T --> E
```

## Training, validation and test data and their roles

**Concept.** Labelled data are used for three different jobs, and each job needs its own rows.

- The **training set** is used to fit the model: the algorithm sees these rows and their labels and chooses the parameters (for example the coefficients of a regression).
- The **validation set** is used to make choices: which model, which features, which settings. We compare candidates by their validation score.
- The **test set** is used once, at the end, to estimate how well the chosen model will perform on new data. It must not influence any choice.

A worked example. A bank has 10,000 past loan applications with known outcomes. It keeps 2,000 applications aside as the test set. Of the remaining 8,000 it uses 6,000 to fit three candidate models and 2,000 to compare them. Model B wins on the validation rows. Only now does the bank score model B on the 2,000 test applications: 81 % accuracy. This number is an honest estimate because no decision depended on the test rows. The validation score of model B (say 83 %) is slightly optimistic: B was chosen *because* it did well on those rows.

**Why it matters.** Each time a score is used to make a decision, the rows behind it stop being "new data". If you choose a model by its test score, the test score is no longer an unbiased estimate; with enough candidates one of them will look good by chance. Keeping the three roles apart is the basic discipline of all model evaluation. In this course the leaderboard of Sessions 13–16 plays the role of the test set: its labels are hidden, so nobody can tune on them.

**How it works in Python.** Two calls of `train_test_split` give a 60/20/20 split. `stratify=y` keeps the share of churners equal in all three parts.

```python
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(URL)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
y = (df["Churn"] == "Yes").astype(int)               # 1 = customer left
X = df.drop(columns=["customerID", "Churn"])

# 20 % test set, then 25 % of the remaining 80 % as validation set
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=0)
print(len(X_tr), len(X_val), len(X_test))           # 4225 1409 1409

num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat = [c for c in X.columns if c not in num]
prep = make_column_transformer((StandardScaler(), num), (OneHotEncoder(handle_unknown="ignore"), cat))

# choose the regularisation strength C on the validation set (more on C in theory page 02)
for C in [0.001, 0.01, 1.0]:
    m = make_pipeline(prep, LogisticRegression(C=C, max_iter=1000)).fit(X_tr, y_tr)
    print(C, round(m.score(X_tr, y_tr), 3), round(m.score(X_val, y_val), 3))
# 0.001 0.773 0.767
# 0.01  0.806 0.796
# 1.0   0.807 0.803   <- best validation accuracy

# refit the chosen setting on training + validation data, evaluate ONCE on the test set
final = make_pipeline(prep, LogisticRegression(C=1.0, max_iter=1000)).fit(X_dev, y_dev)
print(round(final.score(X_test, y_test), 3))        # 0.801
```

**In practice.**
- The Netflix Prize (2006–2009) kept a "quiz" set for the public leaderboard and a separate hidden "test" set that decided the winner, so that teams could not win by tuning to the leaderboard.
- The TRIPOD reporting guideline for clinical prediction models (Collins et al., 2015) asks authors to state how data were split for development and validation, and treats a separate external test as stronger evidence than internal validation.
- Kaggle competitions show a public leaderboard during the competition and rank by a private part of the test set at the end. Rankings often move between the two ("shake-up"), which shows how much teams had adapted to the public part.

> [!WARNING]
> Looking at the test score "just to see" and then changing the model turns the test set into a validation set. Decide on the model first, then look at the test score once and report it, even if it is disappointing.

> [!TIP]
> Fix `random_state` in every split and every model that uses randomness. Then a colleague who runs your notebook gets the same numbers.

## Why a single split is unreliable

**Concept.** A test score is computed on a finite, randomly chosen set of rows. Another random split puts other customers into the test set and gives another score. The score is therefore a **random variable**: it has a mean (the quantity we want, the expected performance on new data) and a spread (noise caused by the particular split). The spread shrinks with the size of the test set: for accuracy, the standard error is roughly √(p(1 − p)/n). With p = 0.8 and n = 1,761 test customers this is √(0.16/1761) ≈ 0.0095, about one percentage point. With n = 200 it is almost three points.

**Why it matters.** Model comparisons, hyperparameter choices and project reports all rest on such estimates. If the score moves by two points when only the seed changes, an "improvement" of one point is not evidence. A single split also wastes data: the rows in the test set never help to train, and the rows in the training set never help to evaluate.

**How it works in Python.** The same model, evaluated on 20 random 75/25 splits:

```python
model = make_pipeline(prep, LogisticRegression(max_iter=1000))

acc = []
for seed in range(20):
    a_tr, a_te, b_tr, b_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    acc.append(model.fit(a_tr, b_tr).score(a_te, b_te))
print(np.round([min(acc), np.mean(acc), max(acc)], 3))   # [0.785 0.802 0.822]
print(round(np.std(acc), 3))                             # 0.009
```

The range from 0.785 to 0.822 comes only from chance. The observed standard deviation (0.009) agrees with the formula above.

**In practice.**
- Kaggle "shake-ups": in competitions with small test sets, teams that ranked high on the public leaderboard regularly drop many places on the private leaderboard, because the public part was too small to separate models.
- The ImageNet replication study by Recht et al. (2019) built new test sets for CIFAR-10 and ImageNet and found accuracy drops of 3–15 percentage points on the new samples. The authors attribute the drops mainly to small differences in how the new images were collected, not to adaptive overfitting: a test score is tied to the particular test sample.

> [!CAUTION]
> Never report a difference between two models that is smaller than the spread of the score across splits as an improvement. Report the spread (or an interval) along with the score.

## k-fold and stratified cross-validation

**Concept.** **k-fold cross-validation** divides the rows into *k* parts (**folds**) of roughly equal size. The model is trained *k* times, each time on *k* − 1 folds, and scored on the remaining fold. Every row is used for validation exactly once. The *k* scores are summarised by their mean (the estimate) and their standard deviation (how stable it is). Common choices are *k* = 5 or *k* = 10; **leave-one-out** (*k* = *n*) is the extreme case.

![Five-fold cross-validation (left): each row of blocks is one split; the orange block is the validation fold, the blue blocks are used for training. Time-series split (right): training always lies before validation.](figures/cv_splits.png)

A small example by hand: 10 rows, *k* = 5. Fold 1 = rows 1–2, fold 2 = rows 3–4, and so on. In split 1 the model trains on rows 3–10 and is scored on rows 1–2; in split 2 it trains on rows 1–2 and 5–10 and is scored on rows 3–4. After five splits each row has been predicted once by a model that did not see it.

**Stratified k-fold** chooses the folds so that each has the same class proportions as the full data. With 26.5 % churners, plain k-fold can produce folds with 24.6 % and 28.9 % churn; stratified folds all have 26.5 %. This matters most for rare classes and small data.

**Why it matters.** Cross-validation uses all data for both training and validation, so the estimate has lower variance than one split, and the fold-to-fold spread shows how much to trust it. Comparing training and validation scores in the same run shows over- or underfitting (theory page 02).

**How it works in Python.** `cross_val_score` returns one score per fold; `cross_validate` returns several metrics, fit times and, optionally, training scores. For classifiers, `cv=5` already means stratified 5-fold; pass a splitter object to control shuffling and the seed.

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate

kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
print(scores.round(3))                                    # [0.797 0.813 0.797 0.796 0.808]
print(round(scores.mean(), 3), round(scores.std(), 3))    # 0.802 0.007

# share of churners in each validation fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(np.round([y.iloc[va].mean() for _, va in kf.split(X, y)], 3))    # [0.261 0.269 0.289 0.262 0.246]
print(np.round([y.iloc[va].mean() for _, va in skf.split(X, y)], 3))   # [0.265 0.265 0.265 0.265 0.266]

# several metrics at once, with training scores
res = cross_validate(model, X, y, cv=skf, scoring=["accuracy", "roc_auc", "f1"], return_train_score=True)
for m in ["accuracy", "roc_auc", "f1"]:
    print(m, round(res[f"train_{m}"].mean(), 3), round(res[f"test_{m}"].mean(), 3))
# accuracy 0.807 0.803
# roc_auc  0.849 0.845
# f1       0.603 0.596   (training and validation close: no sign of overfitting)
```

> [!NOTE]
> In scikit-learn the validation scores of `cross_validate` are called `test_...`. They are validation scores in the sense of this page: they come from the development data, not from a locked test set.

**In practice.**
- scikit-learn uses 5-fold cross-validation (stratified for classifiers) as the default in `cross_val_score`, `GridSearchCV` and related tools.
- In genomics and medical imaging, where studies often have only a few hundred patients, k-fold cross-validation is the usual way to report performance; ISLP (James et al., 2023, Ch. 5) recommends *k* = 5 or 10 as a good bias–variance compromise.
- Credit scoring and churn teams typically compare candidate models by cross-validated ROC AUC before a final out-of-time test.

> [!WARNING]
> The *k* models of cross-validation are thrown away. Cross-validation estimates the performance of a *procedure* (this model type with these settings). To use the model, refit it on all development data.

> [!CAUTION]
> Use `shuffle=True` in `KFold` when the rows are sorted (for example by date or by class). Without shuffling, the first fold contains only the oldest rows or only one class. If rows are sorted *because time matters*, do not shuffle: use a time-series split (next section).

## Grouped and time-series splits

**Concept.** k-fold assumes that rows are independent and that the future looks like a random sample of the past. Two common situations break this.

1. **Groups.** Several rows can belong to the same unit: several visits of the same patient, several transactions of the same customer, several Airbnb listings of the same host. If a group appears in both training and validation folds, the model can recognise the group instead of learning a general pattern. **`GroupKFold`** (or `GroupShuffleSplit` for a single split) assigns whole groups to folds, so no group appears on both sides. It needs a `groups` array, here the host identifier.
2. **Time.** If the model will predict the future, a random split lets it train on rows recorded *after* the ones it is validated on. This hides changes over time (the COVID break in Berlin's Airbnb market, a new registration rule, new kinds of offers), which are called **drift**. **`TimeSeriesSplit`** needs rows sorted by time. Split *i* trains on the first part and validates on the block that follows; the training window grows with each split (right panel of the figure above). A single **out-of-time** split (train on the older years, validate on the most recent one) is the simplest version. Session 12 uses this scheme to backtest the demand forecast, and Session 13 compares random and time-based validation for the customs text classifier of the leaderboard.

```mermaid
flowchart TD
    Q1{"Will the model be used<br/>on new groups?<br/>(new hosts, patients)"} -->|yes| G["GroupKFold"]
    Q1 -->|no| Q2{"Will it predict<br/>the future?"}
    Q2 -->|yes| TS["TimeSeriesSplit or<br/>one out-of-time split"]
    Q2 -->|no| Q3{"Classification?"}
    Q3 -->|yes| S["StratifiedKFold"]
    Q3 -->|no| K["KFold (shuffle=True)"]
```

**Why it matters.** The splitter must reproduce the situation in which the model will be used. Ordinary k-fold measures performance on groups and periods already seen; the real use often involves new ones: a price suggestion is most useful for a host who has never listed before, and a forecast is always about next month.

**How it works in Python: groups.** A price model for Berlin Airbnb listings: predict the log of the nightly price from size, location, room type and reviews with gradient boosting (Session 10; here a black box). The table is the short-stay table of Sessions 4–6 (6,701 listings with a price and a minimum stay below 28 nights) without the 26 listings priced below €10 or above €1,000, which Session 4 treats as outliers. Listings with a minimum stay of 28 nights or more are medium-term rentals with a different price basis (median €23 a night in this snapshot) and belong to another model. Many hosts offer several similar flats, often in the same building, with similar prices.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

lst = pd.read_parquet("case-study/data/airbnb/listings.parquet")
bnb = lst[(lst["minimum_nights"] < 28) & lst["price"].between(10, 1000)].reset_index(drop=True)
bnb["dist_km"] = np.hypot((bnb["latitude"] - 52.5219) * 111.2,        # km to Alexanderplatz
                          (bnb["longitude"] - 13.4132) * 68.0)
cols = ["accommodates", "bedrooms", "beds", "bathrooms", "dist_km", "room_type", "district",
        "number_of_reviews", "review_scores_rating", "availability_365"]
X_bnb = pd.get_dummies(bnb[cols], columns=["room_type", "district"], dtype=int)
y_bnb = np.log(bnb["price"])                                         # log of the nightly price in EUR
hosts = bnb["host_id"]
n_per_host = hosts.map(hosts.value_counts())
print(len(bnb), hosts.nunique(), round((n_per_host > 1).mean(), 3), n_per_host.max())
# 6675 3737 0.565 83   <- listings, hosts, share of listings whose host has several, largest host

gbm = HistGradientBoostingRegressor(random_state=0)
random_cv = cross_val_score(gbm, X_bnb, y_bnb, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
grouped_cv = cross_val_score(gbm, X_bnb, y_bnb, cv=GroupKFold(5), groups=hosts, scoring="r2")
print(random_cv.mean().round(3), random_cv.std().round(3))       # 0.636 0.033
print(grouped_cv.mean().round(3), grouped_cv.std().round(3))     # 0.583 0.022
```

Random 5-fold cross-validation reports R² = 0.64; folds that keep each host on one side report 0.58. The difference is larger than the fold-to-fold spread, so it is not noise. In a random split, 55 % of the validation listings have a host whose other listings sit in the training folds, and for hosts with several listings the host's average alone explains 70 % of the variance of the log price. The model partly recognises "a flat of this host" instead of learning what size and location are worth. The effect depends on the model: the small linear model of Session 6 (guests, room type, distance, district) scores 0.513 with random and 0.502 with grouped folds, because four features leave little room to memorise hosts; the more flexible a model and the richer its features, the larger the gap (theory page 02 and the workbook). Which number is right depends on the use. A price suggestion for a **new host** should be judged with `GroupKFold`; a tool that only prices further flats of hosts already in the data may use the random estimate. The [Airbnb validation workbook](../workbooks/20-case-study-airbnb-price-validation.ipynb) repeats the comparison for a linear model and a random forest.

> [!NOTE]
> Inside Airbnb collects these data from public listing pages (CC BY 4.0; snapshot of 26 June 2026). The course copy has no host names. Use `host_id` only as a grouping key and report results in aggregate; never look up or name individual hosts.

**How it works in Python: time.** `TimeSeriesSplit(5)` on rows sorted by date returns five splits in which every training index lies before every validation index. The listings snapshot is a single day and has no time order to respect, so this page does not use it; the [splitter workbook](../workbooks/03-cross-validation-splitters.ipynb) shows the indices, and Session 12 applies the scheme to monthly demand.

**In practice.**
- Medical imaging: the CheXNet study (Rajpurkar et al., 2017) split the ChestX-ray14 images by patient so that no patient appeared in both training and test data; otherwise a model can recognise the patient rather than the disease.
- Credit risk models in banks are routinely validated out-of-time: trained on older loans and tested on more recent ones.
- Demand and energy-load forecasting use rolling-origin evaluation, the forecasting name for the same scheme (Session 12).

> [!WARNING]
> `TimeSeriesSplit` uses the row order. Sort by date first (`sort_values("date", ignore_index=True)`); otherwise the "past" and "future" are arbitrary.

> [!CAUTION]
> Grouping and time can both apply. A model for next year's prices of *new* hosts would need both: hosts kept together and the latest period held out. Decide which situation matters for your project and say so in the validation plan.

## Bootstrap confidence interval of a metric

**Concept.** The **bootstrap** (Efron, 1979; recap in Session 5) estimates the uncertainty of a statistic from one sample: draw *n* rows **with replacement** from the *n* rows, recompute the statistic, and repeat about 1,000 times. The spread of the recomputed values approximates the sampling variability. For a model, the predictions stay fixed and only the test rows are resampled. The 2.5 % and 97.5 % percentiles of the bootstrap values form a 95 % **percentile confidence interval** for the test metric.

A tiny example by hand. Five test listings with absolute price errors of €10, €20, €30, €40 and €100: the mean absolute error (MAE) is €40. One bootstrap sample draws positions 2, 2, 3, 5, 1 → €20, €20, €30, €100, €10 → MAE €36; another draws 5, 5, 4, 1, 5 → €100, €100, €40, €10, €100 → €70. Repeating this many times gives a distribution of MAEs; with only five rows it is very wide, which is the honest answer.

**Why it matters.** A score without an interval cannot tell whether a difference between two models, or between a validation score and a later test score, is larger than chance. The bootstrap works for any metric, including MAE in euros, macro-F1 and ROC AUC, for which no simple formula exists.

**How it works in Python.** The question of the practice: how far off would the price model be for a host who lists for the first time? Lock away 20 % of the hosts with all their listings, fit the gradient-boosting model on the other hosts, and compute the MAE in euros on the locked hosts. Then bootstrap it twice: once resampling listings, once resampling **hosts** (a **cluster bootstrap**, because listings of the same host are not independent).

```python
from sklearn.model_selection import GroupShuffleSplit

dev, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X_bnb, y_bnb, groups=hosts))
gbm.fit(X_bnb.iloc[dev], y_bnb.iloc[dev])                      # trained on 80 % of the hosts
price = bnb["price"].iloc[test].to_numpy()
abs_err = np.abs(price - np.exp(gbm.predict(X_bnb.iloc[test])))  # error in euros per listing
print(len(test), hosts.iloc[test].nunique(), round(abs_err.mean(), 1), np.median(price))
# 1254 748 50.8 160.0   <- test listings, test hosts, MAE in EUR, median price of the test listings

rng = np.random.default_rng(0)
boot_rows = [abs_err[rng.integers(0, len(abs_err), len(abs_err))].mean() for _ in range(1000)]
by_host = pd.DataFrame({"host": hosts.iloc[test].to_numpy(), "err": abs_err}).groupby("host")["err"]
sums, counts = by_host.sum().to_numpy(), by_host.count().to_numpy()
boot_hosts = []
for _ in range(1000):                                           # draw hosts with replacement
    pick = rng.integers(0, len(sums), len(sums))
    boot_hosts.append(sums[pick].sum() / counts[pick].sum())
print(np.percentile(boot_rows, [2.5, 97.5]).round(1))           # [47.4 54.8]  listings resampled
print(np.percentile(boot_hosts, [2.5, 97.5]).round(1))          # [45.5 56.5]  hosts resampled
```

For a new host the model is off by about €51 a night on average, with a 95 % interval of roughly €46 to €57: about a third of the median price of €160. The interval that resamples listings is narrower (±€3.7 against ±€5.5) because it treats 1,254 listings as independent although they come from only 748 hosts, and a host's listings tend to be all well or all badly priced by the model. A model that reaches €49 instead of €51 on this test set is not demonstrably better.

> [!NOTE]
> Two related summaries are easy to confuse. The **standard deviation across CV folds** describes how much the score varies between training sets and validation folds. The **bootstrap interval on a test set** describes the uncertainty from the finite test sample for one fitted model. Both are useful; say which one you report.

**In practice.**
- Diagnostic accuracy studies in medicine report sensitivity, specificity and AUC with 95 % confidence intervals; bootstrap intervals are common for AUC.
- Machine translation research uses paired bootstrap resampling (Koehn, 2004) to test whether one system's BLEU score is better than another's on the same test set.

> [!WARNING]
> Resample the *test rows*, not the training rows, when you want the uncertainty of a test score. Refitting the model inside every bootstrap round answers a different question (the variability of the training procedure) and is far slower.

> [!CAUTION]
> The bootstrap assumes that the test rows are independent. If they come in groups (listings of one host, visits of one patient, purchases of one customer), resample whole groups as above, otherwise the interval is too narrow.

## Check your understanding

1. You tried 30 models and picked the one with the best validation score. Why is that validation score an optimistic estimate, and what number should you report instead?
2. A 5-fold cross-validation gives accuracies 0.81, 0.79, 0.80, 0.82, 0.78. A colleague's new model gets 0.805 on one split. Is it better? What would you ask for?
3. For each case, choose a splitter and justify it: (a) predicting next month's churn; (b) next month's number of Airbnb reviews in Berlin; (c) classifying 500 tumour images from 120 patients; (d) a price suggestion for people who list their first flat on Airbnb.
4. `StratifiedKFold` needs at least *k* rows of every class. What would you do with a classification target whose rarest class has only three rows?
5. Describe in three steps how to compute a 95 % cluster-bootstrap interval for the MAE of the price model on held-out hosts.
6. Random cross-validation of the Berlin price model gives R² = 0.64, host-grouped cross-validation 0.58. Explain the gap in two sentences.

## Further reading

- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, Chapter 5 "Resampling Methods". Springer. https://www.statlearning.com/
- scikit-learn developers (2025). *Cross-validation: evaluating estimator performance*. User guide. https://scikit-learn.org/stable/modules/cross_validation.html
- Raschka, S. (2018). *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning*. arXiv:1811.12808. https://arxiv.org/abs/1811.12808
- INRIA (2024). *scikit-learn MOOC, Module 2: Selecting the best model* (cross-validation framework). https://inria.github.io/scikit-learn-mooc/
