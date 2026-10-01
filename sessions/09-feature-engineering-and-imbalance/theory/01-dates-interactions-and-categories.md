# Features from dates, interactions and categories

A **feature** is one input column of a model. Feature engineering means constructing new input columns from the raw data so that a model can use information it could not see before. This page covers the first block of Session 9: features from dates and times, interactions between features, categories with many levels (rare-category grouping and target encoding) and custom transformers that put all of this into a scikit-learn pipeline. The running example is the review table of the case study. Session 8 introduced one-hot encoding, scaling and the `ColumnTransformer`; this page builds on them.

> [!NOTE]
> The code blocks on this page build on each other. Run them in order from the repository root, for example in a notebook started with `uv run jupyter lab`. They use `case-study/data/train_sample.parquet` (50,000 reviews) and `case-study/data/products.parquet`.

```mermaid
flowchart LR
    raw["Raw columns<br/>date, store, text"] --> d["Date parts<br/>year, month, sin/cos"]
    raw --> i["Interactions<br/>a × b"]
    raw --> c["Categories<br/>rare grouping,<br/>target encoding"]
    d --> p["Pipeline<br/>fit on training data only"]
    i --> p
    c --> p
    p --> m["Model"]
```

## 1. Features from dates and times

### Concept

A timestamp such as `2019-12-24 18:05` is one value, but it carries several kinds of information: the **year** (long-term trend), the **month** and **day of the week** (seasonal patterns), the **hour** (daily rhythm), and the **time elapsed** since some event (age of an account, days since a product's first review). A model cannot extract these parts on its own; we compute them as separate columns.

Some parts are **cyclical**: month 12 is followed by month 1, and hour 23 by hour 0. If we use the month as a number from 1 to 12, a linear model sees December and January as far apart. A **cyclical encoding** places each value on a circle with two columns:

- `month_sin = sin(2π · month / 12)`
- `month_cos = cos(2π · month / 12)`

Worked example: for January, 2π · 1/12 = 0.52 rad, so sin = 0.50 and cos = 0.87. For December, 2π · 12/12 = 2π, so sin = 0.00 and cos = 1.00. The two points are close on the circle, as they should be. Tree-based models (Session 10) can split a plain month number into ranges and usually do not need the sine and cosine; linear models and k-nearest neighbours do.

![Months on the unit circle with their sine and cosine coordinates; December and January are neighbours](figures/cyclical_month_encoding.png)

### Why it matters

Customer behaviour follows calendars: sales peak before holidays, call centres have busy Mondays, and online reviews have become more critical over the years. A raw timestamp, stored as nanoseconds since 1970, mixes all these effects into one large number. Without date features a model either ignores time or picks up the trend in an uncontrolled way.

### How it works in Python

```python
import numpy as np
import pandas as pd

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
d = reviews["date"]                                   # datetime64 column
dates = pd.DataFrame({
    "year": d.dt.year,
    "month": d.dt.month,
    "dayofweek": d.dt.dayofweek,                      # Monday = 0, Sunday = 6
    "hour": d.dt.hour,
    "is_weekend": d.dt.dayofweek.ge(5).astype(int),
    "days_since_2000": (d - pd.Timestamp("2000-01-01")).dt.days,   # elapsed time, a trend feature
})
dates["month_sin"] = np.sin(2 * np.pi * dates["month"] / 12)      # cyclical encoding
dates["month_cos"] = np.cos(2 * np.pi * dates["month"] / 12)
print(dates.round(2).head(3).to_string())
#    year  month  dayofweek  hour  is_weekend  days_since_2000  month_sin  month_cos
# 0  2001      9          2     0           0              613      -1.00       -0.0
# 1  2003      8          0    20           0             1311      -0.87       -0.5
# 2  2003      9          1    14           0             1347      -1.00       -0.0

# Is there a trend? Share of negative reviews per year
share_neg = reviews["label"].eq("neg").groupby(dates["year"]).mean()
print(share_neg.loc[2015:2021].round(3).to_dict())
# {2015: 0.142, 2016: 0.153, 2017: 0.187, 2018: 0.186, 2019: 0.19, 2020: 0.224, 2021: 0.235}
```

The share of negative reviews rose from 14 % in 2015 to 24 % in 2021, so `year` carries information. The day of the week and the hour hardly change the share of negative reviews in this sample (between 17 % and 22 %); they are weak features here. Checking such simple group means before adding a feature is good practice.

scikit-learn offers the same idea inside a pipeline: a `FunctionTransformer` (Section 5) can compute the sine and cosine, and `SplineTransformer(extrapolation="periodic")` builds smooth periodic features. The workbook [01-time-related-feature-engineering.ipynb](../workbooks/01-time-related-feature-engineering.ipynb) compares these encodings on a bike-sharing demand dataset.

### In practice

- In the Kaggle *Rossmann Store Sales* competition (2015), forecasts of daily sales for about 1,100 German drugstores relied on calendar features such as day of week, month, holidays and the days until and since promotions.
- The M5 forecasting competition on Walmart sales (Makridakis et al., 2022) provided calendar data with events and the days on which food-stamp (SNAP) payments are made, because these dates shift demand.
- The scikit-learn example on bike-sharing demand in Washington, D.C. shows that hour-of-day features with a periodic encoding reduce the error of a linear model considerably.

> [!WARNING]
> **Time zones and the future.** Timestamps in the case study are in UTC; "evening" in UTC is the afternoon in California. Convert to local time (`dt.tz_localize("UTC").dt.tz_convert(...)`) before building hour features about human behaviour. And never use a date feature that is only known later, such as "date of the last review of this product": at prediction time it does not exist yet.

> [!CAUTION]
> A `year` feature lets a model learn a trend, but tree-based models cannot extrapolate it: for the test years 2022 and 2023 they treat every review like one from 2021, the last year they saw. That is often acceptable, but be aware of it when the target drifts (Session 16).

## 2. Interactions between features

### Concept

An **interaction** exists when the effect of one feature depends on the value of another. A model with only additive terms, such as a linear or logistic regression, assumes that each feature adds its own effect regardless of the others. We can give it an interaction by adding the **product** of two features as a new column.

Worked example: a shop records whether a customer is a member and whether a discount was shown. The purchase rates are:

| | no discount | discount |
|---|---|---|
| not a member | 0.17 | 0.19 |
| member | 0.20 | 0.72 |

The discount barely helps non-members (+0.02) but helps members a lot (+0.52). An additive model can only fit one "discount effect" for everybody. With the extra column `member × discount`, which is 1 only in the lower-right cell, the model can fit all four cells.

### Why it matters

Linear models are popular because they are fast and interpretable, but they miss interactions unless we add them. Tree-based models find interactions automatically, because a split on one feature followed by a split on another is an interaction. Adding the right interactions can bring a linear model close to a tree model, while keeping it explainable.

### How it works in Python

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
n = 2000
toy = pd.DataFrame({"member": rng.integers(0, 2, n), "discount": rng.integers(0, 2, n)})
# true purchase probability: 0.7 for members with a discount, 0.2 for everybody else
toy["buy"] = rng.binomial(1, np.where((toy.member == 1) & (toy.discount == 1), 0.7, 0.2))

cells = pd.DataFrame({"member": [0, 0, 1, 1], "discount": [0, 1, 0, 1]})
plain = LogisticRegression().fit(toy[["member", "discount"]], toy["buy"])
inter = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                      LogisticRegression()).fit(toy[["member", "discount"]], toy["buy"])
cells["observed"] = toy.groupby(["member", "discount"])["buy"].mean().to_numpy()
cells["plain"] = plain.predict_proba(cells[["member", "discount"]])[:, 1]
cells["interaction"] = inter.predict_proba(cells[["member", "discount"]])[:, 1]
print(cells.round(2).to_string(index=False))
#  member  discount  observed  plain  interaction
#       0         0      0.17   0.09         0.17
#       0         1      0.19   0.29         0.20
#       1         0      0.20   0.28         0.20
#       1         1      0.72   0.63         0.71
print(inter[0].get_feature_names_out())   # ['member' 'discount' 'member discount']
```

The additive model is wrong in every cell: it underestimates the members with a discount and overestimates the others. With the product column the predictions match the observed rates.

On the reviews, candidate interactions are, for example, `verified_purchase × text length` or `number of exclamation marks × share of capital letters`. Domain knowledge suggests which products are worth trying; `PolynomialFeatures(interaction_only=True)` creates all pairwise products, which grows quickly (p features give p·(p−1)/2 products).

### In practice

- Google's *Wide & Deep* model for app recommendations in Google Play (Cheng et al., 2016) combined a linear model on hand-made cross-product features, such as "installed app × shown app", with a neural network.
- Factorization machines (Rendle, 2010) were designed to estimate pairwise interactions in very sparse data such as click logs, and became a standard method for click-through-rate prediction.

> [!WARNING]
> **Too many interactions.** With 50 features, all pairwise products give 1,225 new columns, most of them noise. The model then overfits (Session 6). Add interactions that you can justify, or use a model that finds them itself (Session 10). Scale the features before multiplying them, or the products have very different ranges.

## 3. High-cardinality categories: grouping rare categories

### Concept

The **cardinality** of a categorical feature is its number of distinct values. `verified_purchase` has 2; `store` (the brand or seller of a product) has 11,672 distinct values in the 50,000-review sample. One-hot encoding (Session 8) would create one column per store. Most of these columns would contain a single 1, because half of all stores appear only once.

**Grouping rare categories** replaces every category that appears fewer than *k* times in the training data by one shared level, often called `"infrequent"` or `"other"`. The frequent categories keep their own column. Categories that appear only in new data are mapped to the same shared level.

### Why it matters

A column with one 1 in 50,000 rows cannot teach a model anything reliable, but it costs memory and invites overfitting. Grouping keeps the information of the frequent categories, gives the model a stable estimate for "small store", and handles unseen categories at prediction time without errors. Frequency itself can be a feature: the number of reviews of a store says something about its size.

### How it works in Python

```python
from sklearn.preprocessing import OneHotEncoder

products = pd.read_parquet("case-study/data/products.parquet", columns=["parent_asin", "store"])
df = reviews.merge(products, on="parent_asin", how="left")      # join the store of each product
df["store"] = df["store"].fillna("missing")
counts = df["store"].value_counts()
print(df["store"].nunique(), (counts == 1).sum())                # 11673 levels (with "missing"), 5805 seen once

for k in [None, 20, 100]:
    enc = OneHotEncoder(min_frequency=k, handle_unknown="infrequent_if_exist")
    n_cols = enc.fit(df[["store"]]).transform(df[["store"]]).shape[1]
    print(k, n_cols)
# None 11673   one column per store
# 20 364       stores with fewer than 20 reviews share one column
# 100 38
```

`min_frequency` sets the threshold *k* (an integer count, or a fraction of rows); `max_categories` keeps only the most frequent levels. With `handle_unknown="infrequent_if_exist"` a store that appears for the first time in the test data is encoded like a rare store. A **count encoding** (`df["store"].map(counts)`) adds the size of the store as one numeric feature; it must also be computed on the training data only.

### In practice

- Retail and e-commerce data have product, brand and seller identifiers with thousands of levels; grouping the long tail is a routine step before modelling.
- In German official statistics, the Federal Statistical Office (Destatis) publishes many tables only with grouped categories so that rare groups do not reveal individuals; the statistical reason (too few observations per cell) is the same.

> [!WARNING]
> The threshold is a hyperparameter. Choose it with cross-validation (Session 7), not by looking at the test data. And compute the counts inside the pipeline: an encoder that was fitted on all data has already seen the test rows.

## 4. Target encoding and its leakage risk

### Concept

**Target encoding** (also called mean encoding) replaces each category by the mean of the target in the training rows of that category. For a binary target "review is negative", the store "B" with 10 negative reviews out of 50 gets the value 10/50 = 0.20. One numeric column replaces thousands of dummy columns, and it orders the categories by their relation to the target.

Two problems arise.

1. **Small categories give noisy means.** A store with 2 reviews, both negative, gets 1.0. **Smoothing** shrinks the mean of a small category towards the overall mean ȳ:

   encoding = (n · ȳ_category + m · ȳ) / (n + m)

   With an overall share of negative reviews ȳ = 0.19 and *m* = 10: store A (n = 2, mean 1.0) gets (2 · 1.0 + 10 · 0.19)/12 = 0.33; store B (n = 50, mean 0.20) gets (50 · 0.20 + 10 · 0.19)/60 = 0.20. The large store keeps its value; the small one moves towards the average.

2. **Target leakage.** **Leakage** means that information that will not be available at prediction time enters the training data. If a row's own label is part of the mean that encodes it, the feature partly *is* the label. For a category with one row, the encoding equals the label exactly. A model learns to trust the feature, and its training score is far too optimistic.

The remedy is **cross-fitting**: split the training data into *k* folds; encode the rows of each fold with means computed from the other *k*−1 folds only. No row ever sees its own label. scikit-learn's `TargetEncoder` does this automatically in `fit_transform` (5 folds by default) and uses smoothing (`smooth="auto"`, an empirical Bayes estimate of *m*). For new data, `transform` uses means from the whole training set.

```mermaid
flowchart TB
    subgraph naive["Naive encoding"]
        n1["Mean per store<br/>from ALL training rows"] --> n2["Row i encoded with<br/>a mean that includes y_i"]
        n2 --> n3["Training score too high"]
    end
    subgraph cross["Cross-fitting (TargetEncoder)"]
        c1["Split training rows<br/>into 5 folds"] --> c2["Encode fold 1 with means<br/>of folds 2-5, and so on"]
        c2 --> c3["No row sees its own label"]
    end
```

### Why it matters

Target encoding is often the best way to use a high-cardinality category in linear and tree models; benchmarks such as Pargent et al. (2022) found regularised target encoding among the most reliable encoders. But the naive version is one of the most common sources of leakage in practice, and the damage is invisible on the training data.

### How it works in Python

The hashed `user_id` makes the leak visible: 49,494 of the 50,000 sample reviews have their own user. A naive encoding of `user_id` therefore copies the label.

```python
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import TargetEncoder

y = df["label"].eq("neg").astype(int)                           # binary target: negative review
tr, te = train_test_split(df.index, test_size=0.3, random_state=0, stratify=y)

for col in ["user_id", "store"]:
    naive = y[tr].groupby(df.loc[tr, col]).mean()               # mean target per category, own row included
    f_tr = df.loc[tr, col].map(naive)
    f_te = df.loc[te, col].map(naive).fillna(y[tr].mean())      # unseen categories: overall mean
    enc = TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=0))
    e_tr = enc.fit_transform(df.loc[tr, [col]], y[tr]).ravel()  # cross-fitted on the training rows
    e_te = enc.transform(df.loc[te, [col]]).ravel()
    print(col, "naive:", round(roc_auc_score(y[tr], f_tr), 3), round(roc_auc_score(y[te], f_te), 3),
          "| cross-fitted:", round(roc_auc_score(y[tr], e_tr), 3), round(roc_auc_score(y[te], e_te), 3))
# user_id naive: 1.0 0.505 | cross-fitted: 0.494 0.505
# store naive: 0.881 0.619 | cross-fitted: 0.622 0.62
```

Read the numbers as "training AUC, test AUC". The naive encoding of `user_id` separates negative from other reviews perfectly on the training rows (AUC 1.0) and is useless on new rows (0.505, chance level). The cross-fitted version is honest: its training AUC already shows that the feature carries no information. For `store`, the naive training AUC of 0.881 promises much more than the 0.619 that the feature delivers on new data; the cross-fitted training AUC (0.622) is a fair estimate.

For the three-class review label, `TargetEncoder(target_type="multiclass")` creates one column per class (`store_neg`, `store_neu`, `store_pos`), each the smoothed share of that class. The workbooks [03-target-encoder.ipynb](../workbooks/03-target-encoder.ipynb) and [04-target-encoder-cross-fitting.ipynb](../workbooks/04-target-encoder-cross-fitting.ipynb) show both points on a wine-review dataset.

### In practice

- Micci-Barreca (2001) proposed smoothed target statistics for high-cardinality attributes such as ZIP codes and IP addresses in fraud-detection models.
- CatBoost, the gradient boosting library from Yandex (Session 10), encodes categories with *ordered target statistics*: each row is encoded only with the labels of rows that come before it in a random order, a variant of the same idea (Prokhorenkova et al., 2018).

> [!CAUTION]
> `TargetEncoder.fit(X, y).transform(X)` is **not** the same as `fit_transform(X, y)`. The first encodes the training rows with means that include their own labels (the naive leak); only `fit_transform` cross-fits. Inside a `Pipeline`, scikit-learn calls `fit_transform` during training and `transform` during prediction, which is correct.

> [!WARNING]
> Cross-fitting protects against the row's own label, not against the future. If the categories' target means change over time, encode with past data only (page 2) and validate with a time-based split (Session 7).

## 5. Custom transformers in scikit-learn pipelines

### Concept

A **transformer** in scikit-learn is an object with two methods: `fit(X, y)` learns what it needs from the training data, and `transform(X)` applies it to any data. `StandardScaler` learns means and standard deviations; `OneHotEncoder` learns the list of categories. When no built-in transformer does what we need, we write our own in one of two ways:

- `FunctionTransformer(func)` wraps a function that needs nothing from the training data, such as "compute the text length" or "take the sine of the month". Its `fit` does nothing.
- A small class that inherits from `BaseEstimator` and `TransformerMixin` is needed when something must be learned in `fit`, such as the list of frequent stores.

Put inside a `Pipeline` or `ColumnTransformer`, both are fitted on the training folds only, exactly like the built-in ones.

```mermaid
classDiagram
    class TransformerMixin {
        +fit_transform(X, y)
    }
    class BaseEstimator {
        +get_params()
        +set_params()
    }
    class RareGrouper {
        +min_count
        +frequent_
        +fit(X, y)
        +transform(X)
    }
    TransformerMixin <|-- RareGrouper
    BaseEstimator <|-- RareGrouper
```

### Why it matters

Feature code written outside the pipeline, for example in a notebook cell that modifies the whole data frame, is easily applied to training and test data together, and is easily forgotten when the model is deployed. Inside the pipeline the same code runs during cross-validation, on the test set and in production (Session 16), and it is saved together with the model.

### How it works in Python

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def date_features(X):
    """Year and cyclical month of a data frame with one datetime column 'date'."""
    month = X["date"].dt.month
    return pd.DataFrame({"year": X["date"].dt.year,
                         "month_sin": np.sin(2 * np.pi * month / 12),
                         "month_cos": np.cos(2 * np.pi * month / 12)}, index=X.index)


class RareGrouper(BaseEstimator, TransformerMixin):
    """Replace categories seen fewer than min_count times in the training data by 'other'."""

    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, X, y=None):
        counts = X.iloc[:, 0].value_counts()
        self.frequent_ = set(counts[counts >= self.min_count].index)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)   # learned from training data
        return self

    def transform(self, X):
        col = X.iloc[:, 0]
        return col.where(col.isin(self.frequent_), "other").to_frame()

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_                 # one output column per input column


features = ColumnTransformer([
    ("dates", FunctionTransformer(date_features,         # names of the three output columns:
                                  feature_names_out=lambda tf, names: ["year", "month_sin", "month_cos"]),
     ["date"]),
    ("store", make_pipeline(RareGrouper(min_count=20),
                            TargetEncoder(target_type="multiclass",
                                          cv=KFold(5, shuffle=True, random_state=0))), ["store"]),
    ("meta", "passthrough", ["helpful_vote", "n_images"]),
])
model = make_pipeline(features, HistGradientBoostingClassifier(random_state=0, class_weight="balanced"))
X = df[["date", "store", "helpful_vote", "n_images"]]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, df["label"], cv=cv, scoring="f1_macro")
print(scores.mean().round(3))                                # 0.337
print(model.fit(X, df["label"])[0].get_feature_names_out()[:4])
# ['dates__year' 'dates__month_sin' 'dates__month_cos' 'store__store_neg']
```

Everything that learns from data (the list of frequent stores, the target means) is fitted again inside each cross-validation fold. The cross-validated macro-F1 of 0.34 is only a little above the 0.28 of always predicting "positive", because the model sees no text; text statistics follow on page 2. The mechanism matters here, not the score.

### In practice

- Feature-engine (Galli, 2021), an open-source library under the BSD licence, provides transformers for rare-label grouping, date features and cyclical encoding with the same `fit`/`transform` interface; it shows how far the pattern carries.
- Deployed models at many companies are stored as one pipeline object (for example with `joblib`, Session 16), so that the serving code calls only `predict` and cannot apply a different feature recipe by mistake.

> [!TIP]
> Name the learned attributes with a trailing underscore (`frequent_`), set them only in `fit`, and store constructor arguments unchanged in `__init__`. Then `clone`, `GridSearchCV` and `get_params` work with your class as with any built-in transformer.

> [!WARNING]
> A `FunctionTransformer` must not compute anything across rows, such as `x - x.mean()`: that mean would come from whichever data are passed in, the test set included. Anything that needs statistics of the data belongs in `fit`.

## Practice

Build date, interaction and target-encoded store features for the reviews in [05-case-study-review-features.ipynb](../workbooks/05-case-study-review-features.ipynb): add year and cyclical month features, one interaction of your choice, the store grouped and target-encoded with cross-fitting, and compare the cross-validated macro-F1 with and without each group of features.

## Check your understanding

1. Why does a logistic regression need a cyclical encoding of the month, while a decision tree usually does not?
2. In the member-discount table, what single extra column lets an additive model fit all four cells, and what values does it take?
3. A store has 3 reviews, all negative; the overall share of negative reviews is 0.19. What is its smoothed target encoding with *m* = 10?
4. Why does a naive target encoding of `user_id` give a training AUC of 1.0 but a test AUC of 0.5?
5. When do you need a class that inherits from `BaseEstimator` and `TransformerMixin` instead of a `FunctionTransformer`?

## Further reading

- scikit-learn developers (2025). *Target Encoder's internal cross fitting*. scikit-learn example gallery. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
- scikit-learn developers (2025). *Time-related feature engineering*. scikit-learn example gallery. https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- Pargent, F., Pfisterer, F., Thomas, J. & Bischl, B. (2022). Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features. *Computational Statistics*, 37, 2671–2692. https://doi.org/10.1007/s00180-022-01207-6
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press. Free online: https://bookdown.org/max/FES/
