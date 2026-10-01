# The ML lifecycle and linear regression

This page opens the machine-learning part of the course. It places every later session in the **machine-learning lifecycle**, a sequence of ten steps from problem definition to monitoring, and introduces the vocabulary of supervised learning: features, target, training and prediction. Then it covers the first model, **linear regression**: the split into training and test data, simple and multiple regression fitted by least squares, residuals, and the three standard error metrics MAE, RMSE and R². The running example predicts the number of helpful votes a review receives.

## The ML lifecycle in ten steps

### Concept

A machine-learning project is more than fitting a model. The **lifecycle** describes the steps from a business question to a model that runs and is maintained in production:

| Step | Question | Course sessions |
|---|---|---|
| 01 Problem definition | What do we predict, for whom, measured how? What is the baseline? | S4, S6 |
| 02 Data collection | Which data exist, and may we use them? | S2, S3 |
| 03 Data cleaning and preprocessing | Are the data correct, complete, consistent? | S4 |
| 04 Exploratory data analysis | What is in the data; which relationships are real? | S5 |
| 05 Feature engineering and selection | Which inputs does the model get? | S8, S9, S11, S13, S14 |
| 06 Model selection | Which family of models fits the problem? | S7–S10 |
| 07 Model training | Fit the parameters on training data. | S6, S8, S10–S15 |
| 08 Model evaluation and tuning | How good is it on new data; which settings are best? | S6–S8, S10–S15 |
| 09 Model deployment | Make predictions available to users or systems. | S16 |
| 10 Model monitoring and maintenance | Does it still work as the world changes? | S16 |

```mermaid
flowchart LR
    P["01 Problem<br/>definition"] --> C["02 Data<br/>collection"] --> D["03 Cleaning"] --> E["04 EDA"] --> F["05 Features"]
    F --> S["06 Model<br/>selection"] --> T["07 Training"] --> V["08 Evaluation<br/>and tuning"] --> Y["09 Deployment"] --> M["10 Monitoring"]
    V -.->|"not good enough"| F
    V -.->|"wrong target or metric"| P
    M -.->|"drift: retrain"| C
    M -.->|"new question"| P
```

The arrows back are the point of the diagram: projects loop. An evaluation that fails sends you back to features or data; monitoring that detects a change sends you back to data collection and retraining.

**Problem definition** fixes three things: the **target** (what is predicted), the **metric** (how success is measured) and the **baseline** (the simplest prediction the model must beat). For the course leaderboard (Session 8 onwards): target = sentiment label (neg/neu/pos), metric = macro-F1, baseline = always "positive" (macro-F1 0.26).

### Why it matters

Most failed ML projects fail outside the training step: a target that does not match the business decision, data that are not available at prediction time, or a model nobody maintains. Naming the steps makes these risks visible early. Sessions 3–5 already covered steps 02–04 for the review data.

### How it works in Python

Step 01 in code: state the target, the metric and the baseline before any model.

```python
import numpy as np
import pandas as pd

reviews = pd.read_parquet("case-study/data/train_sample.parquet")

# 01 problem definition for this page: predict log(1 + helpful votes) of a review
target = np.log1p(reviews["helpful_vote"])
metric = "MAE"                                     # mean absolute error, in log units
baseline = np.full(len(target), target.mean())     # predict the mean for every review
print(round(np.mean(np.abs(target - baseline)), 3))   # 0.486: every model must beat this
```

### In practice

- CRISP-DM (1999), developed by a consortium including Daimler-Benz, SPSS and NCR, is the classic six-phase process model for data mining; most lifecycle diagrams descend from it.
- Google's "Rules of Machine Learning" (Zinkevich) advises starting with a simple baseline and a well-defined metric before investing in complex models.
- Sculley et al. (2015) at Google described the "hidden technical debt" of ML systems: the model code is a small part of a system dominated by data collection, feature extraction, serving and monitoring.

> [!IMPORTANT]
> **Practice (block 1, part 1).** Map the review-sentiment task of the leaderboard to the ten steps: for each step, write one sentence on what it means for this task and which session covers it.

## Supervised learning: features, target, training, prediction

### Concept

- An **observation** (row, example) is one unit: a review, a customer.
- The **features** (inputs, predictors, X) are the information available about it: text length, verified purchase, number of images.
- The **target** (outcome, label, y) is what we want to predict: helpful votes, churn yes/no.
- A **model** is a family of prediction rules with free **parameters**; **training** (fitting) chooses the parameters from labelled examples; **prediction** applies the fitted rule to new observations.

**Supervised learning** learns from examples with a known target. It is **regression** when the target is a number and **classification** when it is a category. **Unsupervised learning** has no target and looks for structure (clusters, Session 11).

```mermaid
flowchart LR
    X["features X<br/>(n rows x p columns)"] --> FIT["model.fit(X_train, y_train)"]
    Y["target y"] --> FIT
    FIT --> M["fitted model<br/>(parameters)"]
    NEW["new X"] --> PRED["model.predict(X_new)"]
    M --> PRED
    PRED --> YH["predictions y-hat"]
```

Every scikit-learn model follows this interface: `fit(X, y)` learns, `predict(X)` returns predictions, `score(X, y)` returns a default metric (R² for regressors, accuracy for classifiers).

### Why it matters

The vocabulary is shared by every library, paper and job description. Being precise about what counts as a feature also prevents **leakage**: a feature that is not known at prediction time (for example the star rating, when predicting sentiment) makes a model look better than it can be (Session 7).

### How it works in Python

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["log_words"] = np.log1p(reviews["text"].str.split().str.len())

X = reviews[["log_words", "n_images"]]          # features: a table, one column per feature
y = np.log1p(reviews["helpful_vote"])           # target: one number per review
model = LinearRegression().fit(X, y)            # training: choose the parameters
print(model.coef_.round(3), round(model.intercept_, 3))   # [0.189 0.047] -0.229
print(model.predict(X.head(3)).round(2))        # [0.77 0.48 0.66]: predictions
```

### In practice

- Spam filters are supervised classifiers trained on messages that users have marked as spam.
- Real-estate platforms such as Zillow predict sale prices (regression) from features of the property.
- Credit scoring predicts default (classification) from application and account data.

> [!NOTE]
> Statisticians and ML practitioners use different words for the same things: predictor = feature, response = target, fit = train, estimate = learned parameter. ISLP ([workbook 04](../workbooks/04-islp-linear-regression-lab.ipynb)) uses the statistical terms.

## The data split into training and test sets

### Concept

A model is useful if it predicts **new** observations well. To estimate this, hold back part of the labelled data as a **test set**, fit only on the **training set**, and evaluate once on the test set. The score on the training data is optimistic because the model has seen the answers.

`train_test_split` shuffles the rows and splits them, commonly 80/20 or 75/25. A fixed `random_state` makes the split reproducible. For classification, `stratify=y` keeps the class shares equal in both parts. When predictions are about the future, split by time instead (Sessions 7 and 12); the course leaderboard does exactly that (train up to 2021, test 2022–2023).

### Why it matters

Without a held-out test set, a more complex model always looks better, whether it has learned a pattern or memorised the data ([page 2](02-overfitting-and-robust-regression.md)). The test score is our estimate of performance on next month's reviews.

### How it works in Python

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["log_words"] = np.log1p(reviews["text"].str.split().str.len())
reviews["log_votes"] = np.log1p(reviews["helpful_vote"])

train, test = train_test_split(reviews, test_size=0.2, random_state=42)
print(len(train), len(test))                                     # 40000 10000
print(round(train["log_votes"].mean(), 3), round(test["log_votes"].mean(), 3))   # 0.341 0.339
```

### In practice

- Kaggle competitions and the course leaderboard keep the test labels hidden, so that nobody can fit to them.
- Regulated applications such as credit scoring validate models on data that were not used for development (out-of-time validation).
- Medical prediction models are judged by external validation: performance on patients from other hospitals.

> [!CAUTION]
> **Touch the test set once.** If you try ten models and keep the one with the best test score, the test score is no longer an honest estimate. Choose models on the training data (cross-validation, Session 7), then evaluate the final choice once.

## Simple and multiple linear regression

### Concept

**Simple linear regression** describes the target as a straight line in one feature: ŷ = b₀ + b₁·x. The **intercept** b₀ is the prediction at x = 0; the **slope** b₁ is the change in ŷ per one-unit increase of x. The **residual** of an observation is y − ŷ, its vertical distance to the line.

**Ordinary least squares (OLS)** chooses the coefficients that minimise the sum of squared residuals. For one feature: b₁ = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)², b₀ = ȳ − b₁·x̄ (the line passes through the point of means; Session 5, [page 4](../../05-eda-and-statistics/theory/04-correlation-and-communication.md#from-correlation-to-the-regression-line)).

Worked example: points (1, 2), (2, 3), (3, 5). x̄ = 2, ȳ = 10/3. Σ(x − x̄)(y − ȳ) = (−1)(−4/3) + 0 + (1)(5/3) = 3; Σ(x − x̄)² = 2. So b₁ = 1.5 and b₀ = 10/3 − 3 = 0.33: ŷ = 0.33 + 1.5x.

**Multiple linear regression** adds features: ŷ = b₀ + b₁x₁ + b₂x₂ + … Each coefficient is the expected change in y for a one-unit change in that feature **while the other features are held fixed**. This is why a coefficient can shrink or change sign when another feature is added. A categorical feature enters as 0/1 **dummy variables**, one per category except a reference category.

A **residual plot** (residuals against fitted values) checks the model: it should show a band without structure around zero.

![Two residual plots: a structureless band for a well-specified model, and a striped, skewed pattern for the helpful-votes model](figures/residual-plot.png)

The right panel shows the helpful-votes model. The lowest stripe consists of reviews with 0 votes: their residual is −ŷ, so it falls as the prediction rises. The long upper tail shows reviews with many more votes than predicted. Both signal that a straight line on log(1 + votes) is a rough approximation; count models (Poisson, negative binomial) are an alternative.

### Why it matters

Linear regression is the reference model for numeric targets: fast, interpretable, and often hard to beat with small data. Its coefficients answer "how much does y change with x, holding the rest fixed?", the form a report needs. Every more flexible model in the course is compared with it.

### How it works in Python

statsmodels gives the statistical view (coefficients with standard errors and confidence intervals); scikit-learn gives the prediction view (fit, predict, score).

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["log_words"] = np.log1p(reviews["text"].str.split().str.len())
reviews["log_votes"] = np.log1p(reviews["helpful_vote"])
train, test = train_test_split(reviews, test_size=0.2, random_state=42)

# statsmodels: formula interface, intercept added automatically, dummies via C() or bools
fit = smf.ols("log_votes ~ log_words + n_images + verified_purchase + rating", data=train).fit()
print(fit.params.round(3).to_dict())
# {'Intercept': -0.31, 'verified_purchase[T.True]': 0.118, 'log_words': 0.194,
#  'n_images': 0.051, 'rating': -0.01}
print(fit.conf_int().loc["log_words"].round(3).tolist())   # [0.188, 0.2]
print(round(fit.rsquared, 3))                             # 0.108

# scikit-learn: the same coefficients, prediction interface
features = ["log_words", "n_images", "verified_purchase", "rating"]
model = LinearRegression().fit(train[features], train["log_votes"])
print({f: round(float(c), 3) for f, c in zip(features, model.coef_)})
# {'log_words': 0.194, 'n_images': 0.051, 'verified_purchase': 0.118, 'rating': -0.01}
residuals = test["log_votes"] - model.predict(test[features])
print(round(residuals.mean(), 3))                         # -0.001: centred near zero on test data
```

Interpretation: holding images, verification and rating fixed, a 10 % longer review predicts about 0.194 × ln(1.1) ≈ 0.018 higher log(1 + votes), roughly 2 % more (1 + votes). Verified purchases receive more votes than unverified ones of the same length, although Session 5 showed that unverified reviews are longer: the coefficient is a comparison **at equal length**.

### In practice

- Hedonic regression: statistical offices regress prices on product characteristics to adjust price indices for quality change (housing, computers).
- Labour economics: wage equations with education and experience as predictors (Mincer equation).
- Marketing mix models regress sales on advertising spend per channel, controlling for season and price.

> [!WARNING]
> **Coefficients are not causal effects.** "Holding fixed" means "comparing observations with equal values in the data", not "changing x in the world". A coefficient can absorb the effect of omitted variables (confounding, Session 5).

## Regression metrics

### Concept

With yᵢ the true values, ŷᵢ the predictions and n observations:

- **MAE** (mean absolute error) = mean of |yᵢ − ŷᵢ|: the typical error, in the units of y.
- **RMSE** (root mean squared error) = √(mean of (yᵢ − ŷᵢ)²): also in units of y, but squares errors first, so large errors weigh more. RMSE ≥ MAE always.
- **R²** = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²: the share of the variance of y that the model explains, compared with predicting the mean. 1 is perfect, 0 is no better than the mean, and on test data it can be negative.

Worked example: errors of 1, −1 and 4 give MAE = (1 + 1 + 4) / 3 = 2 and RMSE = √((1 + 1 + 16) / 3) = √6 ≈ 2.45. The single large error moves RMSE more than MAE.

### Why it matters

The metric defines what "good" means. MAE suits costs that grow linearly with the error; RMSE suits settings where large errors are disproportionately bad. R² has no unit and is easy to compare across models on the same data, but not across datasets. Always compare with the baseline.

### How it works in Python

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

reviews = pd.read_parquet("case-study/data/train_sample.parquet")
reviews["log_words"] = np.log1p(reviews["text"].str.split().str.len())
reviews["log_votes"] = np.log1p(reviews["helpful_vote"])
train, test = train_test_split(reviews, test_size=0.2, random_state=42)

models = {"baseline (mean)": (DummyRegressor(strategy="mean"), ["log_words"]),
          "simple": (LinearRegression(), ["log_words"]),
          "multiple": (LinearRegression(), ["log_words", "n_images", "verified_purchase", "rating"])}
for name, (model, cols) in models.items():
    model.fit(train[cols], train["log_votes"])
    pred = model.predict(test[cols])
    print(f"{name:16s} MAE {mean_absolute_error(test['log_votes'], pred):.3f}  "
          f"RMSE {root_mean_squared_error(test['log_votes'], pred):.3f}  "
          f"R2 {r2_score(test['log_votes'], pred):.3f}")
# baseline (mean)  MAE 0.481  RMSE 0.643  R2 -0.000
# simple           MAE 0.433  RMSE 0.608  R2 0.107
# multiple         MAE 0.431  RMSE 0.606  R2 0.113
```

Length alone explains about 11 % of the variance of log(1 + votes) on unseen reviews; three more features add little. The model is better than the baseline but far from precise, which is typical for popularity outcomes.

### In practice

- Energy companies report forecast errors of electricity load as MAPE or RMSE, depending on whether peak errors are costly.
- Weather services evaluate temperature forecasts with RMSE and MAE against station measurements.
- Real-estate valuation models (automated valuation models) are compared by median absolute percentage error.

> [!CAUTION]
> **R² is not accuracy.** An R² of 0.11 can be the best achievable for a noisy outcome, and an R² of 0.95 can hide large errors for a small but important group. Look at the errors in units, and at the residual plot.

## Check your understanding

1. Name the ten lifecycle steps and the step where the target, metric and baseline are fixed.
2. Why is the error on the training data an optimistic estimate of the error on new data?
3. Compute the least-squares line for the points (0, 1), (1, 3), (2, 5).
4. The coefficient of `verified_purchase` is 0.118. Explain in one sentence what it compares.
5. A model has MAE 0.43 and RMSE 0.61 on the same data. What does the gap tell you about the errors?

## Further reading

- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, chapters 2–3. Springer. <https://www.statlearning.com/>
- INRIA (2024). *scikit-learn MOOC*, module "Linear models". <https://inria.github.io/scikit-learn-mooc/>
- Chapman, P., et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS.
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems 28*. <https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems>
