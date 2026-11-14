# Session 6 · Introduction to machine learning with regression

> **Guiding question.** How does a model learn from data, and how do we know whether it has learned too little or too much?

## 1 · The machine learning lifecycle and linear regression

[Open notebook](workbooks/01-lecture-lifecycle-and-linear-regression.ipynb)

- The machine learning lifecycle: target, metric and baseline.
- Supervised learning: features, target, training, prediction.
- Training and test sets; the median-price baseline.
- Simple and multiple linear regression: least squares, coefficients, dummies, residuals and diagnostics.
- Regression metrics: MAE, RMSE and R² in euros.
- Worked example: what drives nightly prices in Berlin?
- Underfitting and overfitting: polynomial degree, training versus test error, the bias–variance trade-off.

**End-to-end: regression.** Hourly bike rentals in Washington, D.C., in seven steps from the problem to the report: linear regression, the same model tuned with a grid search, and gradient boosting, each scored with MAE, RMSE and R² and a plot of predicted against true rentals; the report gives a bootstrap 95% interval for each MAE and for the gain of each step, with a note on TabPFN.

[Open notebook](workbooks/end-to-end-regression.ipynb)

## 2 · Other losses and other targets: robust and logistic regression

[Open notebook](workbooks/02-lecture-robust-and-logistic-regression.ipynb)

- Robust regression with the Huber loss on prices with extreme asking prices.
- Quantile regression: the median and the 90th-percentile price.
- Logistic regression: from a linear model to probabilities with the sigmoid.
- Log-odds, odds ratios and the interpretation of coefficients.
- Fitting with statsmodels and scikit-learn.
- The scikit-learn pattern: `fit`, `predict`, `score`, `transform`, pipelines.
- A first look at predicted classes; worked example: whom should the retention team call?

## Team project until the next session

Baselines on the project data: the most frequent heading and a simple rule.

## Setup

- Block 2 downloads the IBM Telco churn data (about 1 MB) from GitHub when it runs, so it needs network access.
- The end-to-end notebook downloads the Bike Sharing Demand data (about 1 MB) from OpenML on its first run and works offline afterwards.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
