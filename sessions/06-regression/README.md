# Session 6 · Introduction to machine learning with regression: linear and logistic regression, underfitting and overfitting

> [!NOTE]
> **Guiding question.** How does a model learn from data, and how do we know whether it has learned too little or too much?

**Learning outcomes.** Students are able to

- describe the stages of the machine learning lifecycle, from problem definition to monitoring, and relate them to the earlier sessions
- split data into training and test sets and fit, interpret and evaluate a linear regression
- recognise underfitting and overfitting and use robust regression for data with outliers
- fit and interpret a logistic regression for a binary outcome

## Session plan

### 0:00–0:45 · From the lifecycle to linear regression

- [The machine learning lifecycle](theory/01-lifecycle-and-linear-regression.md#the-machine-learning-lifecycle), from problem definition (target, metric, baseline) to monitoring and maintenance, and how Sessions 3–5 already covered data collection, cleaning and exploration
- [Supervised learning: features, target, training, prediction](theory/01-lifecycle-and-linear-regression.md#supervised-learning-features-target-training-prediction)
- [The data split into training and test sets](theory/01-lifecycle-and-linear-regression.md#the-data-split-into-training-and-test-sets)
- [Simple and multiple linear regression](theory/01-lifecycle-and-linear-regression.md#simple-and-multiple-linear-regression) (least squares, coefficients, residuals)
- [Regression metrics (MAE, RMSE, R²)](theory/01-lifecycle-and-linear-regression.md#regression-metrics)

*Practice:* What drives nightly prices in Berlin? Map a pricing aid for hosts to the stages of the lifecycle; split the Inside Airbnb listings, fit and interpret a price model (guests, room type, distance to the centre, district) and report its error in euros against the median-price baseline.

### 1:00–1:45 · Model complexity and robust regression

- [Underfitting and overfitting: model complexity (polynomial degree), training versus test error](theory/02-overfitting-and-robust-regression.md#underfitting-and-overfitting-model-complexity), [the bias–variance trade-off](theory/02-overfitting-and-robust-regression.md#the-biasvariance-trade-off)
- Robust regression for data with outliers: [Huber](theory/02-overfitting-and-robust-regression.md#robust-regression-huber), [quantile regression](theory/02-overfitting-and-robust-regression.md#robust-regression-quantile-regression)

*Practice:* Compare training and test error for increasing model complexity with 5,360 and with 60 training listings; compare least squares with Huber and quantile regression on prices with extreme asking prices, and give a new host the median and the 90th-percentile price for a comparable listing.

### 2:00–2:45 · Logistic regression

- [Logistic regression: from a linear model to probabilities (sigmoid)](theory/03-logistic-regression.md#from-a-linear-model-to-probabilities-the-sigmoid), [log-odds and the interpretation of coefficients](theory/03-logistic-regression.md#log-odds-and-the-interpretation-of-coefficients)
- [Fitting with statsmodels and scikit-learn](theory/03-logistic-regression.md#fitting-with-statsmodels-and-scikit-learn)
- [A first look at predicted classes](theory/03-logistic-regression.md#a-first-look-at-predicted-classes)

*Practice:* Case study: predict churn probability for the IBM Telco customers with logistic regression and interpret the coefficients.

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-lifecycle-and-linear-regression.md](theory/01-lifecycle-and-linear-regression.md) | Ten-step lifecycle, supervised learning, train–test split, linear regression, MAE/RMSE/R² | 1 | core |
| [theory/02-overfitting-and-robust-regression.md](theory/02-overfitting-and-robust-regression.md) | Under- and overfitting, bias–variance, Huber and quantile regression | 2 | core |
| [theory/03-logistic-regression.md](theory/03-logistic-regression.md) | Sigmoid, log-odds, odds ratios, statsmodels and scikit-learn, thresholds | 3 | core |
| [workbooks/01-linear-regression-without-sklearn.ipynb](workbooks/01-linear-regression-without-sklearn.ipynb) | A linear model by hand: slope, intercept, fitting by eye (penguins) | 1 | core |
| [workbooks/02-linear-regression-in-sklearn.ipynb](workbooks/02-linear-regression-in-sklearn.ipynb) | `LinearRegression`, coefficients, MSE and MAE | 1 | core |
| [workbooks/03-regression-metrics.ipynb](workbooks/03-regression-metrics.ipynb) | MSE, R², MAE, median and percentage errors, prediction-error plots (Ames housing) | 1 | core |
| [workbooks/04-islp-linear-regression-lab.ipynb](workbooks/04-islp-linear-regression-lab.ipynb) | ISLP lab 3: simple and multiple regression, diagnostics, interactions, qualitative predictors | 1 | optional |
| [workbooks/05-underfitting-overfitting.ipynb](workbooks/05-underfitting-overfitting.ipynb) | Polynomial degree 1, 4, 15 on a cosine curve (scikit-learn example) | 2 | core |
| [workbooks/06-polynomial-features.ipynb](workbooks/06-polynomial-features.ipynb) | Non-linear relationships with `PolynomialFeatures` and other feature expansions | 2 | core |
| [workbooks/07-robust-fit-compared.ipynb](workbooks/07-robust-fit-compared.ipynb) | Least squares, Theil–Sen, RANSAC and Huber under outliers in X and y | 2 | core |
| [workbooks/08-huber-vs-ridge.ipynb](workbooks/08-huber-vs-ridge.ipynb) | Huber regression with strong outliers; effect of ε (ridge is only the comparison) | 2 | optional |
| [workbooks/09-statsmodels-robust-models-intro.ipynb](workbooks/09-statsmodels-robust-models-intro.ipynb) | statsmodels `RLM`: Huber's T, Hampel, weights | 2 | core |
| [workbooks/10-statsmodels-m-estimators.ipynb](workbooks/10-statsmodels-m-estimators.ipynb) | M-estimators, norms, robust scale (MAD) | 2 | optional |
| [workbooks/11-quantile-regression-sklearn.ipynb](workbooks/11-quantile-regression-sklearn.ipynb) | `QuantileRegressor` with heteroscedastic and asymmetric noise | 2 | core |
| [workbooks/12-quantile-regression-statsmodels.ipynb](workbooks/12-quantile-regression-statsmodels.ipynb) | `QuantReg` on Engel's food-expenditure data | 2 | optional |
| [workbooks/13-ransac.ipynb](workbooks/13-ransac.ipynb) | RANSAC inliers and outliers | 2 | optional |
| [workbooks/14-theil-sen.ipynb](workbooks/14-theil-sen.ipynb) | Theil–Sen versus least squares and RANSAC | 2 | optional |
| [workbooks/15-logistic-regression.ipynb](workbooks/15-logistic-regression.ipynb) | Logistic regression in scikit-learn, decision boundary, `predict_proba` (penguins) | 3 | core |
| [workbooks/16-islp-classification-lab.ipynb](workbooks/16-islp-classification-lab.ipynb) | ISLP lab 4: logistic regression on stock-market data; sections on LDA, QDA, naive Bayes and KNN are beyond this session | 3 | optional |
| [workbooks/17-case-study-airbnb-price-and-churn.ipynb](workbooks/17-case-study-airbnb-price-and-churn.ipynb) | **Case study** for all three practice tasks: lifecycle of a Berlin price model, linear regression of the log price with metrics in euros, complexity curves, OLS vs Huber vs quantile regression (median and 90th percentile), Telco churn | 1–3 | core |
| [workbooks/data/](workbooks/data/) | `penguins_regression.csv`, `penguins_classification.csv`, `house_prices.csv` for workbooks 01–03 and 15 | – | – |

Sources and licences of third-party notebooks: [source.md](source.md).

## Before and after the session

**Preparation.** Re-read the last section of Session 5, [From correlation to the regression line](../05-eda-and-statistics/theory/04-correlation-and-communication.md#from-correlation-to-the-regression-line). Prepare the Airbnb data once (`uv run python case-study/prepare_airbnb.py`) and run the first cell of the [case-study notebook](workbooks/17-case-study-airbnb-price-and-churn.ipynb). Optional: ISLP sections 2.1–2.2 (link below).

**Team project until the next session.** Baselines on the project data: the most frequent heading and a simple rule.

**Further reading (free).**

- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, chapters 2–4. <https://www.statlearning.com/>
- INRIA *scikit-learn MOOC*, modules "The predictive modeling pipeline" and "Linear models". <https://inria.github.io/scikit-learn-mooc/>
- scikit-learn User Guide: *Linear models*. <https://scikit-learn.org/stable/modules/linear_model.html>
- Gelman, A., Hill, J., & Vehtari, A. (2020). *Regression and Other Stories* (free PDF). <https://avehtari.github.io/ROS-Examples/>

## Setup

The course environment (root `pyproject.toml`) covers the theory pages, the case-study notebook and workbooks 01–03 and 05–15. From the repository root:

```bash
uv run jupyter lab
uv run python sessions/06-regression/theory/figures/make_figures.py   # regenerate figures
```

The case-study notebook, the theory pages and the figures read the Inside Airbnb Berlin listings from `case-study/data/airbnb/`, created by `uv run python case-study/prepare_airbnb.py` (see [case-study/README.md](../../case-study/README.md)); its Telco part and the logistic-regression code on the theory page download the IBM Telco data (about 1 MB) from GitHub.

The two ISLP labs (workbooks 04 and 16) need the `ISLP` package, which pulls in PyTorch (about 1 GB):

```bash
uv run --with ISLP jupyter lab
```

Alternatively, open them in Google Colab with the badge at the top of each notebook.
