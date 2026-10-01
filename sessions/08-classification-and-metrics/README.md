# Session 8 · Classification and evaluation metrics: preprocessing pipelines, k-nearest neighbours, confusion matrix, ROC curves and decision thresholds

> [!NOTE]
> **Guiding question.** Which customers will cancel, and how do we measure whether a classifier is good enough?

**Learning outcomes.** After this session you are able to

- prepare mixed numerical and categorical inputs in a preprocessing pipeline
- build classification models and compare them with baselines
- evaluate classifiers with the confusion matrix, precision, recall, F1, macro-F1, ROC curves and AUC, and set a decision threshold from the costs of errors

## Session plan

### 0:00–0:45 · Classification, preprocessing pipelines and k-NN

Theory: [01-pipelines-and-knn.md](theory/01-pipelines-and-knn.md)

- [Classification tasks: predicting a category; baselines (majority class, hand-made rule)](theory/01-pipelines-and-knn.md#classification-tasks-and-baselines)
- [Preparing inputs: one-hot and ordinal encoding of categories, scaling of numerical features](theory/01-pipelines-and-knn.md#preparing-inputs-one-hot-and-ordinal-encoding-scaling)
- [Pipeline and ColumnTransformer, so that preprocessing is fitted on the training data only](theory/01-pipelines-and-knn.md#pipeline-and-columntransformer)
- [k-nearest neighbours and the decision boundary; underfitting and overfitting with the number of neighbours](theory/01-pipelines-and-knn.md#k-nearest-neighbours-and-the-decision-boundary)

*Practice:* Build a preprocessing pipeline for the IBM Telco data and compare a churn rule, k-NN and logistic regression ([churn workbook](workbooks/05-case-study-churn-pipelines.ipynb), part A).

### 1:00–1:45 · Evaluation metrics for classification

Theory: [02-classification-metrics.md](theory/02-classification-metrics.md)

- [Confusion matrix and accuracy](theory/02-classification-metrics.md#confusion-matrix-and-accuracy); [precision, recall and F1](theory/02-classification-metrics.md#precision-recall-and-f1); [macro-F1](theory/02-classification-metrics.md#macro-f1-for-several-classes)
- [The ROC curve and AUC](theory/02-classification-metrics.md#the-roc-curve-and-auc)
- [The precision–recall curve](theory/02-classification-metrics.md#the-precisionrecall-curve)

*Practice:* Evaluate the churn models with confusion matrices and ROC curves and explain which metric fits the business question ([churn workbook](workbooks/05-case-study-churn-pipelines.ipynb), part B).

### 2:00–2:45 · Decision thresholds, calibration and leaderboard round L1

Theory: [03-thresholds-calibration-leaderboard.md](theory/03-thresholds-calibration-leaderboard.md)

- [Decision thresholds from the costs of errors](theory/03-thresholds-calibration-leaderboard.md#decision-thresholds-from-the-costs-of-errors)
- [Calibration of predicted probabilities](theory/03-thresholds-calibration-leaderboard.md#calibration-of-predicted-probabilities)
- [The first leaderboard submission](theory/03-thresholds-calibration-leaderboard.md#the-first-leaderboard-submission-round-l1)

*Practice:* Case study: first leaderboard submission, a logistic regression on the simple text features of Session 2 ([leaderboard workbook](workbooks/13-case-study-leaderboard-l1.ipynb)).

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-pipelines-and-knn.md](theory/01-pipelines-and-knn.md) | Baselines, encoding, scaling, Pipeline and ColumnTransformer, k-NN and decision boundaries | 1 | core |
| [theory/02-classification-metrics.md](theory/02-classification-metrics.md) | Confusion matrix, precision, recall, F1, macro-F1, ROC/AUC, precision–recall curve | 2 | core |
| [theory/03-thresholds-calibration-leaderboard.md](theory/03-thresholds-calibration-leaderboard.md) | Cost-based thresholds, TunedThresholdClassifierCV, calibration, leaderboard L1 | 3 | core |
| [workbooks/01-categorical-encoding.ipynb](workbooks/01-categorical-encoding.ipynb) | INRIA: OrdinalEncoder and OneHotEncoder on the adult census data | 1 | core |
| [workbooks/02-scaling-in-pipelines.ipynb](workbooks/02-scaling-in-pipelines.ipynb) | INRIA: StandardScaler, fit/transform, pipelines, cross_validate | 1 | core |
| [workbooks/03-column-transformer.ipynb](workbooks/03-column-transformer.ipynb) | INRIA: ColumnTransformer for mixed numeric and categorical data | 1 | core |
| [workbooks/04-knn-decision-boundary.ipynb](workbooks/04-knn-decision-boundary.ipynb) | scikit-learn: k-NN decision boundaries on iris (uniform vs distance weights) | 1 | optional |
| [workbooks/05-case-study-churn-pipelines.ipynb](workbooks/05-case-study-churn-pipelines.ipynb) | Own: Telco pipeline; rule vs k-NN vs logistic regression; confusion matrices, ROC and PR curves | 1–2 | core |
| [workbooks/06-classification-metrics.ipynb](workbooks/06-classification-metrics.ipynb) | INRIA: accuracy, confusion matrix, precision, recall, ROC and PR curves (blood transfusion data) | 2 | core |
| [workbooks/07-confusion-matrix.ipynb](workbooks/07-confusion-matrix.ipynb) | scikit-learn: confusion matrix with and without normalisation | 2 | optional |
| [workbooks/08-roc-curve.ipynb](workbooks/08-roc-curve.ipynb) | scikit-learn: ROC curves, one-vs-rest and one-vs-one for several classes | 2 | optional |
| [workbooks/09-precision-recall-curve.ipynb](workbooks/09-precision-recall-curve.ipynb) | scikit-learn: precision–recall curves, average precision, multiclass | 2 | optional |
| [workbooks/10-tuned-decision-threshold.ipynb](workbooks/10-tuned-decision-threshold.ipynb) | scikit-learn: TunedThresholdClassifierCV on the diabetes (Pima) data | 3 | core |
| [workbooks/11-cost-sensitive-learning.ipynb](workbooks/11-cost-sensitive-learning.ipynb) | scikit-learn: cost-sensitive thresholds on credit and fraud data (large download) | 3 | optional |
| [workbooks/12-calibration-curve.ipynb](workbooks/12-calibration-curve.ipynb) | scikit-learn: calibration curves, Brier score, sigmoid and isotonic calibration | 3 | core |
| [workbooks/13-case-study-leaderboard-l1.ipynb](workbooks/13-case-study-leaderboard-l1.ipynb) | Own: seven simple features + logistic regression, time-based validation, `submission.csv` | 3 | core |

The INRIA notebooks read data from `datasets/` and images from `figures/` in this folder (paths `../datasets/` and `../figures/` relative to `workbooks/`); see [source.md](source.md).

## Before and after the session

**Before.** Re-run your Session 6 logistic regression for Telco churn. Make sure `case-study/data/train.parquet` and `test.parquet` exist (`uv run --with pandas --with pyarrow python case-study/prepare_data.py`). Read the leaderboard rules in [case-study/README.md](../../case-study/README.md).

**Team project until the next session.** Baseline and first validated model: a majority or rule baseline, one model in a pipeline, and cross-validated scores with the metric that fits the project question.

**Further reading (optional).**
- Google Machine Learning Crash Course: Classification. https://developers.google.com/machine-learning/crash-course/classification
- MLU-Explain (Amazon): precision and recall, ROC and AUC, interactive. https://mlu-explain.github.io/
- scikit-learn: Metrics and scoring. https://scikit-learn.org/stable/modules/model_evaluation.html
- INRIA scikit-learn MOOC, Module 1 and "Evaluating model performance". https://inria.github.io/scikit-learn-mooc/

## Setup

The theory code and all workbooks run in the course environment (pandas, pyarrow, scikit-learn ≥ 1.5 for `TunedThresholdClassifierCV`, matplotlib). Workbook 02 also uses `seaborn`. Workbooks 10–12 download data from OpenML on first use; workbook 11 loads the credit-card fraud data (OpenML 1597, about 285,000 rows), which takes a while.

Run from the repository root:

```bash
uv run --with jupyterlab --with pandas --with pyarrow --with scikit-learn --with matplotlib --with seaborn \
    jupyter lab sessions/08-classification-and-metrics
```

Score a leaderboard submission (lecturer, who holds the hidden labels):

```bash
uv run --with pandas --with scikit-learn python case-study/score.py submission.csv
```

To regenerate the figures of the theory pages:

```bash
uv run --with pandas --with scikit-learn --with matplotlib \
    python sessions/08-classification-and-metrics/theory/figures/make_figures.py
```
