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

- Evaluation metrics for classification: [confusion matrix and accuracy](theory/02-classification-metrics.md#confusion-matrix-and-accuracy), [precision, recall and F1](theory/02-classification-metrics.md#precision-recall-and-f1) and [macro-F1](theory/02-classification-metrics.md#macro-f1-for-several-classes)
- [The ROC curve and AUC](theory/02-classification-metrics.md#the-roc-curve-and-auc)
- [The precision–recall curve](theory/02-classification-metrics.md#the-precisionrecall-curve)

*Practice:* Evaluate the churn models with confusion matrices and ROC curves and explain which metric fits the business question ([churn workbook](workbooks/05-case-study-churn-pipelines.ipynb), part B).

### 2:00–2:45 · Decision thresholds and calibration

Theory: [03-thresholds-and-calibration.md](theory/03-thresholds-and-calibration.md)

- [Decision thresholds from the costs of errors](theory/03-thresholds-and-calibration.md#decision-thresholds-from-the-costs-of-errors)
- [Calibration of predicted probabilities](theory/03-thresholds-and-calibration.md#calibration-of-predicted-probabilities)

*Practice:* Case study: choose the churn threshold from the cost of a retention offer against the value of a lost customer, and check whether the predicted probabilities can be trusted ([threshold and calibration workbook](workbooks/13-case-study-churn-threshold-and-calibration.ipynb), part C).

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-pipelines-and-knn.md](theory/01-pipelines-and-knn.md) | Baselines, encoding, scaling, Pipeline and ColumnTransformer, k-NN and decision boundaries | 1 | core |
| [theory/02-classification-metrics.md](theory/02-classification-metrics.md) | Confusion matrix, precision, recall, F1, macro-F1, ROC/AUC, precision–recall curve, all on Telco churn | 2 | core |
| [theory/03-thresholds-and-calibration.md](theory/03-thresholds-and-calibration.md) | Cost-based thresholds, TunedThresholdClassifierCV, calibration curves, Brier score, recalibration of a class-weighted churn model | 3 | core |
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
| [workbooks/13-case-study-churn-threshold-and-calibration.ipynb](workbooks/13-case-study-churn-threshold-and-calibration.ipynb) | Own: who gets a retention offer? Out-of-fold cost curve, TunedThresholdClassifierCV with a cost scorer, sensitivity to the cost assumptions, calibration curves of four churn models, recalibration with CalibratedClassifierCV (Telco) | 3 | core |

The INRIA notebooks read data from `datasets/` and images from `figures/` in this folder (paths `../datasets/` and `../figures/` relative to `workbooks/`); see [source.md](source.md).

## Before and after the session

**Before.** Re-run your Session 6 logistic regression for Telco churn. Think about one decision in your own work that is taken from a score (a credit limit, a reminder, an inspection): what does each kind of error cost?

**Team project until the next session.** A first classifier on simple features (language, member state, description length) in a pipeline, evaluated with accuracy and macro-F1 against the baselines.

**Further reading (optional).**
- Google Machine Learning Crash Course: Classification. https://developers.google.com/machine-learning/crash-course/classification
- MLU-Explain (Amazon): precision and recall, ROC and AUC, interactive. https://mlu-explain.github.io/
- scikit-learn: Metrics and scoring. https://scikit-learn.org/stable/modules/model_evaluation.html
- INRIA scikit-learn MOOC, Module 1 and "Evaluating model performance". https://inria.github.io/scikit-learn-mooc/

## Setup

The theory code and all workbooks run in the course environment (pandas, pyarrow, scikit-learn ≥ 1.5 for `TunedThresholdClassifierCV`, matplotlib). Workbook 02 also uses `seaborn`. The theory pages, the figures and the case-study workbooks download the IBM Telco churn data from GitHub. Workbooks 10–12 download data from OpenML on first use; workbook 11 loads the credit-card fraud data (OpenML 1597, about 285,000 rows), which takes a while.

Run from the repository root:

```bash
uv run --with jupyterlab --with pandas --with pyarrow --with scikit-learn --with matplotlib --with seaborn \
    jupyter lab sessions/08-classification-and-metrics
```

To regenerate the figures of the theory pages:

```bash
uv run python sessions/08-classification-and-metrics/theory/figures/make_figures.py
```
