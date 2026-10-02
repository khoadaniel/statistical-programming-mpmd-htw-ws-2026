# Session 7 · Validation and hyperparameter tuning: data splits, cross-validation, bootstrap and leakage

> [!NOTE]
> **Guiding question.** How well will a model perform on data it has never seen, and how do we choose its settings?

**Learning outcomes.** After this session you are able to

- split data into training, validation and test sets and estimate performance with cross-validation
- diagnose underfitting and overfitting with validation and learning curves
- control model complexity with regularisation, tune hyperparameters with grid and randomised search and prevent data leakage

## Session plan

### 0:00–0:45 · Data splits, cross-validation and the bootstrap

Theory: [01-splits-and-cross-validation.md](theory/01-splits-and-cross-validation.md)

- [Training, validation and test data and their roles](theory/01-splits-and-cross-validation.md#training-validation-and-test-data-and-their-roles)
- [Why a single split is unreliable](theory/01-splits-and-cross-validation.md#why-a-single-split-is-unreliable)
- [k-fold and stratified cross-validation](theory/01-splits-and-cross-validation.md#k-fold-and-stratified-cross-validation)
- [Grouped and time-series splits](theory/01-splits-and-cross-validation.md#grouped-and-time-series-splits)
- [Bootstrap confidence interval of a metric](theory/01-splits-and-cross-validation.md#bootstrap-confidence-interval-of-a-metric)

*Practice:* How well would a price model work for a host listing a first flat in Berlin? Cross-validate the Airbnb price model with random and host-grouped folds and report the error in euros with a bootstrap interval ([Airbnb workbook](workbooks/20-case-study-airbnb-price-validation.ipynb), part A).

### 1:00–1:45 · Regularisation, learning curves and tuning

Theory: [02-regularisation-and-tuning.md](theory/02-regularisation-and-tuning.md)

- [Parameters and hyperparameters](theory/02-regularisation-and-tuning.md#parameters-and-hyperparameters)
- [Regularisation with ridge and lasso, and the penalty as a hyperparameter](theory/02-regularisation-and-tuning.md#regularisation-with-ridge-and-lasso)
- [Validation curves and learning curves to diagnose underfitting and overfitting](theory/02-regularisation-and-tuning.md#validation-curves-and-learning-curves)
- [Hyperparameter tuning with GridSearchCV and RandomizedSearchCV](theory/02-regularisation-and-tuning.md#hyperparameter-tuning-with-gridsearchcv-and-randomizedsearchcv)

*Practice:* Tune the ridge and lasso penalties of the price model and read its validation and learning curves ([Airbnb workbook](workbooks/20-case-study-airbnb-price-validation.ipynb), part B).

### 2:00–2:45 · Data leakage, pipelines, nested CV and the final test

Theory: [03-leakage-and-final-test.md](theory/03-leakage-and-final-test.md)

- [Data leakage: preprocessing outside the cross-validation](theory/03-leakage-and-final-test.md#data-leakage-preprocessing-outside-the-cross-validation), [target leakage](theory/03-leakage-and-final-test.md#data-leakage-target-leakage)
- [Preprocessing inside a Pipeline](theory/03-leakage-and-final-test.md#preprocessing-inside-a-pipeline)
- [Nested cross-validation](theory/03-leakage-and-final-test.md#nested-cross-validation)
- [The final test on held-out data](theory/03-leakage-and-final-test.md#the-final-test-on-held-out-data)

*Practice:* Demonstrate two leaking workflows on the price model (preprocessing fitted outside the cross-validation; the same host in training and test), then fix them with a Pipeline and grouped folds and compare the scores ([Airbnb workbook](workbooks/20-case-study-airbnb-price-validation.ipynb), part C).

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-splits-and-cross-validation.md](theory/01-splits-and-cross-validation.md) | Roles of the data parts, split variance, k-fold, stratified, grouped (Airbnb hosts) and time-series CV, listing and cluster bootstrap CI for new hosts | 1 | core |
| [theory/02-regularisation-and-tuning.md](theory/02-regularisation-and-tuning.md) | Hyperparameters, ridge and lasso, validation and learning curves, grid and randomised search, all on the Berlin price model | 2 | core |
| [theory/03-leakage-and-final-test.md](theory/03-leakage-and-final-test.md) | Preprocessing and target leakage, the same host in training and validation, Pipeline with grouped folds, nested CV, final test on held-out hosts, all on the Berlin price model | 3 | core |
| [workbooks/01-train-test-and-cross-validation.ipynb](workbooks/01-train-test-and-cross-validation.ipynb) | INRIA: one split versus repeated splits, variability of the score | 1 | core |
| [workbooks/02-model-validation.ipynb](workbooks/02-model-validation.ipynb) | PDSH 5.03: holdout, k-fold, validation and learning curves, grid search (overview of blocks 1–2). With current matplotlib, change `plt.style.use('seaborn-whitegrid')` to `'seaborn-v0_8-whitegrid'` when you run it (the file itself must stay unchanged, licence ND) | 1–2 | optional |
| [workbooks/03-cross-validation-splitters.ipynb](workbooks/03-cross-validation-splitters.ipynb) | scikit-learn: visual comparison of KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit | 1 | core |
| [workbooks/04-stratified-cross-validation.ipynb](workbooks/04-stratified-cross-validation.ipynb) | INRIA: KFold versus StratifiedKFold | 1 | optional |
| [workbooks/05-grouped-cross-validation.ipynb](workbooks/05-grouped-cross-validation.ipynb) | INRIA: GroupKFold on handwritten digits written by the same people | 1 | optional |
| [workbooks/06-islp-resampling-lab.ipynb](workbooks/06-islp-resampling-lab.ipynb) | ISLP Lab 5: validation set, LOOCV, k-fold CV, bootstrap | 1 | optional |
| [workbooks/07-bootstrap-and-simulation.ipynb](workbooks/07-bootstrap-and-simulation.ipynb) | statsthinking21: simulation and the bootstrap (CC-BY-NC) | 1 | optional |
| [workbooks/08-islp-ridge-lasso-lab.ipynb](workbooks/08-islp-ridge-lasso-lab.ipynb) | ISLP Lab 6: subset selection, ridge, lasso, PCR/PLS with cross-validated penalty | 2 | optional |
| [workbooks/09-ridge-regularisation.ipynb](workbooks/09-ridge-regularisation.ipynb) | INRIA: ridge on Ames housing, scaling, choosing alpha with RidgeCV | 2 | core |
| [workbooks/10-validation-curve.ipynb](workbooks/10-validation-curve.ipynb) | INRIA: overfitting and underfitting, validation curve of a tree | 2 | core |
| [workbooks/11-learning-curve.ipynb](workbooks/11-learning-curve.ipynb) | INRIA: effect of the sample size, learning curve | 2 | core |
| [workbooks/12-learning-curve-sklearn.ipynb](workbooks/12-learning-curve-sklearn.ipynb) | scikit-learn: learning curves and scalability of naive Bayes and SVM | 2 | optional |
| [workbooks/13-grid-search.ipynb](workbooks/13-grid-search.ipynb) | INRIA: GridSearchCV on the adult census data | 2 | core |
| [workbooks/14-randomized-search.ipynb](workbooks/14-randomized-search.ipynb) | INRIA: RandomizedSearchCV with log-uniform distributions | 2 | core |
| [workbooks/15-grid-search-custom-refit.ipynb](workbooks/15-grid-search-custom-refit.ipynb) | scikit-learn: grid search with several metrics and a custom refit rule | 2 | optional |
| [workbooks/16-data-leakage-feature-selection.ipynb](workbooks/16-data-leakage-feature-selection.ipynb) | INRIA: feature selection outside versus inside the pipeline (leakage) | 3 | core |
| [workbooks/17-nested-cross-validation.ipynb](workbooks/17-nested-cross-validation.ipynb) | INRIA: nested cross-validation | 3 | core |
| [workbooks/18-nested-cv-sklearn.ipynb](workbooks/18-nested-cv-sklearn.ipynb) | scikit-learn: nested versus non-nested CV on iris | 3 | optional |
| [workbooks/19-case-study-churn-validation.ipynb](workbooks/19-case-study-churn-validation.ipynb) | Own: the Telco churn model of Session 6: CV with a bootstrap CI of ROC AUC, tuning `C` and `class_weight`, nested CV | 1–3 | optional |
| [workbooks/20-case-study-airbnb-price-validation.ipynb](workbooks/20-case-study-airbnb-price-validation.ipynb) | Own: Berlin price model: random vs host-grouped CV for four models, held-out hosts with a cluster bootstrap, ridge and lasso, validation and learning curves, grid and randomised search, two leaking workflows (target means from all rows; the same host in training and validation) fixed with a Pipeline and grouped folds and checked on new hosts (Inside Airbnb) | 1–3 | core |

The INRIA notebooks read data from `datasets/` and images from `figures/` in this folder (paths `../datasets/` and `../figures/` relative to `workbooks/`); see [source.md](source.md).

## Before and after the session

**Before.** Re-run your Session 6 notebooks (the price regression, logistic regression for Telco churn). Make sure the case-study data exist: `uv run python case-study/prepare_airbnb.py` (Inside Airbnb, Berlin). Read the first section of the theory page 01.

**Team project until the next session.** Validation plan for the project model (machine learning teams): splitter, metric, uncertainty, leakage checks. Uncertainty of key results (analytics teams): bootstrap intervals for the main numbers of your analysis.

**Further reading (optional).**
- James et al. (2023). *ISLP*, Chapters 5 and 6. https://www.statlearning.com/
- scikit-learn: *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html
- INRIA scikit-learn MOOC, Modules 2 and 3. https://inria.github.io/scikit-learn-mooc/
- Kapoor & Narayanan (2023). Leakage and the reproducibility crisis in ML-based science. https://doi.org/10.1016/j.patter.2023.100804

## Setup

The theory code and the core workbooks need only the course environment (pandas, pyarrow, scikit-learn, scipy, matplotlib) and the case-study data (`uv run python case-study/prepare_airbnb.py`). Some optional workbooks need more:

| Package | Needed by |
|---|---|
| `seaborn` | 13-grid-search |
| `ISLP`, `statsmodels` (and optionally `l0bnb`) | 06 and 08 (ISLP labs). `ISLP` installs PyTorch as a dependency, a large download. |
| `nhanes` | 07-bootstrap-and-simulation |

Run from the repository root:

```bash
uv run --with jupyterlab --with pandas --with pyarrow --with scikit-learn --with scipy --with matplotlib --with seaborn \
    jupyter lab sessions/07-validation-and-tuning
# for the ISLP labs add: --with ISLP --with statsmodels ; for workbook 07 add: --with nhanes
```

To regenerate the figures of the theory pages:

```bash
uv run --with pandas --with pyarrow --with scikit-learn --with matplotlib \
    python sessions/07-validation-and-tuning/theory/figures/make_figures.py
```
