# Session 10 · Tree-based models: decision trees, random forests and gradient boosting

> [!NOTE]
> **Guiding question.** Can a more flexible model do better, and what does it cost?

**Learning outcomes.** Students are able to

- explain how decision trees, random forests and gradient boosting predict
- train, tune and compare XGBoost, LightGBM and CatBoost
- interpret a tree ensemble

## Session plan

**0:00–0:45 · Decision trees** ([theory page 1](theory/01-decision-trees.md))

- [Decision trees: splits, Gini impurity computed by hand](theory/01-decision-trees.md#1-splits-and-gini-impurity-computed-by-hand), [how a tree grows and how to read it](theory/01-decision-trees.md#2-how-a-tree-grows-and-how-to-read-it)
- [Tree depth as a hyperparameter, overfitting of deep trees and pruning](theory/01-decision-trees.md#3-tree-depth-overfitting-of-deep-trees-and-pruning)
- *Practice:* fit and visualise a decision tree on the churn data → [05-churn-decision-tree.ipynb](workbooks/05-churn-decision-tree.ipynb)

**1:00–1:45 · Ensembles and boosting libraries** ([theory page 2](theory/02-bagging-random-forests-and-boosting.md), [theory page 3](theory/03-xgboost-lightgbm-catboost.md))

- [Bagging and random forests](theory/02-bagging-random-forests-and-boosting.md#1-bagging-and-random-forests)
- [Gradient boosting](theory/02-bagging-random-forests-and-boosting.md#2-gradient-boosting)
- [XGBoost, LightGBM and CatBoost](theory/03-xgboost-lightgbm-catboost.md#1-xgboost-lightgbm-and-catboost)
- [Categorical features](theory/03-xgboost-lightgbm-catboost.md#2-categorical-features), [class weights](theory/03-xgboost-lightgbm-catboost.md#3-class-weights), [early stopping](theory/03-xgboost-lightgbm-catboost.md#4-early-stopping) and [tuning](theory/03-xgboost-lightgbm-catboost.md#5-tuning)
- *Practice:* train and tune gradient boosting models on the churn data and compare them → [15-churn-gradient-boosting.ipynb](workbooks/15-churn-gradient-boosting.ipynb)

**2:00–2:45 · Interpretation and model comparison** ([theory page 4](theory/04-interpretation-and-model-comparison.md))

- Interpretation of tree-based models with [feature importance and permutation importance](theory/04-interpretation-and-model-comparison.md#1-feature-importance-impurity-based-and-permutation-importance) and [SHAP values](theory/04-interpretation-and-model-comparison.md#2-shap-values-as-a-model-checking-tool), used to check and debug the model
- [Comparing a challenger with the current model](theory/04-interpretation-and-model-comparison.md#3-comparing-a-challenger-with-the-current-model)
- [The case study: what drives Berlin Airbnb prices, and do trees price better?](theory/04-interpretation-and-model-comparison.md#4-the-case-study-what-drives-berlin-airbnb-prices-and-do-trees-price-better)
- *Practice:* case study: what drives nightly prices in Berlin, and is a tree ensemble worth replacing the linear price model of Session 6? Compare a random forest and gradient boosting with it on host-grouped folds, decide with paired differences and a bootstrap interval, and interpret the challenger with permutation importance and SHAP → [20-case-study-airbnb-price-trees.ipynb](workbooks/20-case-study-airbnb-price-trees.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-decision-trees.md](theory/01-decision-trees.md) | Splits, Gini by hand, growing and reading a tree, depth, pruning | 1 | core |
| [theory/02-bagging-random-forests-and-boosting.md](theory/02-bagging-random-forests-and-boosting.md) | Bagging, random forests, OOB score, gradient boosting by hand | 2 | core |
| [theory/03-xgboost-lightgbm-catboost.md](theory/03-xgboost-lightgbm-catboost.md) | The three libraries, categorical features, class weights, early stopping, tuning | 2 | core |
| [theory/04-interpretation-and-model-comparison.md](theory/04-interpretation-and-model-comparison.md) | Impurity vs permutation importance, SHAP for model checking, challenger vs current model (Telco); trees vs the linear model for Berlin Airbnb prices | 3 | core |
| [workbooks/01-trees-classification.ipynb](workbooks/01-trees-classification.ipynb) | A tree's decision boundary on penguins (INRIA MOOC) | 1 | core |
| [workbooks/02-tree-structure.ipynb](workbooks/02-tree-structure.ipynb) | The arrays inside a fitted tree; decision path of one sample (scikit-learn example) | 1 | optional |
| [workbooks/03-trees-hyperparameters.ipynb](workbooks/03-trees-hyperparameters.ipynb) | Effect of `max_depth` and `min_samples_leaf` (INRIA MOOC) | 1 | core |
| [workbooks/04-cost-complexity-pruning.ipynb](workbooks/04-cost-complexity-pruning.ipynb) | Cost-complexity pruning path (scikit-learn example) | 1 | optional |
| [workbooks/05-churn-decision-tree.ipynb](workbooks/05-churn-decision-tree.ipynb) | **Practice 1:** Gini by hand, depth curve, pruning and plot of a churn tree (own) | 1 | core |
| [workbooks/06-ensemble-bagging.ipynb](workbooks/06-ensemble-bagging.ipynb) | Bootstrap and bagging step by step (INRIA MOOC) | 2 | core |
| [workbooks/07-ensemble-random-forest.ipynb](workbooks/07-ensemble-random-forest.ipynb) | Random forests vs bagging on Adult census (INRIA MOOC) | 2 | core |
| [workbooks/08-ensemble-gradient-boosting.ipynb](workbooks/08-ensemble-gradient-boosting.ipynb) | Boosting as fitting residuals (INRIA MOOC) | 2 | core |
| [workbooks/09-hist-gradient-boosting.ipynb](workbooks/09-hist-gradient-boosting.ipynb) | Histogram-based gradient boosting and its speed (INRIA MOOC) | 2 | optional |
| [workbooks/10-gradient-boosting-early-stopping.ipynb](workbooks/10-gradient-boosting-early-stopping.ipynb) | Early stopping in gradient boosting (scikit-learn example) | 2 | core |
| [workbooks/11-gradient-boosting-categorical.ipynb](workbooks/11-gradient-boosting-categorical.ipynb) | Native categorical support vs one-hot and ordinal encoding (scikit-learn example) | 2 | optional |
| [workbooks/12-ensemble-hyperparameters.ipynb](workbooks/12-ensemble-hyperparameters.ipynb) | Tuning random forests and gradient boosting (INRIA MOOC) | 2 | optional |
| [workbooks/13-catboost-tutorial.ipynb](workbooks/13-catboost-tutorial.ipynb) | CatBoost's Python tutorial: categorical features, CV, early stopping (CatBoost) | 2 | optional |
| [workbooks/14-islp-tree-based-methods-lab.ipynb](workbooks/14-islp-tree-based-methods-lab.ipynb) | ISLP Chapter 8 lab: trees, bagging, random forests, boosting, BART | 1–2 | optional |
| [workbooks/15-churn-gradient-boosting.ipynb](workbooks/15-churn-gradient-boosting.ipynb) | **Practice 2:** six models on the same folds, class weights, early stopping, search (own) | 2 | core |
| [workbooks/16-permutation-importance.ipynb](workbooks/16-permutation-importance.ipynb) | Impurity importance vs permutation importance with random features (scikit-learn example) | 3 | core |
| [workbooks/17-feature-importance.ipynb](workbooks/17-feature-importance.ipynb) | Coefficients, impurity and permutation importance compared (INRIA MOOC) | 3 | optional |
| [workbooks/18-shap-census-xgboost.ipynb](workbooks/18-shap-census-xgboost.ipynb) | SHAP values for an XGBoost model on census income (SHAP; outputs included) | 3 | core |
| [workbooks/19-shap-causal-caution.ipynb](workbooks/19-shap-causal-caution.ipynb) | Why SHAP values are not causal effects (SHAP; read with outputs) | 3 | optional |
| [workbooks/20-case-study-airbnb-price-trees.ipynb](workbooks/20-case-study-airbnb-price-trees.ipynb) | **Practice 3:** Berlin Airbnb prices: linear model vs ridge, random forest and boosting on host-grouped folds, early stopping, randomised search, paired comparison, permutation importance, SHAP, leak check (own) | 3 | core |
| [datasets/](datasets/) | Penguins and Adult census CSV files used by the INRIA MOOC notebooks | – | – |

Third-party sources and licences: [source.md](source.md).

## Before and after the session

**Preparation.** Re-read Session 7 on grouped cross-validation and `RandomizedSearchCV`, Session 6 on the linear price model, and Session 9 page 2 on leaking columns (the revenue leak). Make sure `case-study/data/airbnb/` exists (`uv run python case-study/prepare_airbnb.py`). Run the first cell of [15-churn-gradient-boosting.ipynb](workbooks/15-churn-gradient-boosting.ipynb) once to check that XGBoost, LightGBM and CatBoost import (see Setup).

**Team project until the next session.** Interim review (10 minutes per team): model or dashboard, validation, plan to the end.

**Further reading (optional).**

- James, G. et al. (2023). [*An Introduction to Statistical Learning with Python*](https://www.statlearning.com/), Chapter 8 — free PDF.
- Molnar, C. [*Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/), chapters on permutation importance and SHAP — free online.
- Amazon MLU. [*MLU-Explain: Random Forest*](https://mlu-explain.github.io/random-forest/) — interactive article.
- INRIA. [*scikit-learn MOOC, module "Ensemble of models"*](https://inria.github.io/scikit-learn-mooc/ensemble/ensemble_module_intro.html).

## Setup

The course environment (`uv sync`, then `uv run jupyter lab`) contains scikit-learn, XGBoost, LightGBM, CatBoost and SHAP.

- **macOS:** XGBoost and LightGBM need the OpenMP runtime. Install it once with `brew install libomp`; without it, `import xgboost` fails with an error about `libomp.dylib`.
- Notebooks that load the Telco data and the scikit-learn examples need an internet connection (GitHub, OpenML).
- Optional notebooks need extra packages: `13-catboost-tutorial.ipynb` uses `hyperopt` (`uv run --with hyperopt jupyter lab`); `14-islp-tree-based-methods-lab.ipynb` uses the `ISLP` package (`uv run --with ISLP jupyter lab`, a large install because it pulls in PyTorch); `19-shap-causal-caution.ipynb` uses `econml` and `graphviz` and is meant to be read with its stored outputs.
- `20-case-study-airbnb-price-trees.ipynb` needs the Inside Airbnb data (`uv run python case-study/prepare_airbnb.py`) and runs in about three minutes. Reference result: with host-grouped 5-fold CV, LightGBM on all features has an MAE of €46.5 a night against €56.9 for the Session 6 linear model.
