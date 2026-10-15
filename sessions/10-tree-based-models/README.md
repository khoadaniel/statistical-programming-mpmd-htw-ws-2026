# Session 10 · Tree-based models

> **Guiding question.** Can a more flexible model do better, and what does it cost?

## 1 · From one tree to an ensemble

[Open notebook](workbooks/01-lecture-from-one-tree-to-an-ensemble.ipynb)

- Decision trees: splits and Gini impurity computed by hand, decision boundaries, reading a tree, the arrays inside a fitted tree.
- Tree depth, overfitting and pruning: regression trees, depth as a hyperparameter, pre-pruning, cost-complexity pruning, the instability of single trees.
- Bagging and random forests: bootstrap samples, averaging, `max_features`, the out-of-bag score.
- Gradient boosting as fitting the remaining errors step by step; boosting versus bagging; histogram boosting.
- Worked example: tree, random forest and gradient boosting on the same churn folds.

## 2 · Gradient boosting in practice and interpretation

[Open notebook](workbooks/02-lecture-gradient-boosting-and-interpretation.ipynb)

- XGBoost, LightGBM and CatBoost: shared ideas, differences, one fit of each, native interfaces.
- Categorical features, class weights, early stopping and tuning.
- Interpretation: built-in importance and its bias, permutation importance, coefficients versus importances, SHAP values; why SHAP is not causal.
- Comparing models: paired folds, corrected t-test and Bayesian comparison.
- Worked example: is a tree ensemble worth replacing the linear price model of Session 6?

## Team project until the next session

Interim review (10 minutes per team): data, validation plan, baselines, plan for Sessions 13–16. Sessions 10 and 11 share one teaching day, so the review covers the task of Session 11 as well.

## Setup

- On macOS, XGBoost and LightGBM need the OpenMP runtime: `brew install libomp` once.
- The notebooks read the Telco data from GitHub (internet needed) and the Airbnb data from `case-study/data/airbnb/`: download `airbnb.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- Some extras need more packages: extra 11 `hyperopt`, extra 09 `ISLP`, extra 18 `econml` and `graphviz` (see the extras README).

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
