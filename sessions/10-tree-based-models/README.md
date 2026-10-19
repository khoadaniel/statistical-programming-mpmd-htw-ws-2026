# Session 10 · Tree-based models

> **Guiding question.** Can a more flexible model do better, and what does it cost?

## 1 · From one tree to an ensemble

[Open notebook](workbooks/01-lecture-from-one-tree-to-an-ensemble.ipynb)

- Decision trees: splits and Gini impurity computed by hand, decision boundaries, reading a tree, the arrays inside a fitted tree.
- Tree depth, overfitting and pruning: regression trees, depth as a hyperparameter, pre-pruning, cost-complexity pruning, the instability of single trees.
- Bagging and random forests: bootstrap samples, averaging, `max_features`, the out-of-bag score.
- Gradient boosting as fitting the remaining errors step by step; boosting versus bagging; histogram boosting.
- Worked example: tree, forest or boosting for churn? Tree, random forest and gradient boosting on the same churn folds.

## 2 · Gradient boosting in practice and interpretation

[Open notebook](workbooks/02-lecture-gradient-boosting-and-interpretation.ipynb)

- XGBoost, LightGBM and CatBoost: shared ideas, differences, one fit of each, native interfaces.
- Categorical features, class weights, early stopping and tuning.
- Interpretation: built-in importance and its bias, permutation importance, coefficients versus importances, SHAP values; why SHAP is not causal.
- Comparing models: paired folds, corrected t-test and Bayesian comparison.
- Worked example: is a tree ensemble worth replacing the linear price model? The linear model of Session 6 against tree ensembles, for listings already on Airbnb.

## Team project until the next session

Interim review: data, validation plan, baselines, plan for Sessions 13–16. Sessions 10 and 11 share one teaching day, so the review covers the task of Session 11 as well.

## Setup

- **Install the OpenMP runtime before this session.** XGBoost and LightGBM, the libraries of block 2, use OpenMP, a small system library that lets their compiled code run on all cores of your processor. They do not bring their own copy, so without it `import xgboost` or `import lightgbm` fails with an error that mentions `libomp`, `libgomp` or `vcomp140.dll`. Everything earlier in the course works without it. Install it once:

  | System | Command | Note |
  |---|---|---|
  | macOS | `brew install libomp` | Needed on almost every Mac. Homebrew is already installed from [Getting started](../../GETTING_STARTED.md), step 2. |
  | Windows | Download and run the [Microsoft Visual C++ Redistributable (X64)](https://aka.ms/vc14/vc_redist.x64.exe) | Usually installed already; if the installer says so, you are done. |
  | Linux | `sudo apt install libgomp1` (Ubuntu, Debian) or `sudo dnf install libgomp` (Fedora) | Usually installed already. |

  Then run `uv run check_setup` in the repository folder: the lines for `xgboost` and `lightgbm` must show ✅.
- The notebooks read the Telco data from GitHub (internet needed) and the Airbnb data from `case-study/data/airbnb/`: download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- Some extras need more packages: extra 11 `hyperopt`, extra 09 `ISLP`, extra 18 `econml` and `graphviz` (see the extras README).

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
