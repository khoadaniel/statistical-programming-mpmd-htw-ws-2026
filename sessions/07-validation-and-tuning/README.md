# Session 7 · Validation and hyperparameter tuning

> **Guiding question.** How well will a model perform on data it has never seen, and how do we choose its settings?

## 1 · An honest estimate: data splits, cross-validation and leakage

[Open notebook](workbooks/01-lecture-splits-cross-validation-and-leakage.ipynb)

- Training, validation and test data and their roles; training error versus test error.
- Why a single split is unreliable; repeated shuffle-splits.
- k-fold and stratified cross-validation; how many folds; sorted data.
- Grouped splits by host, hidden groups and time-series splits; all splitters side by side.
- Bootstrap confidence interval of the error in euros, by listing and by host (cluster bootstrap).
- Data leakage (information that the model could not have at prediction time reaches it during training): preprocessing outside the cross-validation.
- Data leakage: target leakage.
- Preprocessing inside a Pipeline; worked example: which validation estimate holds for new hosts.

## 2 · Choosing settings: regularisation and tuning

[Open notebook](workbooks/02-lecture-regularisation-and-tuning.ipynb)

- Parameters and hyperparameters; setting the test hosts aside first.
- Ridge and lasso regularisation: coefficient paths, why scaling matters, `RidgeCV`.
- Validation curves and learning curves; what noise is.
- Grid search and randomised search, tuning ridge and lasso together, the one-standard-error rule.
- Nested cross-validation and how large the optimism is.
- The final test on held-out hosts; worked example with a cluster-bootstrap interval.

## Team project until the next session

Validation design: how the team splits the decisions and why, a time-ordered validation as a comparison, metrics, uncertainty of the scores, inputs that may not be used.

## Setup

- Internet access: notebook 01 downloads the IBM Telco churn table from GitHub.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
