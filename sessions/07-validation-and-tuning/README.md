# Session 7 · Validation and hyperparameter tuning

> **Guiding question.** How well will a model perform on data it has never seen, and how do we choose its settings?

## 1 · An honest estimate: data splits, cross-validation and leakage

[Open notebook](workbooks/01-lecture-splits-cross-validation-and-leakage.ipynb)

- Training, validation and test data; training error versus test error.
- Why a single split is unreliable; repeated shuffle-splits.
- k-fold and stratified cross-validation.
- Grouped splits by host and time-series splits.
- Bootstrap confidence interval of the error in euros, by listing and by host.
- Data leakage: preprocessing outside the cross-validation, target leakage.
- Preprocessing inside a Pipeline; checking the promises on new hosts.

## 2 · Choosing settings: regularisation and tuning

[Open notebook](workbooks/02-lecture-regularisation-and-tuning.ipynb)

- Parameters and hyperparameters.
- Ridge and lasso regularisation; coefficient paths and scaling.
- Validation curves and learning curves.
- Grid search and randomised search, tuning ridge and lasso together.
- Nested cross-validation.
- The final test on held-out hosts, with a cluster-bootstrap interval.

## Team project until the next session

Write a validation plan: how you split by time, which metrics you use, the uncertainty of the scores, and which inputs may not be used.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
