# Session 8 · Classification and evaluation metrics

> **Guiding question.** Which customers will cancel, and how do we measure whether a classifier is good enough?

## 1 · Classifiers, pipelines and the confusion matrix

[Open notebook](workbooks/01-lecture-classifiers-pipelines-and-confusion-matrix.ipynb)

- Classification tasks and baselines: the majority class and a hand-made churn rule.
- Preparing inputs: one-hot and ordinal encoding, scaling, rare and unseen categories.
- `Pipeline` and `ColumnTransformer`: preparation fitted on the training data only.
- k-nearest neighbours, decision boundaries, and choosing *k* by cross-validation.
- Confusion matrix, accuracy, precision, recall, F1, balanced accuracy.
- Macro-F1 for several classes.
- Worked example: rule, k-NN and logistic regression compared for the retention team.

## 2 · Scores, thresholds and rare classes

[Open notebook](workbooks/02-lecture-scores-thresholds-and-rare-classes.ipynb)

- The ROC curve and AUC, also for several classes.
- The precision–recall curve and average precision.
- Decision thresholds from the costs of errors, `TunedThresholdClassifierCV`, customer-specific costs.
- Calibration of predicted probabilities and recalibration.
- Class imbalance, why accuracy misleads, class weights.
- Undersampling, oversampling and SMOTE inside the cross-validation with imbalanced-learn pipelines.
- Worked example: which strategy is cheapest for the retention team.

## Team project until the next session

Together with the task of Session 9, which follows on the same day: a first classifier on simple features (language, member state, description length) in a pipeline, evaluated with accuracy and macro-F1 against the baselines.

## Setup

- Internet access: the notebooks download the IBM Telco churn data from GitHub; extras 09, 10 and 12 download data from OpenML on first use.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
