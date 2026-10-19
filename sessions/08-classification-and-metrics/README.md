# Session 8 · Classification and evaluation metrics

> **Guiding question.** Which customers will cancel, and how do we measure whether a classifier is good enough?

## 1 · Classifiers, pipelines and the confusion matrix

[Open notebook](workbooks/01-lecture-classifiers-pipelines-and-confusion-matrix.ipynb)

- Classification tasks and baselines: the majority class and a hand-made churn rule.
- Preparing inputs: choosing columns by type, one-hot and ordinal encoding, rare and unseen categories, scaling.
- `Pipeline` and `ColumnTransformer`: preparation fitted on the training data only.
- k-nearest neighbours: decision boundaries, choosing *k* by cross-validation, distance weights.
- Confusion matrix, accuracy, precision, recall, F1, balanced accuracy.
- Macro-F1 for several classes: macro, weighted and micro averages, a three-class contract task.
- Worked example: rule, k-NN and logistic regression compared for the retention team.

## 2 · Scores, thresholds and rare classes

[Open notebook](workbooks/02-lecture-scores-thresholds-and-rare-classes.ipynb)

- The ROC curve and AUC, also for several classes.
- The precision–recall curve, average precision, and ROC against PR when positives are rare.
- Decision thresholds from the costs of errors, `TunedThresholdClassifierCV`, customer-specific costs.
- Calibration of predicted probabilities, recalibration, sigmoid against isotonic.
- Class imbalance, why accuracy misleads, balanced and cost-based class weights.
- Resampling inside the cross-validation with imbalanced-learn: undersampling, oversampling, SMOTE and its variants.
- Worked example: which strategy is cheapest for the retention team.

## Team project until the next teaching day

Together with the task of Session 9, which follows on the same day: a first classifier on simple features (language, member state, description length) in a pipeline, evaluated with accuracy and macro-F1 against the baselines.

## Setup

- Internet access: the notebooks download the IBM Telco churn data from GitHub; extras 09, 10 and 12 download data from OpenML on first use.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
