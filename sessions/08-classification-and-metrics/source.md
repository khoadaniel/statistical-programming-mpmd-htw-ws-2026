# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/01-categorical-encoding.ipynb](workbooks/01-categorical-encoding.ipynb) | Encoding of categorical variables: OrdinalEncoder, OneHotEncoder | [INRIA/scikit-learn-mooc](https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/main/notebooks/03_categorical_pipeline.ipynb) | [CC-BY-4.0](https://github.com/INRIA/scikit-learn-mooc/blob/main/LICENSE) | 2026-10-01 | Renamed from `03_categorical_pipeline.ipynb`; content unchanged |
| [workbooks/02-scaling-in-pipelines.ipynb](workbooks/02-scaling-in-pipelines.ipynb) | StandardScaler in pipelines, fit/transform, cross_validate | [INRIA/scikit-learn-mooc](https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/main/notebooks/02_numerical_pipeline_scaling.ipynb) | [CC-BY-4.0](https://github.com/INRIA/scikit-learn-mooc/blob/main/LICENSE) | 2026-09-26 | Renamed from `01-scaling-in-pipelines.ipynb` (originally `inria-mooc_numerical-scaling.ipynb`); content unchanged |
| [workbooks/03-column-transformer.ipynb](workbooks/03-column-transformer.ipynb) | ColumnTransformer for mixed data types | [INRIA/scikit-learn-mooc](https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/main/notebooks/03_categorical_pipeline_column_transformer.ipynb) | [CC-BY-4.0](https://github.com/INRIA/scikit-learn-mooc/blob/main/LICENSE) | 2026-10-01 | Renamed from `03_categorical_pipeline_column_transformer.ipynb`; content unchanged |
| [workbooks/04-knn-decision-boundary.ipynb](workbooks/04-knn-decision-boundary.ipynb) | k-NN decision boundaries | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/47f024d726d245e034c7690b4664721f/plot_classification.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_classification.ipynb`; content unchanged |
| [workbooks/06-classification-metrics.ipynb](workbooks/06-classification-metrics.ipynb) | Classification metrics, confusion matrix, ROC and PR curves | [INRIA/scikit-learn-mooc](https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/main/notebooks/metrics_classification.ipynb) | [CC-BY-4.0](https://github.com/INRIA/scikit-learn-mooc/blob/main/LICENSE) | 2026-10-01 | Renamed from `metrics_classification.ipynb`. Patched for scikit-learn ≥ 1.7: in the two `RocCurveDisplay.from_estimator` cells, the keyword arguments `marker`, `color`, `linestyle` are passed as `curve_kwargs={...}` (the old form raises a TypeError in scikit-learn 1.9); otherwise unchanged |
| [workbooks/07-confusion-matrix.ipynb](workbooks/07-confusion-matrix.ipynb) | Confusion matrix display and normalisation | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/9ad55bf68758018c9961815802c65e18/plot_confusion_matrix.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_confusion_matrix.ipynb`; content unchanged |
| [workbooks/08-roc-curve.ipynb](workbooks/08-roc-curve.ipynb) | ROC curves, multiclass one-vs-rest / one-vs-one | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/40f4aad91af595a370d7582e3a23bed7/plot_roc.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_roc.ipynb`; content unchanged |
| [workbooks/09-precision-recall-curve.ipynb](workbooks/09-precision-recall-curve.ipynb) | Precision–recall curves, average precision | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/764d061a261a2e06ad21ec9133361b2d/plot_precision_recall.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_precision_recall.ipynb`; content unchanged |
| [workbooks/10-tuned-decision-threshold.ipynb](workbooks/10-tuned-decision-threshold.ipynb) | TunedThresholdClassifierCV | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/c158013efa5fd61eaa463a2f88014a07/plot_tuned_decision_threshold.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_tuned_decision_threshold.ipynb`; content unchanged |
| [workbooks/11-cost-sensitive-learning.ipynb](workbooks/11-cost-sensitive-learning.ipynb) | Cost-sensitive decision threshold, business metric | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/133f2198d3ab792c75b39a63b0a99872/plot_cost_sensitive_learning.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_cost_sensitive_learning.ipynb`; content unchanged |
| [workbooks/12-calibration-curve.ipynb](workbooks/12-calibration-curve.ipynb) | Calibration curves, Brier score, sigmoid and isotonic calibration | [scikit-learn example gallery 1.9](https://scikit-learn.org/1.9/_downloads/6d4f620ec6653356eb970c2a6ed62081/plot_calibration_curve.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_calibration_curve.ipynb`; content unchanged |

## Data and figures used by the INRIA notebooks

The INRIA notebooks read `../datasets/...` and `../figures/...` relative to `workbooks/`. These files are copied unchanged from the same repository.

| File | Used by | Source | Licence | Downloaded |
|---|---|---|---|---|
| `datasets/adult-census.csv` (5.1 MB) | 01, 02, 03 | [INRIA/scikit-learn-mooc datasets](https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets); original: Becker & Kohavi (1996), UCI Adult | [CC-BY-4.0](https://github.com/INRIA/scikit-learn-mooc/blob/main/datasets/README.md) | 2026-09-26 |
| `datasets/blood_transfusion.csv` | 06 | INRIA/scikit-learn-mooc datasets; original: Yeh (2008), UCI | CC-BY-4.0 | 2026-10-01 |
| `figures/api_diagram-*.svg` (6 files: transformer fit / transform / fit_transform, pipeline fit / predict, columntransformer) | 02, 03 | [INRIA/scikit-learn-mooc figures](https://github.com/INRIA/scikit-learn-mooc/tree/main/figures) | CC-BY-4.0 | 2026-10-01 |

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-pipelines-and-knn.md](theory/01-pipelines-and-knn.md), [theory/02-classification-metrics.md](theory/02-classification-metrics.md), [theory/03-thresholds-calibration-leaderboard.md](theory/03-thresholds-calibration-leaderboard.md) | Theory pages of the three blocks. Author: course team | CC-BY-4.0 |
| [theory/figures/make_figures.py](theory/figures/make_figures.py) and its PNGs (`confusion_matrix.png`, `roc_pr_curves.png`, `knn_boundaries.png`, `calibration_curve.png`) | Figures of the theory pages (Telco data). Author: course team | CC-BY-4.0 |
| [workbooks/05-case-study-churn-pipelines.ipynb](workbooks/05-case-study-churn-pipelines.ipynb) | Telco pipeline; churn rule vs k-NN vs logistic regression; confusion matrices, ROC and PR curves. Author: course team | CC-BY-4.0 |
| [workbooks/13-case-study-leaderboard-l1.ipynb](workbooks/13-case-study-leaderboard-l1.ipynb) | Leaderboard round L1: seven simple text features, logistic regression, time-based validation, `submission.csv`. Author: course team | CC-BY-4.0 |

## Citations

- Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
- Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of IJCAI 2001*, 973–978. https://cseweb.ucsd.edu/~elkan/rescale.pdf
- Google for Developers (2025). *Machine Learning Crash Course: Classification*. https://developers.google.com/machine-learning/crash-course/classification
- Hanley, J. A. & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36. https://doi.org/10.1148/radiology.143.1.7063747
- Hou, Y. et al. (2024). Bridging language and items for retrieval and recommendation (Amazon Reviews 2023). arXiv:2403.03952. https://arxiv.org/abs/2403.03952
- IBM (2019). *Telco customer churn sample data*. https://github.com/IBM/telco-customer-churn-on-icp4d
- INRIA (2024). *scikit-learn MOOC*. https://inria.github.io/scikit-learn-mooc/
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer. https://www.statlearning.com/
- Linden, G., Smith, B. & York, J. (2003). Amazon.com recommendations: item-to-item collaborative filtering. *IEEE Internet Computing*, 7(1), 76–80. https://doi.org/10.1109/MIC.2003.1167344
- Murphy, A. H. & Winkler, R. L. (1977). Reliability of subjective probability forecasts of precipitation and temperature. *Journal of the Royal Statistical Society C*, 26(1), 41–47. https://doi.org/10.2307/2346866
- Rosenthal, S., Farra, N. & Nakov, P. (2017). SemEval-2017 Task 4: Sentiment analysis in Twitter. *Proceedings of SemEval-2017*, 502–518. https://aclanthology.org/S17-2088/
- Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- scikit-learn developers (2025). *User guide: metrics and scoring, tuning the decision threshold, probability calibration, pipelines and composite estimators*. https://scikit-learn.org/stable/user_guide.html
- Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L. & Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230. https://doi.org/10.1186/s12916-019-1466-7
- Yeh, I.-C. (2008). Blood Transfusion Service Center [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5GS39
