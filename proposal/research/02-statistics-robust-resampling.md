# Research notes — Statistics, robust methods, resampling, fairness

Links checked on 2026-09-24 (HTTP 200 unless marked). All free unless noted.
Items marked [403] are MIT Press pages that block bots; each has an archive.org mirror that returned 200.

## 1. Basic statistics recap
| Title | URL | Type | Why | Exercises |
|---|---|---|---|---|
| Think Stats 3e (Downey, 2025, CC) | https://allendowney.github.io/ThinkStats/ · code https://github.com/AllenDowney/ThinkStats | Book + notebooks | Pandas/SciPy-first descriptive stats, tests, correlation, resampling. Core text. | Colab notebook with exercises per chapter |
| Computational and Inferential Thinking (Data 8) | https://inferentialthinking.com/ · https://data8.org/ | Book + course | Simulation-based inference; good for career changers | Labs/HW on data8.org |
| OpenIntro Statistics 4e | https://www.openintro.org/book/os/ | Book | Inference, chi-square, regression; PDF + slides | Many exercises (R labs) |
| Introduction to Modern Statistics 2e | https://openintro-ims.netlify.app/ | Book | Randomisation/bootstrap first | Exercises (language-agnostic) |
| Learning Statistics with Python | https://ethanweed.github.io/pythonbook/ | Book | Python-native tests, chi-square, ANOVA, effect sizes | Worked examples |
| Modern Statistics for Modern Biology | https://www.huber.embl.de/msmb/ | Book (R) | Multiple testing / FDR, simulation | Exercises (R) |
| SciPy stats | https://docs.scipy.org/doc/scipy/tutorial/stats.html | Docs | Tests, distributions, `bootstrap`, `permutation_test` | Examples |
| Pingouin | https://pingouin-stats.org/ | Docs | Tests with effect sizes, CIs, Bayes factors | Examples |
| statsmodels contingency tables / multitest | https://www.statsmodels.org/stable/contingency_tables.html | Docs | Chi-square, odds ratios, Holm/BH | Examples |
| Statistical Rethinking 2026 (lectures) | https://github.com/rmcelreath/stat_rethinking_2026 · PyMC port https://github.com/pymc-devs/pymc-resources/tree/main/Rethinking_2 | Course (book paid) | Bayesian + causal thinking | Weekly HW + solutions |
| Regression and Other Stories — Python/Bambi ports | https://avehtari.github.io/ROS-Examples/ · https://github.com/bambinos/educational-resources | Code ports | Regression examples | Worked examples |

## 2. Advanced methods
| Title | URL | Type | Why |
|---|---|---|---|
| ISLP (2023) | https://www.statlearning.com/ · labs https://github.com/intro-stat-learning/ISLP_labs | Book + labs | Regularisation, PCA, splines, resampling, trees; Python labs per chapter |
| Elements of Statistical Learning | https://hastie.su.domains/ElemStatLearn/ | Book | Theory reference |
| Feature Engineering and Selection | https://feat.engineering/ | Book (R) | Box-Cox/Yeo-Johnson, missing data, leakage |
| Flexible Imputation of Missing Data 2e | https://stefvanbuuren.name/fimd/ | Book (R) | MCAR/MAR/MNAR, MICE |
| sklearn imputation | https://scikit-learn.org/stable/modules/impute.html | Docs | Simple/KNN/Iterative imputers in pipelines |
| statsmodels MICE | https://www.statsmodels.org/stable/imputation.html | Docs | Multiple imputation with Rubin's rules |
| statsmodels RLM + notebooks | https://www.statsmodels.org/stable/rlm.html · https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_0.html | Docs | Huber/Tukey M-estimators with inference |
| statsmodels quantile regression | https://www.statsmodels.org/stable/examples/notebooks/generated/quantile_regression.html | Notebook | |
| sklearn robust fit example | https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html | Docs | Huber vs RANSAC vs Theil-Sen |
| sklearn outlier detection | https://scikit-learn.org/stable/modules/outlier_detection.html · https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html | Docs | IsolationForest, LOF, EllipticEnvelope |
| PyOD | https://pyod.readthedocs.io/en/latest/ | Library | 40+ detectors |
| sklearn map data to normal | https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html | Docs | Box-Cox vs Yeo-Johnson vs quantile |
| UMAP | https://umap-learn.readthedocs.io/en/latest/ | Docs | |
| How to Use t-SNE Effectively (Distill) | https://distill.pub/2016/misread-tsne/ | Interactive | Misreading embeddings |
| Understanding UMAP (PAIR) | https://pair-code.github.io/understanding-umap/ | Interactive | |

## 3. Resampling, evaluation, UQ, interpretability
| Title | URL | Why |
|---|---|---|
| sklearn cross-validation + CV indices | https://scikit-learn.org/stable/modules/cross_validation.html · https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html | Splitter behaviour visualised |
| sklearn Common Pitfalls | https://scikit-learn.org/stable/common_pitfalls.html | Leakage |
| Kapoor & Narayanan 2023, Patterns | https://doi.org/10.1016/j.patter.2023.100804 · https://reproducible.cs.princeton.edu/ | 8 leakage types, model info sheets |
| SciPy bootstrap | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html | BCa intervals |
| sklearn statistical model comparison | https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html | Comparing CV scores properly |
| Gentle Intro to Conformal Prediction | https://arxiv.org/abs/2107.07511 · https://github.com/aangelopoulos/conformal-prediction | Notebooks |
| MAPIE | https://mapie.readthedocs.io/en/stable/ | Conformal in sklearn |
| Interpretable ML (Molnar) | https://christophm.github.io/interpretable-ml-book/ | PDP, SHAP, limits |
| SHAP | https://shap.readthedocs.io/en/latest/ | |
| sklearn MOOC (Inria) | https://inria.github.io/scikit-learn-mooc/ | Exercises + quizzes |
| MLU-Explain | https://mlu-explain.github.io/cross-validation/ | Visual essays |

## 4. Fairness / gender & diversity
| Title | URL | Why |
|---|---|---|
| Fairness and Machine Learning | https://fairmlbook.org/ | Criteria, trade-offs, law |
| Fairlearn | https://fairlearn.org/ | MetricFrame, mitigation |
| AIF360 | https://aif360.readthedocs.io/en/stable/ | Alternative toolkit |
| Data Feminism | https://data-feminism.mitpress.mit.edu/ [403] · https://archive.org/details/mit_press_book_9780262358521 | Power, gender, classification |
| Datasheets for Datasets | https://arxiv.org/abs/1803.09010 | Assignment template |
| Model Cards | https://arxiv.org/abs/1810.03993 · https://huggingface.co/docs/hub/model-cards | Assignment template |
| Practical Data Ethics (fast.ai) | https://ethics.fast.ai/ | Lectures |
| Deon checklist | https://deon.drivendata.org/ | Ethics checklist in repos |
| Kaggle Intro to AI Ethics | https://www.kaggle.com/learn/intro-to-ai-ethics | Hands-on |

## 5. Modern extensions
| Title | URL |
|---|---|
| Causal Inference for the Brave and True | https://matheusfacure.github.io/python-causality-handbook/landing-page.html |
| DoWhy | https://www.pywhy.org/dowhy/ |
| CausalPy | https://causalpy.readthedocs.io/ |
| The Mixtape / The Effect | https://mixtape.scunning.com/ · https://theeffectbook.net/ |
| PyMC / Bambi | https://www.pymc.io/projects/docs/en/stable/learn.html · https://bambinos.github.io/bambi/ |
| Bayesian Modeling and Computation in Python | https://bayesiancomputationbook.com/ |
| Think Bayes 2 | https://allendowney.github.io/ThinkBayes2/ |
| TabPFN (Nature 2025) | https://www.nature.com/articles/s41586-024-08328-6 · https://github.com/PriorLabs/TabPFN |

## Takeaways for the syllabus
- Core spine: Think Stats 3e (recap) → ISLP + labs (advanced/resampling) → Fairness & ML + Fairlearn.
- Graded documentation tasks: model info sheet (Kapoor & Narayanan), datasheet, model card.
- R-only books (MSMB, van Buuren, Kuhn & Johnson) are conceptual references only; pair them with Python docs.
