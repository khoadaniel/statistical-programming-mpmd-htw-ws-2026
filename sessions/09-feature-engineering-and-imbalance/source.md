# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/01-time-related-feature-engineering.ipynb](workbooks/01-time-related-feature-engineering.ipynb) | Calendar features, cyclical and spline encodings, interactions (bike sharing demand) | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/7012baed63b9a27f121bae611b8285c2/plot_cyclical_feature_engineering.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_cyclical_feature_engineering.ipynb`; content unchanged |
| [workbooks/02-column-transformer-mixed-types.ipynb](workbooks/02-column-transformer-mixed-types.ipynb) | ColumnTransformer with numerical and categorical columns (Titanic) | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/26f110ad6cff1a8a7c58b1a00d8b8b5a/plot_column_transformer_mixed_types.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_column_transformer_mixed_types.ipynb`; content unchanged |
| [workbooks/03-target-encoder.ipynb](workbooks/03-target-encoder.ipynb) | TargetEncoder vs ordinal and one-hot encoding (wine reviews) | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/7b414ce0c39e11cf961fd4fa23008246/plot_target_encoder.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_target_encoder.ipynb`; content unchanged |
| [workbooks/04-target-encoder-cross-fitting.ipynb](workbooks/04-target-encoder-cross-fitting.ipynb) | Internal cross-fitting of TargetEncoder and the leak without it | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/c3f95dc25241c64632f9c3378fd4e89b/plot_target_encoder_cross_val.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_target_encoder_cross_val.ipynb`; content unchanged |
| [workbooks/08-imbalanced-classes-impact.ipynb](workbooks/08-imbalanced-classes-impact.ipynb) | Effect of imbalance; class weights, undersampling, balanced bagging (Adult census) | [imbalanced-learn examples 0.14](https://imbalanced-learn.org/stable/_downloads/5c34a055001acb990ce9a738f4f795d2/plot_impact_imbalanced_classes.ipynb) | [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE) | 2026-10-01 | Renamed from `plot_impact_imbalanced_classes.ipynb`; content unchanged |
| [workbooks/09-smote-sample-generation.ipynb](workbooks/09-smote-sample-generation.ipynb) | How SMOTE interpolates one new sample | [imbalanced-learn examples 0.14](https://imbalanced-learn.org/stable/_downloads/8795f8c2a7ac7ee1ebff2e061ded7a73/plot_illustration_generation_sample.ipynb) | [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE) | 2026-10-01 | Renamed from `plot_illustration_generation_sample.ipynb`; content unchanged |
| [workbooks/10-over-sampling-comparison.ipynb](workbooks/10-over-sampling-comparison.ipynb) | Random oversampling, SMOTE, ADASYN and variants | [imbalanced-learn examples 0.14](https://imbalanced-learn.org/stable/_downloads/36cd3cf9023b1f17acb9c3114f17b10b/plot_comparison_over_sampling.ipynb) | [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE) | 2026-10-01 | Renamed from `plot_comparison_over_sampling.ipynb`; content unchanged |
| [workbooks/11-under-sampling-comparison.ipynb](workbooks/11-under-sampling-comparison.ipynb) | Undersampling methods | [imbalanced-learn examples 0.14](https://imbalanced-learn.org/stable/_downloads/ac6dee8f2622a104974cc8bcf430c917/plot_comparison_under_sampling.ipynb) | [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE) | 2026-10-01 | Renamed from `plot_comparison_under_sampling.ipynb`; content unchanged |
| [workbooks/12-imblearn-pipeline.ipynb](workbooks/12-imblearn-pipeline.ipynb) | Samplers inside an imbalanced-learn Pipeline | [imbalanced-learn examples 0.14](https://imbalanced-learn.org/stable/_downloads/a1eea81aef2766fd95ce2d6acf6f8c3c/plot_pipeline_classification.ipynb) | [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE) | 2026-10-01 | Renamed from `plot_pipeline_classification.ipynb`; content unchanged |

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-dates-interactions-and-categories.md](theory/01-dates-interactions-and-categories.md), [theory/02-aggregates-leakage-and-text-statistics.md](theory/02-aggregates-leakage-and-text-statistics.md), [theory/03-class-imbalance.md](theory/03-class-imbalance.md) | Theory pages of the three blocks | CC-BY-4.0 |
| [theory/figures/make_figures.py](theory/figures/make_figures.py) and the three PNG files | Cyclical month encoding, SMOTE illustration, effect of resampling and of a lower threshold on Telco churn | CC-BY-4.0 |
| [workbooks/05-case-study-airbnb-features.ipynb](workbooks/05-case-study-airbnb-features.ipynb) | Practice 1: date, distance, interaction, category and amenity features for a Berlin Airbnb price model, host-grouped feature-group comparison. Author: course team | CC-BY-4.0 |
| [workbooks/06-case-study-airbnb-leakage.ipynb](workbooks/06-case-study-airbnb-leakage.ipynb) | Practice 2: past-only demand features from monthly reviews, snapshot and revenue leaks, title statistics. Author: course team | CC-BY-4.0 |
| [workbooks/13-case-study-imbalance.ipynb](workbooks/13-case-study-imbalance.ipynb) | Practice 3: imbalance strategies and a cost-tuned threshold for Telco churn. Author: course team | CC-BY-4.0 |

## Citations

- Breiman, L., Friedman, J. H., Olshen, R. A. & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Chawla, N. V., Bowyer, K. W., Hall, L. O. & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
- Cheng, H.-T. et al. (2016). Wide & deep learning for recommender systems. *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*, 7–10. https://doi.org/10.1145/2988450.2988454
- Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE Symposium Series on Computational Intelligence*. https://doi.org/10.1109/SSCI.2015.33
- Elor, Y. & Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE? arXiv:2201.08528. https://arxiv.org/abs/2201.08528
- Galli, S. (2021). Feature-engine: a Python package for feature engineering for machine learning. *Journal of Open Source Software*, 6(65), 3642. https://doi.org/10.21105/joss.03642
- He, X. et al. (2014). Practical lessons from predicting clicks on ads at Facebook. *Proceedings of ADKDD 2014*. https://doi.org/10.1145/2648584.2648589
- Hermann, J. & Del Balso, M. (2017). *Meet Michelangelo: Uber's machine learning platform*. Uber Engineering blog. https://www.uber.com/blog/michelangelo-machine-learning-platform/
- IBM (2019). *Telco customer churn* (sample data set). https://github.com/IBM/telco-customer-churn-on-icp4d
- Inside Airbnb (2026). *Berlin, Germany: listings, calendar and reviews*, snapshot of 26 June 2026, CC BY 4.0; prepared with `case-study/prepare_airbnb.py` (host names and other personal data removed). https://insideairbnb.com/get-the-data/ · *Data assumptions*: https://insideairbnb.com/data-assumptions/
- Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD*, 6(4), 15. https://doi.org/10.1145/2382577.2382579
- King, G. & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis*, 9(2), 137–163. https://doi.org/10.1093/oxfordjournals.pan.a004868
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press. https://bookdown.org/max/FES/
- Lemaître, G., Nogueira, F. & Aridas, C. K. (2017). Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced datasets in machine learning. *JMLR*, 18(17), 1–5. https://jmlr.org/papers/v18/16-365.html
- Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2022). M5 accuracy competition: results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations*, 3(1), 27–32. https://doi.org/10.1145/507533.507538
- Pargent, F., Pfisterer, F., Thomas, J. & Bischl, B. (2022). Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features. *Computational Statistics*, 37, 2671–2692. https://doi.org/10.1007/s00180-022-01207-6
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 31*. https://arxiv.org/abs/1706.09516
- Rendle, S. (2010). Factorization machines. *IEEE International Conference on Data Mining*, 995–1000. https://doi.org/10.1109/ICDM.2010.127
- Rosset, S., Perlich, C., Świrszcz, G., Melville, P. & Liu, Y. (2010). Medical data mining: insights from winning two competitions. *Data Mining and Knowledge Discovery*, 20, 439–468. https://doi.org/10.1007/s10618-009-0158-x
- Santos, M. S., Soares, J. P., Abreu, P. H., Araújo, H. & Santos, J. (2018). Cross-validation for imbalanced datasets: avoiding overoptimistic and overfitting approaches. *IEEE Computational Intelligence Magazine*, 13(4), 59–76. https://doi.org/10.1109/MCI.2018.2866730
- van den Goorbergh, R., van Smeden, M., Timmerman, D. & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models. *JAMIA*, 29(9), 1525–1534. https://doi.org/10.1093/jamia/ocac093
