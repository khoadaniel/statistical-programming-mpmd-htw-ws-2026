# Session 9 · Extras and notebook provenance

These are the downloaded source notebooks curated into the block notebook [01-lecture-feature-construction-without-leakage.ipynb](../01-lecture-feature-construction-without-leakage.ipynb); they stay here as original reference copies, numbered in the order of the block's sections.

## Downloaded notebooks

| File | Content | Block | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|---|
| [01-time-related-feature-engineering.ipynb](01-time-related-feature-engineering.ipynb) | Calendar features; ordinal, one-hot, sine/cosine and periodic spline encodings; spline interactions and Nyström kernel (bike-sharing demand) | 1 | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/7012baed63b9a27f121bae611b8285c2/plot_cyclical_feature_engineering.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_cyclical_feature_engineering.ipynb`; content unchanged. Curated into the block notebook, sections 1.3 and 2.5 |
| [02-table-vectorizer-encodings.ipynb](02-table-vectorizer-encodings.ipynb) | skrub `TableVectorizer`: automatic encoding of dates and low- and high-cardinality columns (employee salaries) | 1 | [skrub examples 0.11](https://skrub-data.org/stable/_downloads/8d3cbbdb02601107a780b72d51bfbf27/0010_encodings.ipynb) | [BSD-3-Clause](https://github.com/skrub-data/skrub/blob/main/LICENSE.txt) | 2026-10-14 | Renamed from `0010_encodings.ipynb`; content unchanged. Curated into section 3.4 |
| [03-target-encoder.ipynb](03-target-encoder.ipynb) | `TargetEncoder` compared with dropping, ordinal and one-hot encoding and native categorical support (wine reviews) | 1 | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/7b414ce0c39e11cf961fd4fa23008246/plot_target_encoder.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_target_encoder.ipynb` (was `02-target-encoder.ipynb` until 2026-10-14); content unchanged. Curated into section 4.5 |
| [04-target-encoder-cross-fitting.ipynb](04-target-encoder-cross-fitting.ipynb) | Why `TargetEncoder` cross-fits: a synthetic dataset with informative, shuffled and near-unique categories | 1 | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/c3f95dc25241c64632f9c3378fd4e89b/plot_target_encoder_cross_val.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_target_encoder_cross_val.ipynb` (was `03-target-encoder-cross-fitting.ipynb` until 2026-10-14); content unchanged. Curated into section 4.3 |
| [05-join-aggregation.ipynb](05-join-aggregation.ipynb) | skrub `AggJoiner`: aggregate-then-join of a related table inside a pipeline (credit fraud, baskets and products) | 1 | [skrub examples 0.11](https://skrub-data.org/stable/_downloads/0f7592e23b14eac9523ac79736c2f189/0070_join_aggregation.ipynb) | [BSD-3-Clause](https://github.com/skrub-data/skrub/blob/main/LICENSE.txt) | 2026-10-14 | Renamed from `0070_join_aggregation.ipynb`; content unchanged. Curated into section 5.6 |
| [06-column-transformer-mixed-types.ipynb](06-column-transformer-mixed-types.ipynb) | `ColumnTransformer` for mixed types, column selection by dtype, search over preprocessing parameters (Titanic). Its feature-selection step is not part of the course | 1 | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/26f110ad6cff1a8a7c58b1a00d8b8b5a/plot_column_transformer_mixed_types.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-01 | Renamed from `plot_column_transformer_mixed_types.ipynb` (was `04-column-transformer-mixed-types.ipynb` until 2026-10-14); content unchanged. Curated into section 7.5 |
| [07-column-transformer-heterogeneous-data.ipynb](07-column-transformer-heterogeneous-data.ipynb) | A `FunctionTransformer` that splits e-mails into subject and body, with separate text features per part (20 newsgroups) | 1 | [scikit-learn example gallery 1.9](https://scikit-learn.org/stable/_downloads/15dc6d7a809edf988a7328336a25faec/plot_column_transformer.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-10-14 | Renamed from `plot_column_transformer.ipynb`; content unchanged. Linked from section 7.5 |

Extras 01 (bike sharing), 03 (wine reviews) and 06 (Titanic) download their data from OpenML, extra 02 and 05 through `skrub.datasets`, extra 07 through `sklearn.datasets.fetch_20newsgroups`; all need internet on first run and cache the data outside the repository.

## Authors and licences

- **scikit-learn developers, example gallery:** extras 01, 03, 04, 06 and 07, BSD-3-Clause.
- **skrub developers, examples:** extras 02 and 05, BSD-3-Clause.
- **scikit-learn developers, user guide "Common pitfalls and recommended practices":** the noise feature-selection example adapted in section 6.4 of the block notebook, BSD-3-Clause; not downloaded.

Keep the original attribution and licence notices when reusing these sources. The block notebook marks each curated section with an "Adapted from" line; its course-written parts are CC BY 4.0.

## Course material

| File | Content | Licence |
|---|---|---|
| [../01-lecture-feature-construction-without-leakage.ipynb](../01-lecture-feature-construction-without-leakage.ipynb) | Block notebook: theory, curated depth sections, worked example, exercises, debrief | CC BY 4.0, course team (curated sections BSD-3-Clause, see above) |
| [../../figures/make_figures.py](../../figures/make_figures.py) and `cyclical_month_encoding.png` | Months on the unit circle (figure of section 1.2) | CC BY 4.0, course team |

To regenerate the figure, run `uv run python sessions/09-feature-engineering/figures/make_figures.py` from the repository root.

Until 2026-10-14 the session also had a separate lecture notebook, two worked examples (`02-case-study-airbnb-features.ipynb`, `03-case-study-airbnb-leakage.ipynb`) and an exercise notebook (`session-09-exercises.ipynb`); their content is now part of the block notebook, and git history keeps the originals. The class-imbalance material of the earlier three-block version of this session moved to [Session 8](../../../08-classification-and-metrics/README.md), Block 2. Simple text statistics of the listing titles moved to Session 13.

## Data

- **Inside Airbnb (2026).** *Berlin, Germany*: listings, monthly review counts and calendar, snapshot of 26 June 2026, CC BY 4.0, prepared with `case-study/prepare_airbnb.py` (host names and other personal data removed). https://insideairbnb.com/get-the-data/ · *Data assumptions*: https://insideairbnb.com/data-assumptions/
- **Open-Meteo.** Historical weather API, daily Berlin weather since 2016, CC BY 4.0, fetched by `case-study/prepare_airbnb.py`. https://open-meteo.com

## References cited in the block notebook

- Feast authors (2025). *Point-in-time joins*. Feast documentation. https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Galli, S. (2021). Feature-engine: a Python package for feature engineering for machine learning. *Journal of Open Source Software*, 6(65), 3642. https://doi.org/10.21105/joss.03642
- Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD*, 6(4), 15. https://doi.org/10.1145/2382577.2382579
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press. https://bookdown.org/max/FES/
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations*, 3(1), 27–32. https://doi.org/10.1145/507533.507538
- Pargent, F., Pfisterer, F., Thomas, J. & Bischl, B. (2022). Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features. *Computational Statistics*, 37, 2671–2692. https://doi.org/10.1007/s00180-022-01207-6
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 31*. https://arxiv.org/abs/1706.09516
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *KDD '16*. https://arxiv.org/abs/1602.04938
