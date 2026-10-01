# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/01-try-pandera.ipynb](workbooks/01-try-pandera.ipynb) | pandera schemas (`DataFrameModel`, `DataFrameSchema`), runtime checks with `check_types`, failure cases, schemas as quality checkpoints | [unionai-oss/pandera, docs/source/notebooks](https://raw.githubusercontent.com/unionai-oss/pandera/main/docs/source/notebooks/try_pandera.ipynb) | [MIT](https://github.com/unionai-oss/pandera/blob/main/LICENSE.txt) | 2026-10-01 | Renamed from `try_pandera.ipynb`; content unchanged |
| [workbooks/02-data-cleaning-missing-and-duplicates.ipynb](workbooks/02-data-cleaning-missing-and-duplicates.ipynb) | Data cleaning: detecting/handling missing values, duplicates | [microsoft/Data-Science-For-Beginners](https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/2-Working-With-Data/08-data-preparation/notebook.ipynb) | [MIT](https://github.com/microsoft/Data-Science-For-Beginners/blob/main/LICENSE) | 2026-09-26 | Renamed from `msft-dsfb_08-data-preparation.ipynb`. 2026-10-01: `fillna(method='ffill')` / `fillna(method='bfill')` (removed in pandas 3) replaced by `ffill()` / `bfill()` in 3 code cells and 1 code comment. The cell `example1.sum()` raises a `TypeError` on purpose |
| [workbooks/04-missing-values-in-pandas.ipynb](workbooks/04-missing-values-in-pandas.ipynb) | None/NaN in pandas, isnull, dropna, fillna | [jakevdp/PythonDataScienceHandbook](https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/03.04-Missing-Values.ipynb) | [MIT (code)](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/LICENSE-CODE) / [CC-BY-NC-ND 3.0 (text)](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/LICENSE-TEXT) | 2026-09-26 | Renamed from `pdsh_03.04-missing-values.ipynb` (was `03-...` until 2026-10-01); content unchanged (the text licence does not allow changes). **NC-ND**: non-commercial use only. With pandas 3, the three `fillna(method=...)` cells fail; use `ffill()` / `bfill()` instead. `vals1.sum()` raises a `TypeError` on purpose |
| [workbooks/05-imputation-methods-compared.ipynb](workbooks/05-imputation-methods-compared.ipynb) | SimpleImputer (mean/zero), KNNImputer, IterativeImputer compared | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/a440a8b10138c855100ed5820fdb36b6/plot_missing_values.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-missing-values.ipynb` (was `04-...`); content unchanged |
| [workbooks/06-iterative-imputer-variants.ipynb](workbooks/06-iterative-imputer-variants.ipynb) | IterativeImputer (MICE-style) with different estimators | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/067cd5d39b097d2c49dd98f563dac13a/plot_iterative_imputer_variants_comparison.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-iterative-imputer-variants.ipynb` (was `05-...`); content unchanged |
| [workbooks/07-distributions-and-outliers.ipynb](workbooks/07-distributions-and-outliers.ipynb) | Frequency tables, spotting outliers, effect size (Cohen's d) | [AllenDowney/ThinkStats (3e)](https://raw.githubusercontent.com/AllenDowney/ThinkStats/v3/nb/chap02.ipynb) | [MIT (code) / CC-BY-NC-SA 4.0 (text)](https://github.com/AllenDowney/ThinkStats/blob/v3/LICENSE) | 2026-09-26 | Renamed from `thinkstats_ch02-distributions-outliers.ipynb` (was `01-...`); content unchanged. **NC**: non-commercial use only. Downloads its data and helper modules on first run; needs `empiricaldist` and `statadict` |
| [workbooks/09-mahalanobis-and-robust-covariance.ipynb](workbooks/09-mahalanobis-and-robust-covariance.ipynb) | Robust (MCD) vs empirical covariance Mahalanobis distances | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/83d33d2afcbf708f386433bb1abb0785/plot_mahalanobis_distances.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-mahalanobis-distances.ipynb` (was `06-...`); content unchanged |
| [workbooks/10-scalers-and-transformers.ipynb](workbooks/10-scalers-and-transformers.ipynb) | Standard/MinMax/MaxAbs/Robust scalers, PowerTransformer, QuantileTransformer, Normalizer | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/e60e99adef360baabc49b925646a39d9/plot_all_scaling.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-all-scaling.ipynb` (was `07-...`); content unchanged |
| [workbooks/11-box-cox-and-yeo-johnson.ipynb](workbooks/11-box-cox-and-yeo-johnson.ipynb) | PowerTransformer Box-Cox / Yeo-Johnson vs QuantileTransformer | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/fc26dc0b9d906466315a49a860435409/plot_map_data_to_normal.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-map-data-to-normal.ipynb` (was `08-...`); content unchanged |
| [workbooks/12-transforming-the-target.ipynb](workbooks/12-transforming-the-target.ipynb) | TransformedTargetRegressor: log/quantile transform of the target | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/ea57d7ab1588de8f5bd1afc68f20de2f/plot_transformed_target.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-transformed-target.ipynb` (was `09-...`); content unchanged |
| [workbooks/13-why-scaling-matters.ipynb](workbooks/13-why-scaling-matters.ipynb) | Why scaling matters (KNN, PCA) | [scikit-learn example gallery 1.9](https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/main/1.9/_downloads/c9688d36cfbf43a68f3613b58110ceaa/plot_scaling_importance.ipynb) | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | 2026-09-26 | Renamed from `sklearn_plot-scaling-importance.ipynb` (was `10-...`); content unchanged |

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-data-quality-checks.md](theory/01-data-quality-checks.md) | Block 1: dimensions of data quality, checks for types/ranges/duplicates/consistency, validation rules as tests | CC-BY-4.0 |
| [theory/02-missing-values-and-univariate-outliers.md](theory/02-missing-values-and-univariate-outliers.md) | Block 2: MCAR/MAR/MNAR, simple/KNN/iterative imputation, missing-value indicators, IQR/z-score/MAD | CC-BY-4.0 |
| [theory/03-multivariate-outliers-transformations-pipeline.md](theory/03-multivariate-outliers-transformations-pipeline.md) | Block 3: Mahalanobis distance (classical and MCD), log/Box–Cox/Yeo–Johnson/scaling, documented cleaning pipeline | CC-BY-4.0 |
| [theory/figures/make_figures.py](theory/figures/make_figures.py) | Script for `missingness-pattern.png`, `univariate-outliers.png`, `box-cox-before-after.png`, `mahalanobis-ellipses.png` | CC-BY-4.0 |
| [workbooks/03-case-study-quality-report.ipynb](workbooks/03-case-study-quality-report.ipynb) | Practice block 1: data quality report for the review data; checks as code; pandera schema | CC-BY-4.0 |
| [workbooks/quality/test_review_quality.py](workbooks/quality/test_review_quality.py) | Validation rules as pytest tests (raw data, known problems as `xfail`, cleaned table) | CC-BY-4.0 |
| [workbooks/08-case-study-missing-prices.ipynb](workbooks/08-case-study-missing-prices.ipynb) | Practice block 2: missing prices vs popularity, chi-square test, missingness model; imputation compared on hidden values; IQR/z/MAD on prices | CC-BY-4.0 |
| [workbooks/14-case-study-cleaned-review-table.ipynb](workbooks/14-case-study-cleaned-review-table.ipynb) | Practice block 3: cleaned review table with a cleaning log (Box–Cox, Yeo–Johnson, MAD and robust Mahalanobis flags, product join) | CC-BY-4.0 |

Author of own material: course team.

## Citations

- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592. https://doi.org/10.1093/biomet/63.3.581
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC. https://stefvanbuuren.name/fimd/
- Sterne, J. A. C., White, I. R., Carlin, J. B., et al. (2009). Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. *BMJ*, 338, b2393. https://doi.org/10.1136/bmj.b2393
- European Medicines Agency (2010). *Guideline on missing data in confirmatory clinical trials* (EMA/CPMP/EWP/1776/99 Rev. 1).
- Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
- NIST/SEMATECH. *e-Handbook of Statistical Methods*. https://www.itl.nist.gov/div898/handbook/
- Rousseeuw, P. J. (1984). Least median of squares regression. *Journal of the American Statistical Association*, 79(388), 871–880.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. *Technometrics*, 41(3), 212–223. https://doi.org/10.1080/00401706.1999.10485670
- Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211–252.
- Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954–959. https://doi.org/10.1093/biomet/87.4.954
- Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A. (2018). Automating large-scale data quality verification. *PVLDB*, 11(12), 1781–1794.
- Breck, E., Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2019). Data validation for machine learning. *MLSys 2019*.
- Bantilan, N. (2020). pandera: Statistical data validation of pandas dataframes. *Proceedings of the 19th Python in Science Conference*, 116–124.
- Herndon, T., Ash, M., & Pollin, R. (2014). Does high public debt consistently stifle economic growth? A critique of Reinhart and Rogoff. *Cambridge Journal of Economics*, 38(2), 257–279.
- ISO/IEC 25012:2008. *Software engineering — Software product Quality Requirements and Evaluation (SQuaRE) — Data quality model*.
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press. https://feat.engineering/
- VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly. https://jakevdp.github.io/PythonDataScienceHandbook/
- Downey, A. B. *Think Stats* (3rd ed.). O'Reilly. https://allendowney.github.io/ThinkStats/
- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging language and items for retrieval and recommendation. arXiv:2403.03952 (course dataset).
