# Session 4 · Data quality and preparation: validation, missing values, outliers and transformations

> [!NOTE]
> **Guiding question.** Can we trust the data, and how do we prepare it for analysis?

**Learning outcomes.** You are able to

- check data quality systematically and express the checks as code
- analyse missing values and choose an imputation method
- detect and treat outliers and apply suitable transformations, documenting each decision

## Session plan

**0:00–0:45 · Data quality checks** ([theory page](theory/01-data-quality-checks.md))

- [Dimensions of data quality](theory/01-data-quality-checks.md#dimensions-of-data-quality)
- [Checks for types, ranges, duplicates and consistency](theory/01-data-quality-checks.md#checks-for-types-ranges-duplicates-and-consistency)
- [Validation rules as tests](theory/01-data-quality-checks.md#validation-rules-as-tests)
- *Practice:* write a data quality report for the review data → [workbook 03](workbooks/03-case-study-quality-report.ipynb), [tests](workbooks/quality/test_review_quality.py)

**1:00–1:45 · Missing values and univariate outliers** ([theory page](theory/02-missing-values-and-univariate-outliers.md))

- [Missing data mechanisms (MCAR, MAR, MNAR)](theory/02-missing-values-and-univariate-outliers.md#missing-data-mechanisms-mcar-mar-mnar)
- [Simple, KNN and iterative imputation](theory/02-missing-values-and-univariate-outliers.md#simple-knn-and-iterative-imputation)
- [Missing-value indicators](theory/02-missing-values-and-univariate-outliers.md#missing-value-indicators)
- [Univariate outliers (IQR rule, z-score, median absolute deviation)](theory/02-missing-values-and-univariate-outliers.md#univariate-outliers-iqr-rule-z-score-median-absolute-deviation)
- *Practice:* is a missing product price related to the number of reviews? Compare imputation methods → [workbook 08](workbooks/08-case-study-missing-prices.ipynb)

**2:00–2:45 · Multivariate outliers, transformations and a cleaning pipeline** ([theory page](theory/03-multivariate-outliers-transformations-pipeline.md))

- [Multivariate outliers with the Mahalanobis distance](theory/03-multivariate-outliers-transformations-pipeline.md#multivariate-outliers-with-the-mahalanobis-distance) (model-based detection follows in Session 11)
- [Transformations (logarithm, Box–Cox, Yeo–Johnson, scaling)](theory/03-multivariate-outliers-transformations-pipeline.md#transformations-logarithm-boxcox-yeojohnson-scaling)
- [A documented cleaning pipeline](theory/03-multivariate-outliers-transformations-pipeline.md#a-documented-cleaning-pipeline)
- *Practice:* case study: produce the cleaned review table with a log of the cleaning decisions → [workbook 14](workbooks/14-case-study-cleaned-review-table.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-data-quality-checks.md](theory/01-data-quality-checks.md) | Dimensions, checks as code, validation rules as tests | 1 | core |
| [theory/02-missing-values-and-univariate-outliers.md](theory/02-missing-values-and-univariate-outliers.md) | MCAR/MAR/MNAR, imputation, indicators, IQR/z-score/MAD | 2 | core |
| [theory/03-multivariate-outliers-transformations-pipeline.md](theory/03-multivariate-outliers-transformations-pipeline.md) | Mahalanobis distance, transformations, cleaning pipeline | 3 | core |
| [workbooks/01-try-pandera.ipynb](workbooks/01-try-pandera.ipynb) | pandera: schemas, runtime validation, failure cases | 1 | core |
| [workbooks/02-data-cleaning-missing-and-duplicates.ipynb](workbooks/02-data-cleaning-missing-and-duplicates.ipynb) | Microsoft Data Science for Beginners: missing values and duplicates in pandas | 1 | optional |
| [workbooks/03-case-study-quality-report.ipynb](workbooks/03-case-study-quality-report.ipynb) | **Case study**: data quality report with checks as code and a pandera schema | 1 | core |
| [workbooks/quality/test_review_quality.py](workbooks/quality/test_review_quality.py) | Validation rules as pytest tests | 1, 3 | core |
| [workbooks/04-missing-values-in-pandas.ipynb](workbooks/04-missing-values-in-pandas.ipynb) | Python Data Science Handbook: `None` and `NaN`, `isnull`, `dropna`, `fillna` | 2 | optional |
| [workbooks/05-imputation-methods-compared.ipynb](workbooks/05-imputation-methods-compared.ipynb) | scikit-learn: simple, KNN and iterative imputation compared | 2 | core |
| [workbooks/06-iterative-imputer-variants.ipynb](workbooks/06-iterative-imputer-variants.ipynb) | scikit-learn: IterativeImputer with different estimators | 2 | optional |
| [workbooks/07-distributions-and-outliers.ipynb](workbooks/07-distributions-and-outliers.ipynb) | Think Stats ch. 2: distributions, spotting outliers | 2 | optional |
| [workbooks/08-case-study-missing-prices.ipynb](workbooks/08-case-study-missing-prices.ipynb) | **Case study**: missing prices vs popularity, imputation compared, IQR/z/MAD | 2 | core |
| [workbooks/09-mahalanobis-and-robust-covariance.ipynb](workbooks/09-mahalanobis-and-robust-covariance.ipynb) | scikit-learn: classical vs robust (MCD) Mahalanobis distances | 3 | core |
| [workbooks/10-scalers-and-transformers.ipynb](workbooks/10-scalers-and-transformers.ipynb) | scikit-learn: all scalers and transformers on data with outliers | 3 | core |
| [workbooks/11-box-cox-and-yeo-johnson.ipynb](workbooks/11-box-cox-and-yeo-johnson.ipynb) | scikit-learn: Box–Cox, Yeo–Johnson and quantile transformation | 3 | core |
| [workbooks/12-transforming-the-target.ipynb](workbooks/12-transforming-the-target.ipynb) | scikit-learn: transforming the target of a regression | 3 | optional |
| [workbooks/13-why-scaling-matters.ipynb](workbooks/13-why-scaling-matters.ipynb) | scikit-learn: effect of scaling on k-nearest neighbours and PCA | 3 | optional |
| [workbooks/14-case-study-cleaned-review-table.ipynb](workbooks/14-case-study-cleaned-review-table.ipynb) | **Case study**: cleaned review table and cleaning log | 3 | core |

Origins and licences of third-party files: [source.md](source.md). Workbooks 04 (text CC-BY-NC-ND) and 07 (text CC-BY-NC-SA) may be used for non-commercial teaching only.

## Before and after the session

**Preparation**

- Make sure `case-study/data/` exists (see [case-study/README.md](../../case-study/README.md)).
- Read the [pandas user guide on missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) (about 20 minutes).
- Look at the data quality findings you noticed in Session 3 and bring them along.

**Team project until the next session.** Project charter: question, stakeholder, emphasis (analytics or machine learning), metric, baseline, data loaded.

**Further reading (optional)**

- [Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd/) (van Buuren), chapters 1–2: mechanisms and why simple fixes fail.
- [scikit-learn: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) and [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
- [NIST e-Handbook: Detection of outliers](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm).
- [Feature Engineering and Selection](https://feat.engineering/) (Kuhn and Johnson), chapters 6 and 8.

## Setup

The case-study notebooks and the tests run in the course environment from the repository root (`uv sync`, then `uv run jupyter lab`).

Extra packages for some workbooks:

```bash
uv run --with pandera --with empiricaldist --with statadict jupyter lab
# pandera: workbooks 01 and 03 (section 5); empiricaldist, statadict: workbook 07
```

Run the validation tests from the repository root:

```bash
uv run --with pytest pytest sessions/04-data-quality/workbooks/quality -v
```

Before workbook 14 has been run, the tests on the cleaned table are skipped; afterwards they must pass. Workbook 14 writes `case-study/data/reviews_clean.parquet` (about 75 MB) and `cleaning_log.csv`; this folder is not part of the repository. Set `CLEAN_OUT=/some/path/reviews_clean.parquet` to write elsewhere (the tests read the same variable).

> [!NOTE]
> With pandas 3, three cells of workbook 04 that use `fillna(method="ffill")` fail; write `data.ffill()` and `data.bfill()` instead. The first `.sum()` error in workbooks 02 and 04 is intended: it shows that `None` cannot be summed.
