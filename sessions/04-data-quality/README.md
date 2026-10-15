# Session 4 · Data quality and preparation

> **Guiding question.** Can we trust the data, and how do we prepare it for analysis?

## 1 · Data quality checks and missing values

[Open notebook](workbooks/01-lecture-data-quality-checks-and-missing-values.ipynb)

- Dimensions of data quality: completeness, validity, uniqueness, consistency, accuracy, timeliness; a first look at a table and how pandas stores a missing value.
- Checks for types, ranges, duplicates and consistency, recorded as a report, including messy categories, near-duplicates and checks across tables.
- Validation rules as pytest tests and pandera schemas (models, runtime checks, checkpoints) that stop the pipeline.
- Missing data mechanisms (MCAR, MAR, MNAR): a simulation, why a third of the prices are missing, a test and a model for missingness.
- Dropping, filling and imputation: `dropna` and `fillna`, simple, KNN and iterative imputation compared on hidden bedroom counts, imputation before an estimator.
- Worked example: a data quality report for the Berlin listings.

## 2 · Outliers, transformations and a cleaning pipeline

[Open notebook](workbooks/02-lecture-outliers-transformations-and-cleaning-pipeline.ipynb)

- Univariate outliers: IQR rule, z-score and median absolute deviation on price and minimum stay; frequency tables and what to do with a flagged value.
- Multivariate outliers with the classical and robust Mahalanobis distance, on simulated data and on prices for the number of guests.
- Transformations: logarithm, Box–Cox, Yeo–Johnson and scaling, why scaling matters, and transforming the target.
- A documented cleaning pipeline: rule, rows, action and reason for every step; preprocessing pipelines in scikit-learn.
- Worked example: the cleaned table of Berlin listings with its cleaning log, validated by tests.

## Team project until the next session

Data quality report of the decisions and project charter: use of the suggestions, metric, baseline, validation on `val.parquet` (the course split is random, not by time: say what that means for your estimate), inputs that may not be used.

## Setup

- Airbnb data: download `airbnb.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- Validation tests: `uv run pytest sessions/04-data-quality/workbooks/quality -v`; set `AIRBNB_CLEAN_OUT` to the cleaned table of block 2 to run the tests on it too.
- Extra 06 only: `uv run --with empiricaldist --with statadict jupyter lab`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
