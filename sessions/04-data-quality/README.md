# Session 4 · Data quality and preparation

> **Guiding question.** Can we trust the data, and how do we prepare it for analysis?

## 1 · Data quality checks and missing values

[Open notebook](workbooks/01-lecture-data-quality-checks-and-missing-values.ipynb)

- Dimensions of data quality: completeness, validity, uniqueness, consistency, accuracy, timeliness.
- Checks for types, ranges, duplicates and consistency, recorded as a report.
- Validation rules as pytest tests and pandera schemas that stop the pipeline.
- Missing data mechanisms (MCAR, MAR, MNAR): why a third of the prices are missing.
- Simple, KNN and iterative imputation compared on hidden bedroom counts.
- Worked example: a data quality report for the Berlin listings.

## 2 · Outliers, transformations and a cleaning pipeline

[Open notebook](workbooks/02-lecture-outliers-transformations-and-cleaning-pipeline.ipynb)

- Univariate outliers: IQR rule, z-score and median absolute deviation on price and minimum stay.
- Multivariate outliers with the classical and robust Mahalanobis distance.
- Transformations: logarithm, Box–Cox, Yeo–Johnson and scaling.
- A documented cleaning pipeline: rule, rows, action and reason for every step.
- Worked example: the cleaned table of Berlin listings with its cleaning log, validated by tests.

## Team project until the next session

Data quality report of the decisions and project charter: use of the suggestions, metric, baseline, validation by time, inputs that may not be used.

## Setup

- Airbnb data: `uv run python case-study/prepare_airbnb.py` (the notebooks also read the raw files it keeps in `case-study/data/raw/airbnb/`).
- Validation tests: `uv run pytest sessions/04-data-quality/workbooks/quality -v`; set `AIRBNB_CLEAN_OUT` to the cleaned table of block 2 to run the tests on it too.
- Extra 06 only: `uv run --with empiricaldist --with statadict jupyter lab`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
