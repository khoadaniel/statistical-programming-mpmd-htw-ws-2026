# Session 9 · Advanced feature engineering and imbalanced data

> [!NOTE]
> **Guiding question.** Which additional inputs make a model better, how do we build them without leakage, and what do we do when one class is rare?

**Learning outcomes.** Students are able to

- construct features from dates, interactions, high-cardinality categories and joined tables
- implement feature construction in pipelines without leakage
- handle class imbalance with undersampling, oversampling and class weights, applied only to the training data

## Session plan

**0:00–0:45 · Dates, interactions and categories** ([theory page 1](theory/01-dates-interactions-and-categories.md))

- [Features from dates and times](theory/01-dates-interactions-and-categories.md#1-features-from-dates-and-times); [interactions between features](theory/01-dates-interactions-and-categories.md#2-interactions-between-features)
- High-cardinality categories: [grouping rare categories](theory/01-dates-interactions-and-categories.md#3-high-cardinality-categories-grouping-rare-categories), [target encoding and its leakage risk](theory/01-dates-interactions-and-categories.md#4-target-encoding-and-its-leakage-risk)
- [Custom transformers in scikit-learn pipelines](theory/01-dates-interactions-and-categories.md#5-custom-transformers-in-scikit-learn-pipelines)
- *Practice:* What drives nightly prices in Berlin, and which constructed features help? Build distance, amenity, review-date and interaction features and a target-encoded neighbourhood for the Airbnb price model and measure each group's gain with host-grouped validation → [05-case-study-airbnb-features.ipynb](workbooks/05-case-study-airbnb-features.ipynb)

**1:00–1:45 · Aggregates, leakage and text statistics** ([theory page 2](theory/02-aggregates-leakage-and-text-statistics.md))

- [Aggregates from joined tables, computed only from past data](theory/02-aggregates-leakage-and-text-statistics.md#1-aggregates-computed-only-from-past-data)
- [Target leakage through aggregates and late information](theory/02-aggregates-leakage-and-text-statistics.md#3-target-leakage-through-aggregates-and-late-information)
- [Simple text statistics as features](theory/02-aggregates-leakage-and-text-statistics.md#2-simple-text-statistics-as-features)
- *Practice:* Which Berlin listings will be busy next year? Build past-only demand features from the monthly reviews, expose the snapshot columns that already contain the target period, and detect the revenue column that leaks the price → [06-case-study-airbnb-leakage.ipynb](workbooks/06-case-study-airbnb-leakage.ipynb); optional second example on the EBTI decisions (justification leak, past-only description lookups) → [07-case-study-ebti-past-only-lookups.ipynb](workbooks/07-case-study-ebti-past-only-lookups.ipynb)

**2:00–2:45 · Class imbalance** ([theory page 3](theory/03-class-imbalance.md))

- [Class imbalance and why accuracy misleads](theory/03-class-imbalance.md#1-class-imbalance-and-why-accuracy-misleads)
- [Random undersampling and oversampling](theory/03-class-imbalance.md#2-random-undersampling-and-oversampling)
- [Synthetic oversampling (SMOTE)](theory/03-class-imbalance.md#3-synthetic-oversampling-smote)
- [Class weights as an alternative](theory/03-class-imbalance.md#4-class-weights-as-an-alternative)
- [Resampling inside the cross-validation only](theory/03-class-imbalance.md#5-resampling-inside-the-cross-validation-only), never on validation or test data
- [Pipelines with imbalanced-learn](theory/03-class-imbalance.md#6-pipelines-with-imbalanced-learn)
- *Practice:* Which customers will leave, and which rare headings does the classifier miss? Compare undersampling, oversampling, SMOTE and class weights with a tuned threshold for Telco churners (precision, recall, F1, PR AUC), and class weights, oversampling and grouping for the long tail of EBTI headings (macro-F1) → [13-case-study-imbalance.ipynb](workbooks/13-case-study-imbalance.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-dates-interactions-and-categories.md](theory/01-dates-interactions-and-categories.md) | Date features, interactions, rare categories, target encoding, custom transformers (Berlin price model) | 1 | core |
| [theory/02-aggregates-leakage-and-text-statistics.md](theory/02-aggregates-leakage-and-text-statistics.md) | Past-only aggregates from monthly reviews, title statistics, the revenue and snapshot leaks; EBTI justification leak in brief | 2 | core |
| [theory/03-class-imbalance.md](theory/03-class-imbalance.md) | Long tail of EBTI headings, resampling, SMOTE, class weights, imbalanced-learn pipelines (Telco churn) | 3 | core |
| [workbooks/01-time-related-feature-engineering.ipynb](workbooks/01-time-related-feature-engineering.ipynb) | Calendar features, one-hot vs cyclical vs spline encodings (scikit-learn example, bike sharing) | 1 | optional |
| [workbooks/02-column-transformer-mixed-types.ipynb](workbooks/02-column-transformer-mixed-types.ipynb) | ColumnTransformer recap for mixed types (scikit-learn example; its feature-selection step is not part of the course) | 1 | optional |
| [workbooks/03-target-encoder.ipynb](workbooks/03-target-encoder.ipynb) | `TargetEncoder` compared with ordinal and one-hot encoding (scikit-learn example) | 1 | core |
| [workbooks/04-target-encoder-cross-fitting.ipynb](workbooks/04-target-encoder-cross-fitting.ipynb) | Why `TargetEncoder` cross-fits: the leak of naive encoding (scikit-learn example) | 1 | core |
| [workbooks/05-case-study-airbnb-features.ipynb](workbooks/05-case-study-airbnb-features.ipynb) | **Practice 1:** dates, distance, interaction, rare property types, target-encoded neighbourhood, custom transformers and a feature-group comparison for the Berlin price model (own) | 1 | core |
| [workbooks/06-case-study-airbnb-leakage.ipynb](workbooks/06-case-study-airbnb-leakage.ipynb) | **Practice 2:** past-only demand features from monthly reviews, out-of-time check, snapshot and revenue leaks, title statistics (own) | 2 | core |
| [workbooks/07-case-study-ebti-past-only-lookups.ipynb](workbooks/07-case-study-ebti-past-only-lookups.ipynb) | The justification leak and past-only description lookups on the EBTI decisions (own) | 2 | optional |
| [workbooks/08-imbalanced-classes-impact.ipynb](workbooks/08-imbalanced-classes-impact.ipynb) | Effect of imbalance and of each remedy on the Adult census data (imbalanced-learn example) | 3 | core |
| [workbooks/09-smote-sample-generation.ipynb](workbooks/09-smote-sample-generation.ipynb) | How SMOTE generates one new sample (imbalanced-learn example) | 3 | optional |
| [workbooks/10-over-sampling-comparison.ipynb](workbooks/10-over-sampling-comparison.ipynb) | Random oversampling, SMOTE, ADASYN and variants compared (imbalanced-learn example) | 3 | optional |
| [workbooks/11-under-sampling-comparison.ipynb](workbooks/11-under-sampling-comparison.ipynb) | Undersampling methods compared (imbalanced-learn example) | 3 | optional |
| [workbooks/12-imblearn-pipeline.ipynb](workbooks/12-imblearn-pipeline.ipynb) | A sampler inside an imbalanced-learn `Pipeline` (imbalanced-learn example) | 3 | core |
| [workbooks/13-case-study-imbalance.ipynb](workbooks/13-case-study-imbalance.ipynb) | **Practice 3:** imbalance strategies for Telco churners and the long tail of EBTI headings (own) | 3 | core |

Third-party sources and licences: [source.md](source.md).

## Before and after the session

**Preparation.** Re-read the Session 7 pages on grouped cross-validation and leakage, and the Session 8 pages on `Pipeline`, `ColumnTransformer`, thresholds and macro-F1. Make sure the case-study data exist: `uv run python case-study/prepare_airbnb.py` (Inside Airbnb, Berlin) and `uv run python case-study/prepare_data.py` (EBTI decisions; the optional notebook 07 needs the full `train.parquet`).

**Team project until the next session.** Feature set for the project model, documented in the repository: for every feature, its source, the moment at which it is known, and how it is computed inside the pipeline.

**Further reading (optional).**

- imbalanced-learn developers. [*Common pitfalls and recommended practices*](https://imbalanced-learn.org/stable/common_pitfalls.html).
- scikit-learn developers. [*Common pitfalls: data leakage*](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- Kuhn, M. & Johnson, K. (2019). [*Feature Engineering and Selection*](https://bookdown.org/max/FES/), chapters 5 (categorical predictors) and 8 (handling missing data) — free online.
- Inside Airbnb. [*Data assumptions*](https://insideairbnb.com/data-assumptions/): how the occupancy and revenue columns are estimated, and why they must not explain the price.
- Kapoor, S. & Narayanan, A. (2023). [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804). *Patterns* 4(9).

## Setup

Everything runs in the course environment (`uv sync` in the repository root, then `uv run jupyter lab`); it includes `imbalanced-learn`. The case-study data are created once with `uv run python case-study/prepare_airbnb.py` and `uv run python case-study/prepare_data.py`. The scikit-learn and imbalanced-learn example notebooks download their data from OpenML on first run, and the Telco data come from GitHub (internet needed). The own notebooks find `case-study/data/` from any working directory inside the repository; each runs in one to two minutes.
