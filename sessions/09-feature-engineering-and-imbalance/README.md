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
- *Practice:* build date, interaction and target-encoded features for the BTI decisions (first word of the description, chapter-85 question, time-based validation) → [05-case-study-decision-features.ipynb](workbooks/05-case-study-decision-features.ipynb)

**1:00–1:45 · Aggregates, leakage and text statistics** ([theory page 2](theory/02-aggregates-leakage-and-text-statistics.md))

- [Aggregates from earlier decisions, computed only from past data](theory/02-aggregates-leakage-and-text-statistics.md#1-aggregates-computed-only-from-past-data)
- [Target leakage through aggregates and late information](theory/02-aggregates-leakage-and-text-statistics.md#3-target-leakage-through-aggregates-and-late-information)
- [Simple text statistics as features](theory/02-aggregates-leakage-and-text-statistics.md#2-simple-text-statistics-as-features)
- *Practice:* case study: detect the leaking `classification_justification` (it names the heading in 70 % of the decisions and is missing in the test data) and replace a leaky description lookup by a past-only one → [06-case-study-justification-leakage.ipynb](workbooks/06-case-study-justification-leakage.ipynb)

**2:00–2:45 · Class imbalance** ([theory page 3](theory/03-class-imbalance.md))

- [Class imbalance and why accuracy misleads](theory/03-class-imbalance.md#1-class-imbalance-and-why-accuracy-misleads)
- [Random undersampling and oversampling](theory/03-class-imbalance.md#2-random-undersampling-and-oversampling)
- [Synthetic oversampling (SMOTE)](theory/03-class-imbalance.md#3-synthetic-oversampling-smote)
- [Class weights as an alternative](theory/03-class-imbalance.md#4-class-weights-as-an-alternative)
- [Resampling inside the cross-validation only](theory/03-class-imbalance.md#5-resampling-inside-the-cross-validation-only), never on validation or test data
- [Pipelines with imbalanced-learn](theory/03-class-imbalance.md#6-pipelines-with-imbalanced-learn)
- *Practice:* case study: compare undersampling, oversampling, SMOTE and class weights for rare headings (the long tail of 1,114 headings and the six footwear headings) by macro-F1 → [12-case-study-imbalance.ipynb](workbooks/12-case-study-imbalance.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-dates-interactions-and-categories.md](theory/01-dates-interactions-and-categories.md) | Date features, interactions, rare categories (first word), target encoding, custom transformers | 1 | core |
| [theory/02-aggregates-leakage-and-text-statistics.md](theory/02-aggregates-leakage-and-text-statistics.md) | Past-only aggregates of repeated descriptions, text statistics, the justification leak | 2 | core |
| [theory/03-class-imbalance.md](theory/03-class-imbalance.md) | Long tail of headings, resampling, SMOTE, class weights, imbalanced-learn pipelines (footwear task) | 3 | core |
| [workbooks/01-time-related-feature-engineering.ipynb](workbooks/01-time-related-feature-engineering.ipynb) | Calendar features, one-hot vs cyclical vs spline encodings (scikit-learn example, bike sharing) | 1 | optional |
| [workbooks/02-column-transformer-mixed-types.ipynb](workbooks/02-column-transformer-mixed-types.ipynb) | ColumnTransformer recap for mixed types (scikit-learn example; its feature-selection step is not part of the course) | 1 | optional |
| [workbooks/03-target-encoder.ipynb](workbooks/03-target-encoder.ipynb) | `TargetEncoder` compared with ordinal and one-hot encoding (scikit-learn example) | 1 | core |
| [workbooks/04-target-encoder-cross-fitting.ipynb](workbooks/04-target-encoder-cross-fitting.ipynb) | Why `TargetEncoder` cross-fits: the leak of naive encoding (scikit-learn example) | 1 | core |
| [workbooks/05-case-study-decision-features.ipynb](workbooks/05-case-study-decision-features.ipynb) | **Practice 1:** date, interaction and target-encoded first-word features for the decisions (own) | 1 | core |
| [workbooks/06-case-study-justification-leakage.ipynb](workbooks/06-case-study-justification-leakage.ipynb) | **Practice 2:** the justification leak and past-only description lookups (own) | 2 | core |
| [workbooks/07-imbalanced-classes-impact.ipynb](workbooks/07-imbalanced-classes-impact.ipynb) | Effect of imbalance and of each remedy on the Adult census data (imbalanced-learn example) | 3 | core |
| [workbooks/08-smote-sample-generation.ipynb](workbooks/08-smote-sample-generation.ipynb) | How SMOTE generates one new sample (imbalanced-learn example) | 3 | optional |
| [workbooks/09-over-sampling-comparison.ipynb](workbooks/09-over-sampling-comparison.ipynb) | Random oversampling, SMOTE, ADASYN and variants compared (imbalanced-learn example) | 3 | optional |
| [workbooks/10-under-sampling-comparison.ipynb](workbooks/10-under-sampling-comparison.ipynb) | Undersampling methods compared (imbalanced-learn example) | 3 | optional |
| [workbooks/11-imblearn-pipeline.ipynb](workbooks/11-imblearn-pipeline.ipynb) | A sampler inside an imbalanced-learn `Pipeline` (imbalanced-learn example) | 3 | core |
| [workbooks/12-case-study-imbalance.ipynb](workbooks/12-case-study-imbalance.ipynb) | **Practice 3:** imbalance strategies for rare headings, linear model and random forest (own) | 3 | core |

Third-party sources and licences: [source.md](source.md).

## Before and after the session

**Preparation.** Re-read the Session 8 pages on `Pipeline`, `ColumnTransformer` and macro-F1. Make sure `case-study/data/` exists (`uv run python case-study/prepare_data.py`) and that `train.parquet` loads; the leakage notebook uses the full training table (309,529 decisions).

**Team project until the next session.** Feature set for the project model, documented in the repository: for every feature, its source, the moment at which it is known, and how it is computed inside the pipeline.

**Further reading (optional).**

- imbalanced-learn developers. [*Common pitfalls and recommended practices*](https://imbalanced-learn.org/stable/common_pitfalls.html).
- scikit-learn developers. [*Common pitfalls: data leakage*](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- Kuhn, M. & Johnson, K. (2019). [*Feature Engineering and Selection*](https://bookdown.org/max/FES/), chapters 5 (categorical predictors) and 8 (handling missing data) — free online.
- World Customs Organization. [*What is the Harmonized System (HS)?*](https://www.wcoomd.org/en/topics/nomenclature/overview/what-is-the-harmonized-system.aspx): the nomenclature behind the case-study headings.
- Kapoor, S. & Narayanan, A. (2023). [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804). *Patterns* 4(9).

## Setup

Everything runs in the course environment (`uv sync` in the repository root, then `uv run jupyter lab`); it includes `imbalanced-learn`. The scikit-learn and imbalanced-learn example notebooks download their data from OpenML on first run (internet needed). The own notebooks find `case-study/data/` from any working directory inside the repository; `05-case-study-decision-features.ipynb` takes about two minutes, the other two about one minute each.
