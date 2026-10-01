# Running case study: classifying goods in the EU customs tariff

Every product that crosses the EU border is classified in the customs nomenclature, and the classification decides the duty. Traders who are unsure can ask a customs authority for a **Binding Tariff Information (BTI)** decision: they describe the product, and customs states its code. The European Commission publishes all decisions in the [EBTI database](https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en).

**Task of the course:** predict the four-digit **HS heading** of a decision (for example `6404`, footwear with textile uppers) from its description of goods.

The descriptions are written in the language of the issuing country: about 57 % German, 16 % French, and the rest in 20 other EU languages. Each training decision also has English keywords, which help you read a decision you cannot read in its original language.

## Tables

| Table | Rows | Content |
|---|---|---|
| `train` | 309,529 decisions, 2017–2023 | bti_reference, issuing_country, language, start_date, end_date, date_of_issue, status, invalidation_reason, description, keywords, classification_justification, cn_code, **heading** (the label), chapter |
| `train_sample` | 50,000 | random sample of `train`, for quick experiments |
| `test` | 113,188 decisions, 2024–2026 | id, issuing_country, language, start_date, description: what a trader's request contains, plus where and when it was decided |
| `nomenclature` | 1,229 headings | heading, heading_description, chapter, chapter_description, section, section_name (English, HS 2022) |
| `monthly_counts` | 107,801 | number of decisions per month, issuing country and chapter, 2004–2026 |

## Preparing the data

The data are downloaded from the official source by each student (about 400 MB, once):

```bash
uv run python case-study/prepare_data.py
# also load the tables into PostgreSQL (Session 3):
uv run python case-study/prepare_data.py --postgres postgresql+psycopg://postgres:course@localhost/postgres
```

The script writes the tables as Parquet files to `case-study/data/` (ignored by Git).

## Leaderboard

- **Task:** predict the heading of each test decision.
- **Split by time:** training decisions start in 2017–2023. The public leaderboard uses 2024, the private leaderboard 2025–2026. Test decisions whose description repeats a training description word for word are removed.
- **Metric:** accuracy (share of correct headings), with macro-F1 reported alongside. With more than 1,000 headings and a long tail of rare ones, the two tell different stories; Session 8 explains why.
- **Submission:** a CSV file with the columns `id,heading` and one row per test decision. Score it locally:
  ```bash
  uv run python case-study/score.py submission.csv
  ```

**Reference values (public leaderboard, 2024):** on the private years 2025–2026 the same models score about 3 points lower, an effect of drift (Session 16). The linear SVM is scikit-learn's `SGDClassifier(loss="hinge")`.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always the most frequent heading (3926, other articles of plastics) | 0.041 | 0.000 |
| Logistic regression on simple features (length, language, country) | 0.076 | 0.002 |
| Word TF-IDF + linear model, 50,000-decision sample | 0.806 | 0.517 |
| Word TF-IDF + linear model, full training set | 0.872 | 0.682 |
| Character TF-IDF (3–5) + linear model, full training set | 0.882 | 0.695 |

**Rounds:**

| Round | Session | Model |
|---|---|---|
| L1 | 8 | logistic regression on simple features |
| L2 | 10 | tree-based model on engineered features |
| L3 | 13 | TF-IDF text classifier |
| L4 | 16 | final submission after retraining with the released 2024 labels, any method, with a one-page description |

The leaderboard is not graded. Decisions can be looked up in the public EBTI database, so a high rank earns no marks; what counts is that methods are applied and compared correctly. In Session 16 the lecturer releases the 2024 labels as feedback data (`data/feedback_2024.csv`); the 2025–2026 decisions stay hidden.

## Things to watch out for

- **Leakage through the justification.** `classification_justification` explains the decision and names the heading in most cases. It exists only in the training data, because it is written after classification. Never use it as a model input; Session 9 uses it as an example of a feature that is not available at prediction time. The same holds for `keywords`, `cn_code`, `status` and the end dates.
- **Quoted codes.** Some descriptions quote the tariff text of their own code. Numbers that repeat the decision's own code are replaced by `<CODE>` in all tables.
- **Changing nomenclature.** The HS nomenclature is revised every five years (latest: 2022). Some headings were created, split or deleted, which changes the labels over time (Session 16).
- **Data quality.** Dates contain typing errors (end dates before start dates, start dates far in the future); Session 4 deals with them.

## Sources and terms of use

- European Commission, *European Binding Tariff Information (EBTI)* database, full export, retrieved by `prepare_data.py`. Reuse is permitted with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU). The holders of the decisions are not published.
- Harmonized System nomenclature (HS 2022, English), [datasets/harmonized-system](https://github.com/datasets/harmonized-system), ODC Public Domain Dedication and Licence (PDDL).
