# Course datasets

| Dataset | Sessions | Script |
|---|---|---|
| Inside Airbnb, Berlin, with Berlin weather from Open-Meteo | 1–12 | `prepare_airbnb.py` |
| IBM Telco customer churn | 6, 8–11 | loaded in the notebooks from IBM's GitHub repository (Apache 2.0) |
| EBTI, EU customs decisions, with the class leaderboard | 13–16 | `prepare_data.py`, `score.py` |

## Inside Airbnb, Berlin (Sessions 1–12)

Sessions 1–12 (Python, classes, SQL and Polars, data quality, statistics, regression, validation, features, tree-based models, clustering, demand over time) use the Berlin listings of [Inside Airbnb](https://insideairbnb.com/get-the-data/): about 12,800 listings with price, district, room type, size, ratings, availability and registration number, plus the number of reviews per listing and month since 2009 as a proxy for demand.

```bash
uv run python case-study/prepare_airbnb.py   # downloads the latest Berlin snapshot (about 100 MB, once) and the weather
# also load the tables into PostgreSQL (Session 3):
uv run python case-study/prepare_airbnb.py --postgres postgresql+psycopg://postgres:course@localhost/postgres
```

| Table | Rows | Content |
|---|---|---|
| `airbnb/listings.parquet` | ~12,800 | location (district, neighbourhood, coordinates), property, price per night, minimum nights, availability, reviews and ratings, host, registration number |
| `airbnb/calendar.parquet` | ~4.7 M | availability of each listing for the next 365 days |
| `airbnb/reviews_monthly.parquet` | ~227,000 | reviews per listing and month since 2009 |
| `airbnb/weather_daily.parquet` | ~3,800 days | daily Berlin weather since 2016 from the Open-Meteo archive (temperature, rain, sunshine); Session 2 shows how to fetch it from the API yourself |
| `airbnb/neighbourhoods.geojson` | | district boundaries |

The script removes personal data: host names, profile texts, photos, review texts, reviewer names, and names that hosts entered in the registration field (`license` keeps only registration numbers; `license_status` gives the type of entry). Inside Airbnb publishes a new snapshot every quarter, so numbers in the materials (based on the snapshot of 26 June 2026) can differ slightly from yours.

## EBTI: classifying goods in the EU customs tariff (Sessions 13–16)

Every product that crosses the EU border is classified in the customs nomenclature, and the classification decides the duty. Traders who are unsure can ask a customs authority for a **Binding Tariff Information (BTI)** decision: they describe the product, and customs states its code. The European Commission publishes all decisions in the [EBTI database](https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en).

**Task of Sessions 13–16:** predict the four-digit **HS heading** of a decision (for example `6404`, footwear with textile uppers) from its description of goods.

The descriptions are written in the language of the issuing country: about 57 % German, 16 % French, and the rest in more than 20 other EU languages. Each training decision also has English keywords, which help you read a decision you cannot read in its original language.

### Tables

| Table | Rows | Content |
|---|---|---|
| `train` | 309,529 decisions, 2017–2023 | bti_reference, issuing_country, language, start_date, end_date, date_of_issue, status, invalidation_reason, description, keywords, classification_justification, cn_code, **heading** (the label), chapter |
| `train_sample` | 50,000 | random sample of `train`, for quick experiments |
| `test` | 113,188 decisions, 2024–2026 | id, issuing_country, language, start_date, description: what a trader's request contains, plus where and when it was decided |
| `nomenclature` | 1,229 headings | heading, heading_description, chapter, chapter_description, section, section_name (English, HS 2022) |
| `monthly_counts` | 107,801 | number of decisions per month, issuing country and chapter, 2004–2026 |

### Preparing the data

The data are downloaded from the official source by each student (about 400 MB, once):

```bash
uv run python case-study/prepare_data.py
# also load the tables into PostgreSQL (Session 15):
uv run python case-study/prepare_data.py --postgres postgresql+psycopg://postgres:course@localhost/postgres
```

The script writes the tables as Parquet files to `case-study/data/` (ignored by Git).

### Leaderboard

- **Task:** predict the heading of each test decision.
- **Split by time:** training decisions start in 2017–2023. The public leaderboard uses 2024, the private leaderboard 2025–2026. Test decisions whose description repeats a training description word for word are removed.
- **Metric:** accuracy (share of correct headings), with macro-F1 reported alongside. With more than 1,000 headings and a long tail of rare ones, the two tell different stories; Session 13 explains why.
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
| L1 | 13 | TF-IDF text classifier |
| L2 | 14 | embeddings or a language model, any method of the session |
| L3 | 16 | final submission after retraining with the released 2024 labels, any method, with a one-page description |

The leaderboard task is also the final project of all teams. Decisions can be looked up in the public EBTI database, so the score is not converted into marks: the project is graded on how a team frames, handles, solves and explains the problem, and its final submission must be reproducible from the team's repository. In Session 16 the lecturer releases the 2024 labels as feedback data (`data/feedback_2024.csv`); the 2025–2026 decisions stay hidden.

### Things to watch out for

- **Leakage through the justification.** `classification_justification` explains the decision and names the heading in most cases. It exists only in the training data, because it is written after classification. Never use it as a model input; Session 13 uses it as an example of a feature that is not available at prediction time. The same holds for `keywords`, `cn_code`, `status` and the end dates.
- **Quoted codes.** Some descriptions quote the tariff text of their own code. Numbers that repeat the decision's own code are replaced by `<CODE>` in all tables.
- **Changing nomenclature.** The HS nomenclature is revised every five years (latest: 2022). Some headings were created, split or deleted, which changes the labels over time (Session 16).
- **Data quality.** Annulled decisions carry the placeholder end date 1900-01-01, the raw export contains a few impossible start dates (years such as 2055 or 2200), some descriptions are database templates or test entries, and a few headings no longer exist in HS 2022.

## Sources and terms of use

- European Commission, *European Binding Tariff Information (EBTI)* database, full export, retrieved by `prepare_data.py`. Reuse is permitted with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU). The holders of the decisions are not published.
- Open-Meteo, historical weather API, [open-meteo.com](https://open-meteo.com), Creative Commons Attribution 4.0 (CC BY 4.0); free for non-commercial use.
- Inside Airbnb, Berlin snapshot (listings, calendar, reviews), [insideairbnb.com](https://insideairbnb.com/get-the-data/), Creative Commons Attribution 4.0 (CC BY 4.0). The data are scraped from public Airbnb pages; report results in aggregate and do not name hosts.
- Harmonized System nomenclature (HS 2022, English), [datasets/harmonized-system](https://github.com/datasets/harmonized-system), ODC Public Domain Dedication and Licence (PDDL).
