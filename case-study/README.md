# Course datasets

| Dataset | Sessions | Folder after download | Introduction |
|---|---|---|---|
| Inside Airbnb, Berlin, with Berlin weather from Open-Meteo | 1–12 | `case-study/data/airbnb/` | [airbnb/README.md](airbnb/README.md) |
| IBM Telco customer churn | 6, 8–11 | none: the notebooks read it from IBM's GitHub repository (Apache 2.0) | |
| EBTI, EU customs decisions: the final project | 13–16 | `case-study/data/ebti/` | [ebti/README.md](ebti/README.md) |

Start with the two introductions: where the data come from, why the course uses them, and a first look with charts. This page covers the folder, the download and what was held back.

## The folder at a glance

```
case-study/
  README.md                 this page: folder, download, what was held back
  airbnb/README.md          introduction to the Airbnb data, with charts
  ebti/README.md            introduction to the EBTI data, with charts
  evaluate_api.py           calls a heading API on labelled decisions and scores it (final project)
  make_dataset_figures.py   draws the charts of the two introductions

  data/                     YOUR DATA: unzip the course-drive files here (empty in the repository)
    airbnb/                   from airbnb-v1.zip
    ebti/                     from ebti-v1.zip
```

The only folder you change is `data/`.

## Getting the data

Download `airbnb-v1.zip` and `ebti-v1.zip` from the **course drive** and unzip them **inside `case-study/data/`**, so that you get:

```
case-study/data/
  airbnb/     from airbnb-v1.zip (about 3 MB)
  ebti/       from ebti-v1.zip (about 54 MB)
```

On macOS and Windows, double-clicking a zip creates the folder next to it: move `airbnb/` and `ebti/` into `case-study/data/`. In a terminal, from the repository folder:

```bash
unzip ~/Downloads/airbnb-v1.zip -d case-study/data/
unzip ~/Downloads/ebti-v1.zip -d case-study/data/
```

**Provenance and versions.** Each folder contains a `PROVENANCE.md`: where every file comes from, when it was downloaded, what was done to it, its licence, and the data version, which is also in the zip name (`airbnb-v1`, `ebti-v1`). If the lecturer corrects a dataset, a zip with a higher version appears on the course drive: delete the old folder, unzip the new one, and check the version at the top of `PROVENANCE.md`.

Everything in `case-study/data/` except its README is ignored by Git, so the data stay on your computer. Check the result with `ls case-study/data/airbnb case-study/data/ebti`: the file names must match the tables below. Session 3 also builds a SQLite database from the Airbnb files: `uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py` writes `case-study/data/airbnb/airbnb.sqlite` (about 100 MB).

**Why a download, and why not all of the data?** Both datasets are public, but a model is only judged fairly on data it has never seen. The lecturer therefore holds back part of each dataset before handing out the rest. How you split what you receive into training and validation data is up to you (Sessions 7 and 9); the held-back part measures how well your final design works on new cases, as it would after deployment.

## Inside Airbnb, Berlin (Sessions 1–12)

Sessions 1–12 (Python, classes, SQL, data quality, statistics, regression, validation, features, tree-based models, clustering, demand over time) use the Berlin listings of [Inside Airbnb](https://insideairbnb.com/get-the-data/), snapshot of 26 June 2026: listings with price, district, room type, size, ratings, availability and registration number, plus the number of reviews per listing and month since 2009 as a proxy for demand.

| File in `airbnb/` | Rows | Content |
|---|---:|---|
| `listings.parquet` | 10,221 | one row per listing: location (district, neighbourhood, coordinates), property, price per night, minimum nights, availability, reviews and ratings, host (no names), registration number |
| `calendar.parquet` | 3,759,500 | availability of each listing for the next 365 days |
| `reviews_monthly.parquet` | 180,656 | reviews per listing and month since 2009 (no review texts) |
| `weather_daily.parquet` | 3,830 | daily Berlin weather since 2016 from the Open-Meteo archive (temperature, rain, sunshine); Session 16 shows how to fetch such data from a web API yourself |
| `neighbourhoods.geojson` | | district boundaries |
| `PROVENANCE.md` | | where the files come from, the snapshot date, the data version |

**The held-back listings.** The snapshot has 12,776 listings; 2,555 of them (20 %) are held back as a test set, together with their calendar and review rows. The split follows the course's prediction task, the price of a listing:

- **By host.** All listings of a host are on the same side. Hosts with many listings post near-identical flats. If such near-copies were on both sides of the split, a model could find the price of the copy instead of predicting it.
- **Stratified by district, room type and whether a price is shown.** These have the same mix in your data and in the test set (for example 69 % entire homes, 22 % in Mitte, 66 % with a price on both sides), so a score on the test set measures the same task you trained for.

Personal data are removed before the split: host names, profile texts, photos, review texts, reviewer names, and names that hosts entered in the registration field (`license` keeps only registration numbers; `license_status` gives the type of entry).

## EBTI: classifying goods in the EU customs tariff (Sessions 13–16)

Every product that crosses the EU border is classified in the customs nomenclature, and the classification decides the duty. Traders who are unsure can ask a customs authority for a **Binding Tariff Information (BTI)** decision: they describe the product, and customs states its code. The European Commission publishes all decisions in the [EBTI database](https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en).

**Task of Sessions 13–16:** predict the four-digit **HS heading** of a decision (for example `6404`, footwear with textile uppers) from its description of goods.

The descriptions are written in the language of the issuing country: about 61 % German, 16 % French, and the rest in more than 20 other EU languages. Each training decision also has English keywords, which help you read a decision you cannot read in its original language.

**Which decisions.** A BTI decision is binding for three years. The course uses the decisions that were **valid on 15 October 2026**, the day of the export: 122,003 decisions that started between late 2023 and 2026. Decisions repeating another one word for word (renewals) are kept once, leaving 119,514.

**What was held back.** One fold of seven (about 14 %, 17,074 decisions) forms an independent evaluation set, drawn at random and **stratified by heading and language together**, so that every combination of heading and language has about the same share as in your data. Combinations too small to divide are placed on one side only. It is never released: after the code freeze the lecturer sends it to your team's API ([final project](../sessions/17-18-final-presentations/final-project.md)).

| File in `ebti/` | Rows | Content |
|---|---:|---|
| `decisions.parquet` | 102,440 | bti_reference, issuing_country, language, start_date, end_date, date_of_issue, description, keywords, classification_justification, cn_code, **heading** (the label), chapter |
| `nomenclature/Nomenclature EN.xlsx` (and `DE`, `FR`) | 25,866 each | the EU goods nomenclature (TARIC) of 1 October 2026, raw as the European Commission publishes it: every line of the customs tariff with its validity dates |

There is no ready-made training, validation or test split: designing it is part of the final project.

### Evaluation

- **Task:** a REST API that takes a description (and optionally its language) and returns the heading. The interface is specified in the [final project](../sessions/17-18-final-presentations/final-project.md#1--a-rest-api).
- **Metric:** accuracy (share of correct headings), with macro-F1 alongside. There are about 1,000 headings with a long tail: a few headings are very frequent and hundreds are rare. Accuracy and macro-F1 therefore measure different things; Session 13 explains why.
- **Your reported score:** each team reports the accuracy and macro-F1 of its final design, measured with its own validation design.
- **Independent score:** the lecturer runs `evaluate_api.py` on the final evaluation set; the scores are revealed in Session 18 and compared with the reported ones. Check your own service the same way on your validation decisions:
  ```bash
  uv run python case-study/evaluate_api.py --url http://localhost:8000 --data my_validation.parquet
  ```

**Reference values** for orientation, from one possible validation design: one fold of a stratified 7-fold split of `decisions.parquet` by heading and language (`random_state=0`), training on 87,805 decisions and validating on the other 14,635. The linear model is scikit-learn's `SGDClassifier(loss="hinge")`.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Always the most frequent heading (3926, other articles of plastics) | 0.042 | 0.000 |
| Word TF-IDF + linear model, 50,000 training decisions | 0.856 | 0.667 |
| Word TF-IDF + linear model, all 87,805 training decisions | 0.877 | 0.706 |
| Character TF-IDF (3–5) + linear model, all 87,805 training decisions | 0.885 | 0.718 |

**Checkpoints** (not graded, voluntary but recommended): in Session 13 (TF-IDF models) and Session 14 (embeddings or language models) teams may post their validation design and score on the class board. Decisions can be looked up in the public EBTI database, so the score is not converted into marks: the project is graded on how a team frames, handles, solves and explains the problem.

### Things to watch out for

- **Leakage through the justification.** Data leakage means that a model uses information it would not have for a real request, so its score looks better than it is. `classification_justification` explains the decision and names the heading in most cases. It is written after classification and does not exist for a new request. Never use it as a model input; Session 13 measures how much it raises a score. The same holds for `keywords`, `cn_code`, `chapter` and the dates.
- **Quoted codes.** Some descriptions quote the tariff text of their own code. Numbers that repeat the decision's own full code are replaced by `<CODE>`.
- **Unseen headings.** 16 decisions of the evaluation set (15 headings) have a heading that never occurs in `decisions.parquet`. No model trained on the course data gets them right.
- **Changing nomenclature.** The HS nomenclature is revised every five to six years (HS 2022, next HS 2028). Some headings are created, split or deleted, which changes the labels over time (Session 16).
- **Data quality.** Some descriptions are database templates or test entries.

## Sources and terms of use

- European Commission, *European Binding Tariff Information (EBTI)* database, full export of 15 October 2026. Reuse is permitted with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU). The holders of the decisions are not published.
- Open-Meteo, historical weather API, [open-meteo.com](https://open-meteo.com), Creative Commons Attribution 4.0 (CC BY 4.0); free for non-commercial use.
- Inside Airbnb, Berlin snapshot of 26 June 2026 (listings, calendar, reviews), [insideairbnb.com](https://insideairbnb.com/get-the-data/), Creative Commons Attribution 4.0 (CC BY 4.0). The data are scraped from public Airbnb pages; report results in aggregate and do not name hosts.
- European Commission, TARIC monthly extraction of 1 October 2026 (`Nomenclature EN/DE/FR.xlsx`), CIRCABC group *TARIC & Quota Data and Information*; same reuse conditions as the EBTI database.
