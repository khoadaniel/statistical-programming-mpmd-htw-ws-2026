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

**Read `PROVENANCE.md` first.** Each folder contains one: it explains what each file contains (what one row is, the columns, how the files link), where the files come from, what was changed, the licence and the data version, which is also in the zip name (`airbnb-v1`, `ebti-v1`). If the lecturer corrects a dataset, a zip with a higher version appears on the course drive: delete the old folder, unzip the new one, and check the version at the top of `PROVENANCE.md`.

**Check the result** from the repository folder (works on macOS, Linux and Windows):

```bash
uv run python -c "from pathlib import Path; [print(p) for p in sorted(Path('case-study/data').rglob('*'))]"
```

The file names must match the tables below. A common mistake is a doubled folder such as `case-study/data/ebti/ebti/` (the zip was unzipped into a folder of its own): move the inner folder up. Session 3 also builds a SQLite database from the Airbnb files: `uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py` writes `case-study/data/airbnb/airbnb.sqlite` (about 100 MB). Everything in `case-study/data/` except its README is ignored by Git, so the data stay on your computer.

**Why Parquet and not CSV?** [Parquet](https://parquet.apache.org/) is an open file format for tables that stores each column with its type, compressed. Headings keep their leading zeros (`0901` is coffee; a CSV read back with `pd.read_csv` turns it into the number `901`), dates stay dates, and `decisions.parquet` is about half the size of the same table as CSV. Read a file with `pd.read_parquet(path)`. A Parquet file is binary, so it does not open in a text editor or Excel: use `df.head()`, or write a sample with `df.head(100).to_csv(...)`.

**Why a download, and why not all of the data?** Both datasets are public, but a model is only judged fairly on data it has never seen. The lecturer therefore holds back part of each dataset before handing out the rest. How you split what you receive into training and validation data is up to you (Sessions 7 and 9); the held-back part measures how well your final design works on new cases, as it would after deployment.

## Inside Airbnb, Berlin (Sessions 1–12)

The Berlin listings of Inside Airbnb, snapshot of 26 June 2026, with the daily Berlin weather. Where the data come from and a first look: [airbnb/README.md](airbnb/README.md).

| File in `airbnb/` | Rows | One row is |
|---|---:|---|
| `listings.parquet` | 10,221 | a listing |
| `calendar.parquet` | 3,759,500 | a listing on one day of the following year |
| `reviews_monthly.parquet` | 180,656 | a listing in a month with reviews |
| `weather_daily.parquet` | 3,830 | a day of Berlin weather |
| `neighbourhoods.geojson` | | a neighbourhood with its border, for maps |
| `PROVENANCE.md` | | what each file contains, its columns, how the files link, where they come from |

**The held-back listings.** Part of the listings is held back as a test set, together with their calendar and review rows. The split follows the course's prediction task, the price of a listing:

- **By host.** All listings of a host are on the same side. Hosts with many listings post near-identical flats. If such near-copies were on both sides of the split, a model could find the price of the copy instead of predicting it.
- **Stratified by district, room type and whether a price is shown.** These have the same mix in your data and in the test set, so a score on the test set measures the same task you trained for.

Personal data are removed before the split: host names, profile texts, photos, review texts, reviewer names, and names that hosts entered in the registration field (`license` keeps only registration numbers; `license_status` gives the type of entry).

## EBTI: classifying goods in the EU customs tariff (Sessions 13–16)

Binding Tariff Information decisions of the EU customs, published by the European Commission. The task is to predict the four-digit **HS heading** of a decision from its description of goods. Where the data come from, what a decision looks like and its pitfalls: [ebti/README.md](ebti/README.md).

**Which decisions.** The decisions that are **binding on 1 October 2026**, the date of the EBTI export and of the nomenclature file (`PROVENANCE.md` explains why one date matters).

**What was held back.** Part of the decisions is held back as the independent evaluation set of the final project: a random sample with the same mix of headings and languages as your data. It is never released: after the code freeze the lecturer sends it to your team's API ([how the evaluation works](../sessions/17-18-final-presentations/final-project.md#how-the-evaluation-works)).

| File in `ebti/` | Rows | One row is |
|---|---:|---|
| `decisions.parquet` | 103,656 | a decision: a description of goods and its code; **heading** is the label |
| `nomenclature/Nomenclature EN.xlsx` (and `DE`, `FR`) | 25,866 each | a line of the EU customs tariff, raw as published |
| `PROVENANCE.md` | | what each file contains, its columns, how the files link, where they come from |

There is no ready-made training, validation or test split: designing it is part of the [final project](../sessions/17-18-final-presentations/final-project.md), which also describes the API, the evaluation and the grading.

### Reference values

For orientation, results from one possible validation design, the split used in Sessions 13–16: `decisions.parquet` divided by heading and language, 88,848 decisions for training and 14,808 for validation. The linear model is scikit-learn's `SGDClassifier(loss="hinge")`. The second row trains the same model on a random sample of the training decisions, to show what more data adds.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Always the most frequent heading (3926, other articles of plastics) | 0.042 | 0.000 |
| Word TF-IDF + linear model, trained on a random sample of 50,000 training decisions | 0.859 | 0.668 |
| Word TF-IDF + linear model, all 88,848 training decisions | 0.883 | 0.736 |
| Character TF-IDF (3–5) + linear model, all 88,848 training decisions | 0.890 | 0.746 |

## Sources and terms of use

- European Commission, *European Binding Tariff Information (EBTI)* database, full export downloaded on 2 October 2026. Reuse is permitted with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU). The holders of the decisions are not published.
- Open-Meteo, historical weather API, [open-meteo.com](https://open-meteo.com), Creative Commons Attribution 4.0 (CC BY 4.0); free for non-commercial use.
- Inside Airbnb, Berlin snapshot of 26 June 2026 (listings, calendar, reviews), [insideairbnb.com](https://insideairbnb.com/get-the-data/), Creative Commons Attribution 4.0 (CC BY 4.0). The data are scraped from public Airbnb pages; report results in aggregate and do not name hosts.
- European Commission, TARIC monthly extraction of 1 October 2026 (`Nomenclature EN/DE/FR.xlsx`), CIRCABC group *TARIC & Quota Data and Information*; same reuse conditions as the EBTI database.
