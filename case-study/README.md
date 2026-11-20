# Course datasets

| Dataset | Sessions | Data folder | Introduction |
|---|---|---|---|
| Inside Airbnb, Berlin, with Berlin weather from Open-Meteo | 1–12 | `case-study/airbnb/data/` | [airbnb/README.md](airbnb/README.md) |
| IBM Telco customer churn | 6, 8–11 | none: the notebooks read it from IBM's GitHub repository (Apache 2.0) | |
| EBTI, EU customs decisions: the final challenge | 13–16 | `case-study/final-challenge/data/` | [final-challenge/README.md](final-challenge/README.md) |

Start with the two introductions: where the data come from, why the course uses them, and a first look with charts. This page covers the folders, how to get the data and what was held back.

## The folder at a glance

```
case-study/
  README.md                 this page
  airbnb/
    README.md               introduction to the Airbnb data, with charts
    figures/
    data/                   the Airbnb data (Git LFS)
  final-challenge/
    README.md               introduction to the EBTI data, with charts
    figures/
    data/                   decisions.parquet and the nomenclature (Git LFS)
  make_dataset_figures.py   draws the charts of the two introductions
```

The data folders are part of the repository. You do not change them.

## Getting the data

The data are stored with [Git LFS](https://git-lfs.com), which keeps large files out of the normal Git history. Git LFS is a small add-on to Git: macOS (`brew install git-lfs`) and Linux (`sudo apt install git-lfs`) need it installed, the Windows installer of Git includes it ([Getting started, step 2](../GETTING_STARTED.md#step-2--install-git)). Switch it on **once per computer**, before you clone or pull:

```bash
git lfs install      # once per computer
git pull             # with Git LFS switched on, this downloads the data too (about 60 MB)
```

If a data file contains only three lines of text, Git LFS was not switched on when you pulled. Run `git lfs pull`: it replaces each placeholder with the real file.

**Already have the data from another source?** If `git pull` stops with `untracked working tree files would be overwritten`, your copies sit at the same paths as the files in the repository. Delete the files that the message lists, or move them out of the folder, and pull again: the repository brings them back.

If you cloned the repository with Git LFS switched on, the data are already there. **Read `PROVENANCE.md` first.** Each data folder contains one: it explains what each file contains (what one row is, the columns, how the files link), where the files come from, what was changed, the licence and the data version. If the lecturer corrects a dataset, a higher version is pushed to the repository: run `git pull` and check the version at the top of `PROVENANCE.md`.

**Check the result** from the repository folder (works on macOS, Linux and Windows):

```bash
uv run python -c "from pathlib import Path; [print(p) for p in sorted(Path('case-study').glob('*/data/*'))]"
```

The file names must match the tables below. A file of a few lines of text instead of data means that Git LFS has not fetched it yet: run `git lfs pull`. Session 3 also loads the Airbnb files into a local PostgreSQL database: `uv run python sessions/03-sql/workbooks/make_airbnb_postgres.py` starts the server and stores its data in `case-study/airbnb/data/pgdata/` (about 700 MB). That folder is ignored by Git.

**Why Parquet and not CSV?** [Parquet](https://parquet.apache.org/) is an open file format for tables that stores each column with its type, compressed. Headings keep their leading zeros (`0901` is coffee; a CSV read back with `pd.read_csv` turns it into the number `901`), dates stay dates, and `decisions.parquet` is about half the size of the same table as CSV. Read a file with `pd.read_parquet(path)`. A Parquet file is binary, so it does not open in a text editor or Excel: use `df.head()`, or write a sample with `df.head(100).to_csv(...)`.

**Why not all of the data?** Both datasets are public, but a model is only judged fairly on data it has never seen. The lecturer therefore holds back part of each dataset before handing out the rest. How you split what you receive into training and validation data is up to you (Sessions 7 and 9); the held-back part measures how well your final design works on new cases, as it would after deployment.

## Inside Airbnb, Berlin (Sessions 1–12)

The Berlin listings of Inside Airbnb, snapshot of 26 June 2026, with the daily Berlin weather. Where the data come from and a first look: [airbnb/README.md](airbnb/README.md).

| File in `airbnb/data/` | Rows | One row is |
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

Binding Tariff Information decisions of the EU customs, published by the European Commission. The final project is to predict the six-digit **HS subheading** (HS6) of a decision from its description of goods. Where the data come from, what a decision looks like and its pitfalls: [final-challenge/README.md](final-challenge/README.md).

**Which decisions.** The decisions that are **binding on 1 October 2026**, the date of the EBTI export and of the nomenclature file (`PROVENANCE.md` explains why one date matters).

**What was held back.** Part of the decisions is held back as the independent evaluation set of the final project: a random sample with the same mix of subheadings and languages as your data. It is never released: after the code freeze the lecturer sends it to your team's API ([how the evaluation works](../sessions/17-18-final-presentations/final-project.md#how-the-evaluation-works)).

| File in `final-challenge/data/` | Rows | One row is |
|---|---:|---|
| `decisions.parquet` | 103,656 | a decision: a description of goods and its code; **subheading** is the label to predict |
| `nomenclature/Nomenclature EN.xlsx` (and `DE`, `FR`) | 25,866 each | a line of the EU customs tariff, raw as published |
| `PROVENANCE.md` | | what each file contains, its columns, how the files link, where they come from |

There is no ready-made training, validation or test split: designing it is part of the [final project](../sessions/17-18-final-presentations/final-project.md), which also describes the API, the evaluation and the grading.

### Reference values

For orientation, results from one possible validation design: `decisions.parquet` divided by subheading and language, 6 of 7 folds for training and the 7th for validation. The linear model is scikit-learn's `SGDClassifier(loss="hinge")` on word TF-IDF. The flat model keeps one weight per class and word, so its memory grows with the number of classes: working within your hardware is part of the task ([the question in the brief](../sessions/17-18-final-presentations/final-project.md#questions-and-answers)).

| Model | Subheading accuracy | Macro-F1 | Heading accuracy | Peak memory while training |
|---|---:|---:|---:|---:|
| Word TF-IDF + linear model, course settings (`alpha=1e-5`) | 0.831 | 0.597 | 0.871 | 9.0 GB |
| The same, tuned (`alpha=2e-6`) | 0.847 | 0.633 | 0.882 | 9.0 GB |
| Heading first, then subheading within it (hierarchical), word TF-IDF + linear models | 0.838 | 0.614 | 0.888 | 3.5 GB |

The practice sessions work on the four-digit heading. Reference values for it, with `decisions.parquet` divided by heading and language (88,848 decisions for training, 14,808 for validation); the second row trains on a random sample of the training decisions, to show what more data adds:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Word TF-IDF + linear model, trained on a random sample of 50,000 training decisions | 0.859 | 0.668 |
| Word TF-IDF + linear model, all 88,848 training decisions | 0.883 | 0.736 |
| Character TF-IDF (3–5) + linear model, all 88,848 training decisions | 0.890 | 0.746 |

## Sources and terms of use

- European Commission, *European Binding Tariff Information (EBTI)* database, full export downloaded on 2 October 2026. Reuse is permitted with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU). The holders of the decisions are not published.
- Open-Meteo, historical weather API, [open-meteo.com](https://open-meteo.com), Creative Commons Attribution 4.0 (CC BY 4.0); free for non-commercial use.
- Inside Airbnb, Berlin snapshot of 26 June 2026 (listings, calendar, reviews), [insideairbnb.com](https://insideairbnb.com/get-the-data/), Creative Commons Attribution 4.0 (CC BY 4.0). The data are scraped from public Airbnb pages; report results in aggregate and do not name hosts.
- European Commission, TARIC monthly extraction of 1 October 2026 (`Nomenclature EN/DE/FR.xlsx`), CIRCABC group *TARIC & Quota Data and Information*; same reuse conditions as the EBTI database.
