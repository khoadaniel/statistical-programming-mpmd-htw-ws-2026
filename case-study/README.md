# Course datasets

| Dataset | Sessions | Folder after download | Introduction |
|---|---|---|---|
| Inside Airbnb, Berlin, with Berlin weather from Open-Meteo | 1–12 | `case-study/data/airbnb/` | [airbnb/README.md](airbnb/README.md) |
| IBM Telco customer churn | 6, 8–11 | none: the notebooks read it from IBM's GitHub repository (Apache 2.0) | |
| EBTI, EU customs decisions, with the class leaderboard | 13–16 | `case-study/data/ebti/` | [ebti/README.md](ebti/README.md) |

Start with the two introductions: where the data come from, why the course uses them, and a first look with charts. This page covers the download, the split and the leaderboard.

## Getting the data

The lecturer prepares both datasets and puts two zip files on the **course drive**. Download them and unzip them **inside `case-study/data/`**, so that you get:

```
case-study/data/
  airbnb/     from airbnb.zip (about 3 MB)
  ebti/       from ebti.zip (about 80 MB)
```

On macOS and Windows, double-clicking a zip creates the folder next to it: move `airbnb/` and `ebti/` into `case-study/data/`. In a terminal, from the repository folder:

```bash
unzip ~/Downloads/airbnb.zip -d case-study/data/
unzip ~/Downloads/ebti.zip -d case-study/data/
```

Everything in `case-study/data/` except its README is ignored by Git, so the data stay on your computer. Check the result with `ls case-study/data/airbnb case-study/data/ebti`: the file names must match the tables below. Session 3 also builds a SQLite database from the Airbnb files: `uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py` writes `case-study/data/airbnb/airbnb.sqlite` (about 100 MB).

**Why a download, and why not all of the data?** Both datasets are public, but a model is only judged fairly on data it has never seen. The lecturer therefore holds back a **test set** of each dataset before handing out the rest. You train and validate on what you receive; the held-back part measures how well your model works on new cases, as it would after deployment.

## Inside Airbnb, Berlin (Sessions 1–12)

Sessions 1–12 (Python, classes, SQL, data quality, statistics, regression, validation, features, tree-based models, clustering, demand over time) use the Berlin listings of [Inside Airbnb](https://insideairbnb.com/get-the-data/), snapshot of 26 June 2026: listings with price, district, room type, size, ratings, availability and registration number, plus the number of reviews per listing and month since 2009 as a proxy for demand.

| File in `airbnb/` | Rows | Content |
|---|---:|---|
| `listings.parquet` | 10,221 | one row per listing: location (district, neighbourhood, coordinates), property, price per night, minimum nights, availability, reviews and ratings, host (no names), registration number |
| `calendar.parquet` | 3,759,500 | availability of each listing for the next 365 days |
| `reviews_monthly.parquet` | 180,656 | reviews per listing and month since 2009 (no review texts) |
| `weather_daily.parquet` | 3,830 | daily Berlin weather since 2016 from the Open-Meteo archive (temperature, rain, sunshine); Session 16 shows how to fetch such data from a web API yourself |
| `neighbourhoods.geojson` | | district boundaries |
| `snapshot.txt` | | date of the Inside Airbnb snapshot |

**The held-back listings.** The snapshot has 12,776 listings; 2,555 of them (20 %) are held back as a test set, together with their calendar and review rows. The split follows the course's prediction task, the price of a listing:

- **By host.** All listings of a host are on the same side. Hosts with many listings post near-identical flats; if such twins were split, a model could look one up instead of predicting it.
- **Stratified by district, room type and whether a price is shown.** These have the same mix in your data and in the test set (for example 69 % entire homes, 22 % in Mitte, 66 % with a price on both sides), so a score on the test set measures the same task you trained for.

Personal data are removed before the split: host names, profile texts, photos, review texts, reviewer names, and names that hosts entered in the registration field (`license` keeps only registration numbers; `license_status` gives the type of entry).

## EBTI: classifying goods in the EU customs tariff (Sessions 13–16)

Every product that crosses the EU border is classified in the customs nomenclature, and the classification decides the duty. Traders who are unsure can ask a customs authority for a **Binding Tariff Information (BTI)** decision: they describe the product, and customs states its code. The European Commission publishes all decisions in the [EBTI database](https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en).

**Task of Sessions 13–16:** predict the four-digit **HS heading** of a decision (for example `6404`, footwear with textile uppers) from its description of goods.

The descriptions are written in the language of the issuing country: about 61 % German, 16 % French, and the rest in more than 20 other EU languages. Each training decision also has English keywords, which help you read a decision you cannot read in its original language.

**Which decisions.** A BTI decision is binding for three years. The course uses the decisions that were **valid on 15 October 2026**, the day of the export: 122,003 decisions that started between late 2023 and 2026. Decisions repeating another one word for word (renewals) are kept once, leaving 119,514.

**The split.** All years are mixed and the decisions are split 70 / 15 / 15, **stratified by heading**: every heading has the same share in the three parts (3926, other articles of plastics, is 4.2 % of each). Headings with fewer than ten decisions are stratified together.

| File in `ebti/` | Rows | Content |
|---|---:|---|
| `train.parquet` | 83,659 | bti_reference, issuing_country, language, start_date, end_date, date_of_issue, description, keywords, classification_justification, cn_code, **heading** (the label), chapter |
| `train_sample.parquet` | 50,000 | random sample of `train`, for quick experiments |
| `val.parquet` | 17,927 | same columns as `train`: the validation set to compare and tune your models |
| `test.parquet` | 17,928 | id, issuing_country, language, start_date, description: what a trader's request contains, plus where and when it was decided. **No heading**: this is the leaderboard set |
| `nomenclature.parquet` | 1,229 | heading, heading_description, chapter, chapter_description, section, section_name (English, HS 2022) |

### Leaderboard

- **Task:** predict the heading of each decision in `test.parquet`.
- **Public and Private.** The test set is split in two halves, again stratified by heading. During the course the leaderboard shows the score on the Public half; the Private half decides the final ranking and stays hidden until the end.
- **Metric:** accuracy (share of correct headings), with macro-F1 reported alongside. With almost 1,000 headings (988 in the training set) and a long tail of rare ones, the two tell different stories; Session 13 explains why.
- **Submission:** a CSV file with the columns `id,heading` and one row per test decision, uploaded to the leaderboard. The test labels are only on the lecturer's computer; to check a model yourself, score it on `val.parquet`.

**Reference values** (validation set; the Public half gives the same numbers within half a point). The linear model is scikit-learn's `SGDClassifier(loss="hinge")`.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Always the most frequent heading (3926, other articles of plastics) | 0.042 | 0.000 |
| Word TF-IDF + linear model, 50,000-decision sample | 0.851 | 0.622 |
| Word TF-IDF + linear model, full training set | 0.872 | 0.675 |
| Character TF-IDF (3–5) + linear model, full training set | 0.880 | 0.682 |

**Rounds:**

| Round | Session | Model |
|---|---|---|
| L1 | 13 | TF-IDF text classifier |
| L2 | 14 | embeddings or a language model, any method of the session |
| L3 | 16 | final submission after retraining with the released labels of the Public half, any method, with a one-page description |

The leaderboard task is also the final project of all teams. Decisions can be looked up in the public EBTI database, so the score is not converted into marks: the project is graded on how a team frames, handles, solves and explains the problem, and its final submission must be reproducible from the team's repository. In Session 16 the lecturer releases the labels of the Public half (`public_labels.csv` on the course drive) as feedback data; the Private half stays hidden.

### Things to watch out for

- **Leakage through the justification.** `classification_justification` explains the decision and names the heading in most cases. It is written after classification, so it is not in the test set. Never use it as a model input; Session 13 uses it as an example of a feature that is not available at prediction time. The same holds for `keywords`, `cn_code` and the end dates.
- **Quoted codes.** Some descriptions quote the tariff text of their own code. Numbers that repeat the decision's own code are replaced by `<CODE>` in all tables.
- **Unseen headings.** A few test decisions (about 20) have a heading that never occurs in `train`. No classifier trained on `train` alone can get them right.
- **Changing nomenclature.** The HS nomenclature is revised every five years (latest: 2022). Some headings were created, split or deleted, which changes the labels over time (Session 16).
- **Data quality.** Some descriptions are database templates or test entries, and a few headings no longer exist in HS 2022.

## Sources and terms of use

- European Commission, *European Binding Tariff Information (EBTI)* database, full export of 15 October 2026. Reuse is permitted with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU). The holders of the decisions are not published.
- Open-Meteo, historical weather API, [open-meteo.com](https://open-meteo.com), Creative Commons Attribution 4.0 (CC BY 4.0); free for non-commercial use.
- Inside Airbnb, Berlin snapshot of 26 June 2026 (listings, calendar, reviews), [insideairbnb.com](https://insideairbnb.com/get-the-data/), Creative Commons Attribution 4.0 (CC BY 4.0). The data are scraped from public Airbnb pages; report results in aggregate and do not name hosts.
- Harmonized System nomenclature (HS 2022, English), [datasets/harmonized-system](https://github.com/datasets/harmonized-system), ODC Public Domain Dedication and Licence (PDDL).
