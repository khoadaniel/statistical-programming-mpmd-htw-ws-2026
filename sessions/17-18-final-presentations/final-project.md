# Final project · Suggest the customs heading of a product

Every product that enters the European Union needs a customs code, and the code decides the duty. When traders are unsure, customs experts classify the product for them in a **Binding Tariff Information (BTI) decision**. Your team builds a service that reads such a product description, in any of 23 languages, and suggests the four-digit **heading**, so that a customs officer can check it faster.

All teams work on the same task, in teams of three (one team of four), from Session 1 to Session 16, and present their solution in Session 17 or 18. The project is the only graded part of the module (100 %).

```mermaid
flowchart LR
  A["📦 ebti-v1.zip<br/>102,440 labelled decisions"] --> B["🧪 Your validation design<br/>how you split, what you measure"]
  B --> C["🧠 Your model<br/>from a baseline to the best<br/>approach you can justify"]
  C --> D["🔌 Your REST API<br/>description in,<br/>heading out"]
  D --> E["🔒 Independent evaluation<br/>decisions you never see,<br/>run by the lecturer"]
  B -. "your reported score" .-> F["📊 Session 18<br/>reported vs independent score"]
  E --> F
```

## The goal in one sentence

> Build a **REST API** that takes the description of a product and returns its four-digit HS heading, **report how well your final design works**, and show that your report can be trusted.

The last part matters most. The lecturer keeps an **independent evaluation set** of decisions that no team ever sees, and runs every team's API on it after the code freeze. If your reported score is close to your independent score, your validation design worked. If there is a large gap, for example a reported accuracy of 0.95 and an independent accuracy of 0.80, the validation design overestimated the model, and your presentation should explain why.

## Rules

Read these first: they apply from Session 1. Breaking a rule lowers the grade more than a better model can raise it.

- **Reproducible on any computer.** Your work must run on another machine, not only on yours. The tagged repository rebuilds the model from `ebti-v1.zip` and starts the service with `docker compose up`, and its README says how. The lecturer may follow your README to rerun the training and deployment scripts. If they fail, the grade drops considerably. A service that does not start from the tag is not evaluated. Reproducibility is one of the most important habits in machine learning, and the first thing a colleague will test in your job.
- **Collaborate through GitHub.** Work on branches, open pull requests and review each other's code before it is merged. The lecturer reads the Git history and the pull requests to see how the team worked together and who did what.
- **Train only on the data you are given.** Do not add decisions downloaded from the public EBTI database: the held-back evaluation decisions are in it too, so your model would have seen the evaluation data during training. This is called **data leakage**: information that the model should not have reaches it during training, and the score looks better than it is.
- **Predict, never look up.** Your service must predict the heading from the description, not search for the decision online or in a copy of the database.
- **Only inputs that exist when a request arrives:** the description, and optionally the language. The justification, keywords, CN code, chapter and dates are written with or after the decision; a model that uses them looks excellent in validation and fails in use (Session 13).
- **No data in Git.** Never commit data, models larger than a few megabytes, or credentials.
- **AI tools** are allowed and must be documented with the HTW declaration. You must be able to explain and test all code you submit.

## The data

Download `airbnb-v1.zip` (Sessions 1–12) and `ebti-v1.zip` (the project, needed from Session 3) from the course drive and unzip both **inside `case-study/data/`** ([how](../../case-study/README.md#getting-the-data)). Then check the result from the repository folder; this command works on macOS, Linux and Windows:

```bash
uv run python -c "from pathlib import Path; [print(p) for p in sorted(Path('case-study/data').rglob('*'))]"
```

It prints one path per line, such as `case-study/data/ebti/decisions.parquet`. If the data are in the right place, the paths match this tree:

```text
case-study/data/
├── .gitkeep
├── README.md
├── airbnb/
│   ├── PROVENANCE.md
│   ├── calendar.parquet
│   ├── listings.parquet
│   ├── neighbourhoods.geojson
│   ├── reviews_monthly.parquet
│   └── weather_daily.parquet
└── ebti/
    ├── decisions.parquet      about 57 MB
    ├── nomenclature/          Nomenclature EN.xlsx, DE, FR and the Commission's guide to them
    └── PROVENANCE.md          where every file comes from, and the data version
```

Typical mistakes: a doubled folder such as `case-study/data/ebti/ebti/` (the zip was unzipped into a folder of its own; move the inner folder up), or `airbnb/` and `ebti/` next to `case-study/` instead of inside `case-study/data/`. From Session 3, `airbnb/` also holds `airbnb.sqlite`, which you build yourself.

**Why Parquet and not CSV?** [Parquet](https://parquet.apache.org/) is an open file format for tables, common in data platforms. Unlike a CSV text file it stores each column together with its type, compressed. For our data this matters:

- **Types survive.** Headings are text with leading zeros: `0901` is coffee. Saved as CSV and read back with `pd.read_csv`, `0901` becomes the number `901`, and dates come back as plain text. In Parquet, headings stay text and dates stay dates.
- **Smaller and faster.** `decisions.parquet` is about 57 MB; the same table as CSV is about 120 MB, and reading it takes longer.
- **How to read it:** `pd.read_parquet("case-study/data/ebti/decisions.parquet")` in pandas, which uses the [pyarrow](https://arrow.apache.org/docs/python/) library in the background (both come with `uv sync`). Other tools read it too (Polars, DuckDB, Spark), but the course uses pandas. A Parquet file is binary, so it does not open in a text editor or Excel: look at it with `df.head()`, or write a small sample with `df.head(100).to_csv(...)` if you need a spreadsheet.

Read [the introduction to the data](../../case-study/ebti/README.md) first: where it comes from, what a decision looks like, and its pitfalls.

| File | Rows | Content |
|---|---:|---|
| `ebti/decisions.parquet` | 102,440 | decisions valid on 15 October 2026: issuing country, language, dates, description, English keywords, justification, CN code and the **heading** (your label) |
| `ebti/nomenclature/Nomenclature EN.xlsx` (and `DE`, `FR`) | 25,866 each | the EU goods nomenclature (TARIC) of 1 October 2026, raw: every code of the customs tariff with its text and validity dates |

There is **no ready-made split**. How you divide the decisions to develop, tune and judge your model is your first design decision, and you justify it (Sessions 6–9). Things to weigh: about 1,000 headings, with a **long tail** (a few headings have thousands of decisions, hundreds have fewer than ten), 23 languages, decisions from late 2023 to 2026, and near-identical texts across decisions (exact repeats were removed).

## What you deliver

### 1 · A REST API

Your service must answer this interface. The Session 16 workspace already implements it; you may extend it, but not change it.

```http
POST /predict
Content-Type: application/json

{"description": "Damenstiefel mit Oberteil aus Rindleder und Laufsohle aus Gummi", "language": "de"}
```

```json
{
  "heading": "6403",
  "top": [
    {"heading": "6403", "score": 2.1},
    {"heading": "6405", "score": 0.4},
    {"heading": "6402", "score": 0.1}
  ]
}
```

- `description` (text, required) and `language` (two-letter code, optional) are the only inputs.
- `heading` (four digits) is required in the answer and is what gets scored; `top` (your three best headings) is shown in your demonstration. Other fields are allowed.
- `GET /health` answers `{"status": "ok"}` when the model is loaded.
- The service starts with `docker compose up` from your tagged repository, listens on port 8000, needs no internet access while it answers, and answers a single request within two seconds.

#### Check your API before you submit

The lecturer evaluates every service with the script `case-study/evaluate_api.py` of the course repository. Run it yourself on your own validation decisions first.

**1. Save your validation decisions** as a Parquet file. `validation` is your validation table, rows of `decisions.parquet` that your model was not trained on:

```python
validation[["description", "language", "heading"]].to_parquet("my_validation.parquet")
```

The file then looks like this (one row per decision):

| description | language | heading |
|---|---|---|
| Polished granite slabs , not carved of weight of 10kg or more | en | 6802 |
| Peigne pour cheveux composé à 100% de bois. | fr | 9615 |
| Chaise d'extérieur en acier avec dossier et assise en matière textile non rembourrés. | fr | 9401 |

- `description` and `heading` are required; `heading` is text with four digits, such as `0901`.
- `language` is optional. When the column is there, its value is sent with each description.

**2. Start your service**, then run the script from the course repository folder (add `--limit 200` for a quick check):

```bash
uv run python case-study/evaluate_api.py --url http://localhost:8000 --data my_validation.parquet
```

The script first calls `GET /health`. It then sends each description to `POST /predict`, one at a time, and compares the answer with `heading`. At the end it prints five lines (the numbers here are an example):

```text
decisions          17,074
failed requests    0
accuracy           0.8810
macro-F1           0.7120
seconds per answer median 0.015, max 0.210
```

| Line | Meaning |
|---|---|
| `decisions` | how many decisions were sent |
| `failed requests` | answers that were an error, a timeout (10 seconds) or had no `heading`; each counts as wrong |
| `accuracy` | share of decisions whose `heading` was right: the main metric |
| `macro-F1` | F1 per heading, averaged so that every heading counts equally: shows how you do on rare headings |
| `seconds per answer` | median and slowest response time; both must stay under two seconds |

The lecturer runs the same script on the independent evaluation set, so these are exactly the numbers you will see in Session 18.

### 2 · A written report with your reported score

A short report in your repository, written with the **[report template](report-template.md)** (copy it to `docs/report.md`). Part A, about four pages, is the team's work: problem, data, validation design, the models you compared and the final one. It states the **accuracy and macro-F1 of your final design**, how you estimated them and how uncertain they are: your prediction of how the service will do on decisions it has never seen. Part B is one short entry per member on what they did, because each member is marked individually. Part C documents the use of AI tools.

### 3 · The repository and the presentation

- A **tagged repository** that rebuilds everything from `ebti-v1.zip` to the running service, with a README that tells a stranger how (see *Reproducible on any computer* under [Rules](#rules)).
- A **15-minute presentation** with a live demonstration (Sessions 17–18).

#### How to submit: the `submission` tag

You submit by setting **one Git tag named exactly `submission`** on your repository on GitHub. There is nothing to upload or send: the lecturer takes the commit this tag points to, and that commit is your submission, including the report in `docs/report.md`. A **tag** is a fixed name for one commit; unlike a branch it never moves ([Session 2](../02-software-engineering/workbooks/02-lecture-notebook-to-application-git-and-ci.ipynb) explains tags and GitHub releases).

**Deadline:** the **code freeze**, the day before Session 17. The lecturer announces the exact date and time.

**Steps** (one team member does this, after the team agrees that `main` is ready):

1. Merge every pull request that should be part of the submission into `main`, and check that CI is green on `main`.
2. Check the submission as a stranger would: clone the repository into a new folder, follow your README from `ebti-v1.zip` to the running service, and run `case-study/evaluate_api.py` against it (see [the API section](#1--a-rest-api)).
3. Set the tag on the latest commit of `main` and push it:

   ```bash
   git switch main
   git pull                                              # the latest merged state
   git tag -a submission -m "Final project submission"   # an annotated tag: records who and when
   git push origin submission                            # a plain git push does not send tags
   ```

4. Check on GitHub: under *Code → Tags*, `submission` is listed and points to the commit you meant.
5. Check that the tag works: `git clone --branch submission <your repository URL> check-submission`, then follow the README in that folder once more.

**Rules for the tag**

- **Exactly one tag named `submission`**, annotated (`-a`), on the `main` branch. Version tags from Session 16 such as `v2.0.0` may stay; the same commit may carry both. Only `submission` is evaluated.
- **Only the tagged commit counts.** At the code freeze the lecturer records the commit the tag points to. The evaluation run, the rerun from the README and the grading of code, report and Git history all use that commit.
- **After the freeze, do not move, delete or recreate the tag.** A tag changed after the freeze is not accepted; the commit recorded at the freeze is graded. If you find a bug afterwards, say so in the presentation, with the fix you would make: knowing the limits of your release counts in your favour.
- **You may keep pushing commits after the tag** (for example to tidy your slides), but they are not graded and do not reach the evaluation.
- **Wrong commit tagged before the freeze?** Delete the tag and set it again: `git tag -d submission`, `git push origin --delete submission`, then step 3.
- **No tag at the freeze means no submission:** there is nothing to evaluate.

## How the evaluation works

```mermaid
flowchart LR
  A["Your team<br/>sets the tag submission<br/>the day before Session 17"] --> B["Lecturer<br/>clones the tag, may rerun<br/>training from the README"]
  B --> C["docker compose up<br/>starts your service"]
  C --> D["evaluate_api.py<br/>sends each hidden decision<br/>to POST /predict"]
  D --> E["accuracy and macro-F1<br/>compared with your<br/>reported score"]
  E --> F["Results<br/>in Session 18"]
```

The evaluation set holds decisions of the same kind as yours (valid on 15 October 2026, all languages; one fold of seven, about 14 % or 17,074 decisions, drawn at random and stratified by heading and language), held back before `ebti-v1.zip` was made. They stay hidden until the end. To practise monitoring and retraining, Session 16 treats your own decisions of 2026 as the requests that arrive after a release.

## Suggested check-ins

Nothing is graded before the final presentation, and you do not have to hand in or post anything along the way. The table is a suggested pace, not a list of deadlines; the only fixed date is the code freeze. The lecturer sets aside some time at each check-in to look at what teams share, so sharing your intermediate results is recommended: it is how you get early feedback and can correct your direction before the end. Questions about the project or the team work are welcome at any time, in class or as an issue in your repository.

```mermaid
flowchart LR
  A["S1–S2<br/>teams, repository"] --> B["S3–S4<br/>data loaded,<br/>quality report, charter"]
  B --> C["S5–S9<br/>exploration, validation<br/>design, baselines"]
  C --> D["S10<br/>interim review"]
  D --> E["S13–S14<br/>text models,<br/>checkpoints 1 and 2"]
  E --> F["S16<br/>API release,<br/>monitoring"]
  F --> G["S17–S18<br/>presentation,<br/>independent scores"]
```

| Session | Suggested state of the work | What the lecturer offers |
|---|---|---|
| S1 | Challenge introduced; teams formed | Answers in class |
| S2 | Team repository from the template, with branch protection and CI | A look at the repository, if you ask |
| S3 | `decisions.parquet` loaded into the team's SQLite database; first counts per year, language and member state in SQL | None |
| S4 | Data quality report and a short **project charter** | Written comments, if you share them |
| S5 | Exploration: decisions per heading, language, member state and year; the long tail | In class |
| S6–S7 | Baselines (most frequent heading, a simple rule) and the **validation design**: how you split, which metrics, how you measure uncertainty, which inputs are forbidden | On request |
| S8–S9 | A first classifier on simple features in a pipeline, with accuracy and macro-F1; a plan for rare headings | On request |
| S10 | **Interim review** (10 minutes per team): data, validation design, baselines, plan for S13–S16 | Feedback in class |
| S11–S12 | Open points from the interim review | None |
| S13 | TF-IDF classifier and error analysis; **checkpoint 1**: your validation score and design on the class board | Comparison with other teams in class |
| S14 | Do embeddings or a language model help? **Checkpoint 2** on the class board | Comparison with other teams in class |
| S15 | Optional: a language-model component (for example a classification assistant with retrieval), with its evaluation | On request |
| S16 | **API release**: tested service with the top-3 headings, started with Docker Compose and released through CI/CD; monitoring plan and model card. You retrain on all your decisions, the newest included, and update your reported score | A check that your API answers the interface, if you ask |
| Day before S17 | **Code freeze**: the tag `submission` set on `main` (**required**, see [How to submit](#how-to-submit-the-submission-tag)) | None |
| S17–S18 | **Final presentation** (graded); independent scores revealed in Session 18 | Grade and written feedback |

Posting at the checkpoints is voluntary. Teams that do see how the others validate, and see the gap between a reported and an independent score early, while there is still time to act on it.

## Project charter (Session 4)

Half a page in your repository (`docs/charter.md`) with your **plan**, written before you build anything. It is not graded on its own; its points later become the first sections of your report.

1. **Use:** who would use the suggestions, and what a wrong suggestion costs.
2. **Metric and baseline:** what you measure, and the simplest result you must beat.
3. **Validation design:** how you split the decisions, and why.
4. **Roles:** who works on what until the interim review.

> **Note.** The charter is not the model card. The charter is the plan at the start, for your team; the [model card](../16-deployment-and-monitoring/workspace/MODEL_CARD.md) (Session 16) describes the finished model at its release, for the people who will use it.

## Presentation and assessment

Each team presents for 15 minutes, including a live demonstration of the API, followed by 10 minutes of questions with individual questions to each member. The project is graded once, after the final presentation, on three things: the **repository** at its `submission` tag (including a rerun from its README and its Git history), the **written report** and the **presentation**. The first five criteria (80 %) are assessed per team; *collaboration and presentation* (20 %) per member, from your entry in Part B of the report, your commits and pull requests, your presented part and your answers.

| Criterion | Weight | Evidence | Expected for a very good grade |
|---|---|---|---|
| Problem definition | 10 % | Report, presentation | The use of the suggestions is clear; metric, baseline and, if used, abstention follow from that use |
| Data | 20 % | Repository, rerun from the README | A pipeline that reruns from `ebti-v1.zip` to the model on another computer; quality checked, every cleaning decision recorded; inputs limited to what exists when a request arrives |
| Model | 25 % | Repository, report, presentation | Several approaches compared fairly with the baseline and with each other on the same validation design; the final choice justified |
| Reliability of the results | 15 % | Report, independent evaluation, questions | A validation design without leakage, with its uncertainty; error analysis by language, heading frequency and chapter; a reported score that the independent evaluation confirms, or a convincing explanation of the gap |
| Delivery | 10 % | Live demonstration, evaluation run | An API that meets the interface, starts from the tag and answers within the limits; monitoring plan and model card |
| Collaboration and presentation (per member) | 20 % | Report Part B, Git history and pull requests, presentation, questions | Own commits and reviewed pull requests with passing CI; a clear presented part; correct and confident answers to individual questions |

**The independent score itself is not converted into marks.** A well-validated, well-explained model with 0.84 can earn a better grade than an unexplained one with 0.88. What counts is how well you know your own model.

## Questions and answers

**Why is there no training, validation and test split in the download?**
Because choosing it is part of the work. Random, stratified (by heading, or by heading and language, as the evaluation set), grouped by near-identical texts, or by date: each answers a different question. Sessions 7, 9 and 13 show the options and what stratification does with rare headings; your report says which one you took and why.

**What should I report as "my score"?**
The accuracy and macro-F1 of the design you submit, measured on decisions that played no part in building or tuning it, with an interval. If you tried many models, choose the final one before you look at that last measurement. Otherwise the score is too optimistic, because it also measures how well your choice fits that particular data.

**Can I train on all 102,440 decisions for the final service?**
Yes, and it is usually a good idea once the design is fixed. Report the score you measured before the final refit and say so.

**Will the evaluation set look like my data?**
Yes: the same kind of decisions, drawn at random from the same pool before the download was made, with the same mix of headings and languages. It contains no decisions from `ebti-v1.zip`.

**Is there a leaderboard during the course?**
No hidden scores during the course. At the checkpoints in Sessions 13 and 14 teams may post their own validation score and design (voluntary, but recommended), so you can compare approaches. The independent scores are revealed once, in Session 18.

**What if a heading in the evaluation set never occurs in my data?**
It can happen for a few rare headings. No model trained on `ebti-v1.zip` gets them right; it costs every team the same.

**Do I have to use a language model?**
No. A well-tuned TF-IDF model is a strong baseline (about 0.88 accuracy in Session 13). Use embeddings or a language model only if your validation shows that they help.

**How fast must the API be?**
Under two seconds for one request on a laptop. The evaluation sends about 17,000 requests one after another.

**What if the live demonstration fails?**
Have a recorded fallback ready. The evaluation run uses your tag, not the demonstration.

**May we use AI to write the report?**
Yes, for drafting and for code, as long as every member reviews the text, can explain every sentence and every line of code, and Part C of the report says what you used it for. Your individual entry in Part B must describe what you did yourself.

**Who do I ask?**
The lecturer, about the task, the data or the team work: open an issue in your team repository and mention the lecturer, or ask in class.

**Is anything graded before the presentation?**
No. Check-ins, checkpoints, the charter and the interim review are there for feedback. The grade comes only from the repository at its tag, the report and the presentation.
