# Advanced SQL, Python access and reproducible loading

This page covers the second block: common table expressions and window functions for questions that plain `GROUP BY` cannot answer; access to a database from Python with SQLAlchemy and pandas; loading data reproducibly with an ingestion script and constraints; and documenting a dataset with a data card. Together these steps turn a folder of files into a shared, documented team database, which is the team-project task for this week.

```mermaid
flowchart LR
    RAW["Raw files<br/>(download)"] --> SCRIPT["Ingestion script<br/>prepare_data.py"]
    SCRIPT --> SCHEMA["Schema with keys<br/>and constraints"]
    SCHEMA --> DB[("PostgreSQL")]
    DB --> CHECK["Checks: row counts,<br/>rejected rows"]
    CHECK --> CARD["Data card"]
    DB --> PY["Python: SQLAlchemy,<br/>pandas.read_sql"]
```

## Common table expressions and window functions

### Concept

A **common table expression** (CTE), written `WITH name AS (SELECT ...)`, gives an intermediate result a name. Later parts of the query use it like a table. Several CTEs separated by commas turn a long query into a sequence of readable steps, much like intermediate variables in Python.

A **window function** computes a value for each row from a set of related rows, the **window**, without collapsing them into one row. This is the difference from `GROUP BY`, which returns one row per group. The window is defined after `OVER`:

- `PARTITION BY issuing_country, year` restricts the window to rows of the same country and year (like a group);
- `ORDER BY n DESC` sorts the rows inside the window;
- the function decides what is computed: `RANK()` numbers the rows (ties share a rank and leave a gap), `DENSE_RANK()` numbers without gaps, `ROW_NUMBER()` numbers without ties, `SUM(n) OVER (ORDER BY year)` is a **running total**, and `LAG(n)` returns the value of the previous row.

Window functions are evaluated after `WHERE`, `GROUP BY` and `HAVING`. A filter on their result, such as "rank 1 only", must therefore be placed one query level higher, typically in the next CTE or in the outer query.

A worked example by hand. Counts of decisions per heading for one country and two years:

| year | heading | n | `RANK() OVER (PARTITION BY year ORDER BY n DESC)` | `DENSE_RANK()` |
|---|---|---|---|---|
| 2020 | 9503 | 80 | 1 | 1 |
| 2020 | 3926 | 80 | 1 | 1 |
| 2020 | 6307 | 50 | 3 | 2 |
| 2021 | 6307 | 90 | 1 | 1 |
| 2021 | 3926 | 60 | 2 | 2 |

The rank restarts in each year (partition). 9503 and 3926 tie in 2020; `RANK` then skips to 3, `DENSE_RANK` continues with 2.

```mermaid
flowchart TB
    T["decisions<br/>309,529 rows"] --> G["CTE per_year:<br/>GROUP BY country, year, heading"]
    G --> W["CTE ranked:<br/>RANK() OVER (PARTITION BY<br/>country, year ORDER BY n DESC)"]
    W --> F["outer query:<br/>WHERE rnk = 1"]
    F --> J["LEFT JOIN nomenclature<br/>for the English name"]
```

### Why it matters

Rankings within groups, running totals, changes against the previous period and moving averages are standard analytical questions. Without window functions they need self-joins or loops in Python; with them they are one readable query that the database executes over all rows.

### How it works in Python

The practice query ranks headings by the number of decisions per issuing country and year (also in [`workbooks/sql/03-rank-headings-per-country-year.sql`](../workbooks/sql/03-rank-headings-per-country-year.sql)):

```sql
WITH per_year AS (                    -- step 1: one row per country, year and heading
    SELECT issuing_country, EXTRACT(YEAR FROM start_date)::int AS year, heading, COUNT(*) AS n
    FROM decisions
    GROUP BY issuing_country, year, heading
), ranked AS (                        -- step 2: rank inside each country and year
    SELECT *, RANK() OVER (PARTITION BY issuing_country, year ORDER BY n DESC) AS rnk
    FROM per_year
)
SELECT r.issuing_country, r.year, r.heading, r.n, LEFT(n.heading_description, 35) AS heading_description
FROM ranked AS r
LEFT JOIN nomenclature AS n ON n.heading = r.heading
WHERE r.rnk = 1 AND r.issuing_country IN ('DE', 'FR') AND r.year >= 2019   -- filter one level up
ORDER BY r.issuing_country, r.year;
-- DE | 2019 | 3926 | 1083 | Articles of plastics and articles o
-- DE | 2020 | 9503 |  894 | Tricycles, scooters, pedal cars and
-- DE | 2021 | 6307 |  852 | Textiles; made up articles n.e.c. i
-- DE | 2022 | 6307 |  860 | Textiles; made up articles n.e.c. i
-- DE | 2023 | 3926 |  964 | Articles of plastics and articles o
-- FR | 2019 | 2106 |  264 | Food preparations not elsewhere spe
-- FR | 2020 | 3926 |  317 | Articles of plastics and articles o
-- FR | 2021 | 3926 |  341 | Articles of plastics and articles o
-- FR | 2022 | 9405 |  315 | Luminaires and light fittings; incl
-- FR | 2023 | 3926 |  250 | Articles of plastics and articles o
```

Running totals and month-over-month changes, here on the table `monthly_counts` (decisions per month, issuing country and chapter, 2004–2026) for chapter 63 (made-up textile articles, including face masks) in the first half of 2020:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW monthly_counts AS SELECT * FROM 'case-study/data/monthly_counts.parquet'")
print(con.sql("""
    WITH monthly AS (
        SELECT month::date AS month, SUM(n_decisions) AS n
        FROM monthly_counts
        WHERE chapter = '63' AND month BETWEEN '2020-01-01' AND '2020-06-01'
        GROUP BY month
    )
    SELECT month, n,
           SUM(n) OVER (ORDER BY month)                         AS running_total,
           n - LAG(n) OVER (ORDER BY month)                     AS change,
           ROUND(AVG(n) OVER (ORDER BY month ROWS 2 PRECEDING)) AS moving_avg_3m
    FROM monthly
    ORDER BY month
""").df())
#         month      n  running_total  change  moving_avg_3m
# 0  2020-01-01  104.0          104.0     NaN          104.0
# 1  2020-02-01  136.0          240.0    32.0          120.0
# 2  2020-03-01  186.0          426.0    50.0          142.0
# 3  2020-04-01  100.0          526.0   -86.0          141.0
# 4  2020-05-01   91.0          617.0    -9.0          126.0
# 5  2020-06-01  105.0          722.0    14.0           99.0
```

`LAG` has no previous row for January, so the change is `NULL` (shown as `NaN` in pandas). The moving average uses fewer than three months at the start.

![Decisions per start year 2004-2025 with the year-over-year change computed by a window function](figures/decisions-per-year-window.png)

The same window logic in pandas, for comparison: `cumsum()` is a running `SUM`, `diff()` is `n - LAG(n)`, and `groupby(...).rank(method="min", ascending=False)` is `RANK() OVER (PARTITION BY ...)`.

```python
import pandas as pd

decisions = pd.read_parquet("case-study/data/train.parquet", columns=["issuing_country", "start_date", "heading"])
per_year = (decisions.assign(year=decisions["start_date"].dt.year)
            .groupby(["issuing_country", "year", "heading"]).size().rename("n").reset_index())
per_year["rnk"] = per_year.groupby(["issuing_country", "year"])["n"].rank(method="min", ascending=False)
print(per_year.query("rnk == 1 and issuing_country == 'DE' and year >= 2021")
      [["year", "heading", "n"]].to_string(index=False))
# year heading   n
# 2021    6307 852
# 2022    6307 860
# 2023    3926 964
```

### In practice

- **Trade statistics** are often published as rankings, for example the most important import goods of a country per year; each such list is a ranking within a partition (country, year).
- **Finance teams** compute month-over-month growth and cumulative revenue with `LAG` and running sums for management reports.
- **Health-services research** with claims data uses `LAG` over the admissions of each patient to measure the time to readmission, the basis of readmission indicators such as those of the US Hospital Readmissions Reduction Program.

> [!WARNING]
> `RANK`, `DENSE_RANK` and `ROW_NUMBER` differ only when there are ties, so a bug may go unnoticed for a long time. Decide explicitly how ties should be handled, and add a second sort key (for example `ORDER BY n DESC, heading`) when you need a reproducible order.

> [!CAUTION]
> `WHERE RANK() OVER (...) = 1` is an error: `WHERE` is evaluated before window functions. Compute the rank in a CTE and filter in the next step.

## Access from Python with SQLAlchemy and pandas

### Concept

**SQLAlchemy** is the standard Python toolkit for relational databases. `create_engine(url)` creates an **engine**, an object that opens and reuses connections. The URL names the **dialect** (which database) and the **driver** (which Python package talks to it), followed by user, password, host, port and database:

```
postgresql+psycopg://course:course@localhost:5432/course
└─dialect─┘ └driver┘ └user┘ └pw─┘ └─host──┘ └port┘└─db─┘
```

`pandas.read_sql(query, engine)` runs a query and returns a DataFrame; `DataFrame.to_sql(name, engine)` writes a DataFrame into a table. Values that change between runs are passed as **bound parameters**: in `text("... WHERE start_date >= :start")` the placeholder `:start` is filled from `params={"start": "2021-01-01"}`. The driver sends the values separately from the SQL text, so they can never be interpreted as SQL.

```mermaid
sequenceDiagram
    participant N as Notebook
    participant E as SQLAlchemy engine
    participant D as PostgreSQL
    N->>E: read_sql(text(query), params)
    E->>D: SQL text + parameter values
    D-->>E: result rows
    E-->>N: pandas DataFrame
```

### Why it matters

The database does the filtering, joining and aggregation on all rows; Python receives a small result for plotting and modelling. Parameters protect against **SQL injection**, an attack in which text entered by a user changes the meaning of a query. With string formatting, a "country code" such as `x' OR '1'='1` would return every row.

### How it works in Python

With PostgreSQL (after starting the server, see the next section):

```python
# requires a running PostgreSQL server and the environment variable DATABASE_URL
import os

import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])   # never hard-code passwords
query = text("""
    SELECT EXTRACT(YEAR FROM start_date)::int AS year, COUNT(*) AS n_decisions,
           COUNT(DISTINCT heading) AS n_headings
    FROM decisions
    WHERE issuing_country = :country AND start_date >= :start
    GROUP BY year ORDER BY year
""")
yearly = pd.read_sql(query, engine, params={"country": "FR", "start": "2021-01-01"})
print(yearly)   # 2021: 7843, 579 | 2022: 6995, 564 | 2023: 7133, 585
```

The same code runs against DuckDB through the `duckdb-engine` dialect, which is how the case-study notebook works without a server:

```python
import tempfile
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

db = Path(tempfile.mkdtemp()) / "course.duckdb"
engine = create_engine(f"duckdb:///{db}")
with engine.begin() as con:   # begin(): one transaction, committed at the end
    con.exec_driver_sql("CREATE TABLE decisions AS SELECT * FROM 'case-study/data/train.parquet'")

yearly = pd.read_sql(text("""
    SELECT EXTRACT(YEAR FROM start_date)::int AS year, COUNT(*) AS n_decisions,
           COUNT(DISTINCT heading) AS n_headings
    FROM decisions
    WHERE issuing_country = :country AND start_date >= :start
    GROUP BY year ORDER BY year
"""), engine, params={"country": "FR", "start": "2021-01-01"})
print(yearly)
#    year  n_decisions  n_headings
# 0  2021         7843         579
# 1  2022         6995         564
# 2  2023         7133         585

with engine.connect() as con:    # a single value, without pandas
    n = con.execute(text("SELECT COUNT(*) FROM decisions WHERE heading = :h"), {"h": "9503"}).scalar_one()
print(n)                                                             # 8883
```

### In practice

- **Analysts** in most companies pull aggregated data from a data warehouse into notebooks with exactly this pattern; the heavy work stays in the database.
- **Web applications** written in Python, such as the FastAPI service in Session 16, use SQLAlchemy to read and write their database.
- **SQL injection** is part of the "Injection" category of the OWASP Top 10 web application security risks; the 2008 breach of the card processor Heartland Payment Systems, one of the largest of its time, started with an SQL injection.

> [!CAUTION]
> Never build SQL with f-strings from values that come from users, files or the web: `f"... WHERE heading = '{heading}'"`. Use bound parameters. Table and column names cannot be parameters; if they must vary, check them against a fixed list.

> [!TIP]
> `pd.read_sql` loads the whole result into memory. For large results, aggregate in SQL first, or read in pieces with `chunksize=` and process each piece.

## Loading data reproducibly with an ingestion script and constraints

### Concept

**Ingestion** is the step that brings data from their source into your database. It is **reproducible** if running the same script on the same source always gives the same tables, without manual steps in between. A good ingestion script:

1. downloads or reads the raw data from a documented source;
2. applies the same transformations every time (in the case study: parsing the dates, deriving the heading from the code, masking quoted codes, splitting by date);
3. creates the tables with **keys and constraints**, so that invalid rows are rejected;
4. loads parent tables before child tables (the nomenclature before the decisions), because the foreign key requires the parent row to exist;
5. checks the result: row counts against the source, and no rejected rows.

The case-study script `case-study/prepare_data.py` does steps 1 and 2 and can write the tables `decisions`, `decisions_test` and `nomenclature` to PostgreSQL with `--postgres`. Because `pandas.to_sql` creates tables without keys, the keys are added afterwards with [`workbooks/sql/02-add-constraints.sql`](../workbooks/sql/02-add-constraints.sql); alternatively, the case-study notebook creates the schema first with [`01-schema.sql`](../workbooks/sql/01-schema.sql) and then loads the rows.

```mermaid
stateDiagram-v2
    [*] --> Downloaded: download raw files
    Downloaded --> Built: build tables (dedupe, split)
    Built --> Loaded: load nomenclature, then decisions
    Loaded --> Constrained: add keys and checks
    Constrained --> Verified: counts match, no errors
    Loaded --> Failed: constraint violated
    Failed --> Built: fix the data or the rule
    Verified --> [*]
```

### Why it matters

Manual loading ("I imported the CSV in a graphical tool and fixed a few rows") cannot be repeated or reviewed, and nobody knows later which version of the data an analysis used. A script can be rerun when the source is updated, reviewed in a pull request and run by every team member. Constraints turn silent data problems into loud errors at load time.

### How it works in Python

Start PostgreSQL in Docker (one command; the container keeps running in the background):

```bash
docker run --name course-db -e POSTGRES_USER=course -e POSTGRES_PASSWORD=course \
    -e POSTGRES_DB=course -p 5432:5432 -d postgres:17
export DATABASE_URL=postgresql+psycopg://course:course@localhost:5432/course

uv run python case-study/prepare_data.py --postgres $DATABASE_URL   # several minutes
psql $DATABASE_URL -f sessions/03-sql-and-polars/workbooks/sql/02-add-constraints.sql   # or from Python
```

> [!NOTE]
> `psql` expects a plain `postgresql://` URL; for psql, write `postgresql://course:course@localhost:5432/course`. SQLAlchemy needs the `+psycopg` part to choose the driver.

The core of a schema-first loader, here with DuckDB so that it runs anywhere. The foreign key rejects the decisions of the deleted heading 8803 (see [block 1](01-relational-model-and-sql.md#how-it-works-in-python)), and the error message names the offending key:

```python
import duckdb

con = duckdb.connect()
con.execute("""
    CREATE TABLE nomenclature (
        heading             TEXT PRIMARY KEY CHECK (length(heading) = 4),
        heading_description TEXT NOT NULL
    );
    CREATE TABLE decisions (
        bti_reference TEXT PRIMARY KEY,
        heading       TEXT NOT NULL REFERENCES nomenclature (heading),
        language      TEXT NOT NULL CHECK (length(language) = 2),
        start_date    DATE NOT NULL
    );
""")
con.execute("INSERT INTO nomenclature SELECT heading, heading_description FROM 'case-study/data/nomenclature.parquet'")
try:
    con.execute("""INSERT INTO decisions
                   SELECT bti_reference, heading, language, start_date FROM 'case-study/data/train.parquet'""")
except duckdb.ConstraintException as e:
    print(str(e).splitlines()[0])
# Constraint Error: Violates foreign key constraint because key "heading: 8803" does not exist in the referenced table
print(con.sql("SELECT COUNT(*) FROM decisions").fetchone())   # (0,): the whole statement was rolled back

# a decision, documented in the data card: load only decisions with a known heading
con.execute("""INSERT INTO decisions
               SELECT bti_reference, heading, language, start_date FROM 'case-study/data/train.parquet'
               WHERE heading IN (SELECT heading FROM nomenclature)""")
print(con.sql("SELECT (SELECT COUNT(*) FROM nomenclature), (SELECT COUNT(*) FROM decisions)").fetchone())
# (1229, 309478): 51 decisions fewer than in the file, and we know exactly which ones
```

### In practice

- **GitLab** published a detailed post-mortem of a 2017 incident in which production PostgreSQL data were deleted by mistake and several backup procedures turned out not to work; restoring depended on a copy that happened to exist. Scripted, tested loading and restoring is the lesson.
- **Public Health England** lost about 16,000 positive COVID-19 test results in October 2020 because a manual step used an old Excel file format whose row limit was exceeded; an automated pipeline with row-count checks would have caught it.
- **Data engineering teams** use tools such as dbt, which runs SQL transformations from version-controlled files and checks `unique` and `not_null` rules after each run.

> [!WARNING]
> `to_sql(..., if_exists="replace")` drops the table and recreates it **without** keys, constraints or indexes. After a reload, run the constraints script again, or load with `if_exists="append"` into a table that you created with its schema.

> [!TIP]
> With PostgreSQL, `to_sql(..., method="multi", chunksize=5_000)` sends many rows per statement. A statement may hold at most 65,535 parameters, so `chunksize × number of columns` must stay below that limit.

## Documenting a dataset (data card)

### Concept

A **data card** is a short document that travels with a dataset and answers the questions a new user would ask: what is in it, where does it come from, how was it processed, what is it for, and what are its limitations. The idea was proposed as *Datasheets for Datasets* by Gebru et al. (2021), who compared it to the datasheets that accompany electronic components. Hugging Face dataset cards and Google's *Data Cards* (Pushkarna et al., 2022) are widely used variants.

The course template ([`workbooks/data-card-template.md`](../workbooks/data-card-template.md)) has these sections:

```mermaid
mindmap
  root((Data card))
    Summary
      name, version, owner
    Motivation
      why created
      our question
    Source
      origin, licence
      time period
      who is missing
    Composition
      tables, keys
      label, missing values
      personal data
    Processing
      script, filters
      split
    Uses
      intended
      to avoid
    Limitations
```

Some facts can be computed (row counts, date range, shares of missing values, label distribution); others need judgement (who is under-represented, which questions the data cannot answer). For the BTI decisions, a limitation is **selection**: a decision exists only where a trader was unsure enough to ask for one, so products whose classification is obvious (live animals, ores) are almost absent, while toys, plastics articles and food preparations are frequent. A second limitation is the **changing nomenclature**: the 2022 revision deleted and created headings.

### Why it matters

Without documentation, knowledge about a dataset lives in the heads of the people who prepared it. Later users repeat mistakes (such as using a column that leaks the target, see Session 9), use data outside their licence, or draw conclusions about populations the data do not cover. A data card also forces the team to look at the data before modelling.

### How it works in Python

The computed part of the card comes from a few queries:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW decisions AS SELECT * FROM 'case-study/data/train.parquet'")
con.execute("CREATE VIEW decisions_test AS SELECT * FROM 'case-study/data/test.parquet'")
facts = con.sql("""
    SELECT COUNT(*) AS n_decisions,
           COUNT(DISTINCT heading) AS n_headings,
           MIN(start_date)::date AS first_start, MAX(start_date)::date AS last_start,
           ROUND(AVG((language = 'de')::int), 3) AS share_german,
           ROUND(AVG((status = 'VALID')::int), 3) AS share_valid,
           ROUND(AVG((keywords IS NULL)::int), 4) AS share_keywords_missing,
           (SELECT COUNT(*) FROM decisions_test) AS n_test_decisions
    FROM decisions
""").df().T
print(facts)
#                                           0
# n_decisions                          309529
# n_headings                             1114
# first_start             2017-01-01 00:00:00
# last_start              2023-12-31 00:00:00
# share_german                          0.572
# share_valid                            0.03
# share_keywords_missing               0.0041
# n_test_decisions                     113188
```

### In practice

- **Hugging Face** shows a dataset card on the page of every dataset on its Hub; the card of a dataset is the first thing users see.
- **Google** introduced *Data Cards* for its own datasets and published the template and the experience from using it (Pushkarna et al., 2022).
- **Public open-data portals**, for example GovData in Germany, publish metadata following the DCAT-AP standard: source, licence, update frequency and contact for every dataset.

> [!IMPORTANT]
> The EBTI data may be reused with acknowledgement of the source under the Commission's reuse policy (Commission Decision 2011/833/EU); the HS nomenclature table is in the public domain (ODC-PDDL). A data card must record such terms; "found on the internet" is not a licence.

## Check your understanding

1. Rewrite "top three headings per year by number of decisions" as two CTEs and an outer query. In which step does the window function appear, and in which step the filter?
2. When do `RANK()` and `ROW_NUMBER()` give different results?
3. Why is `text("... WHERE issuing_country = :country")` with `params={"country": c}` safer than an f-string?
4. The script writes the tables with `to_sql(if_exists="replace")`. What is missing afterwards, and why can the foreign key from `decisions` to `nomenclature` not simply be added?
5. Name two facts of a data card that can be computed by a query and two that need human judgement.

## Further reading

- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92. https://arxiv.org/abs/1803.09010
- The PostgreSQL Global Development Group. *PostgreSQL documentation: Window functions* (tutorial section 3.5). https://www.postgresql.org/docs/current/tutorial-window.html
- SQLAlchemy. *SQLAlchemy Unified Tutorial* (version 2.0). https://docs.sqlalchemy.org/en/20/tutorial/
- Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data cards: Purposeful and transparent dataset documentation for responsible AI. *Proceedings of FAccT 2022*. https://arxiv.org/abs/2204.01075
