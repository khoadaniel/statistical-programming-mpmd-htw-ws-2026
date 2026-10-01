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

- `PARTITION BY year` restricts the window to rows of the same year (like a group);
- `ORDER BY n DESC` sorts the rows inside the window;
- the function decides what is computed: `RANK()` numbers the rows (ties share a rank and leave a gap), `DENSE_RANK()` numbers without gaps, `ROW_NUMBER()` numbers without ties, `SUM(n) OVER (ORDER BY year)` is a **running total**, and `LAG(n)` returns the value of the previous row.

Window functions are evaluated after `WHERE`, `GROUP BY` and `HAVING`. A filter on their result, such as "rank 1 only", must therefore be placed one query level higher, typically in the next CTE or in the outer query.

A worked example by hand. Counts of reviews per product and year:

| year | product | n | `RANK() OVER (PARTITION BY year ORDER BY n DESC)` | `DENSE_RANK()` |
|---|---|---|---|---|
| 2020 | A | 80 | 1 | 1 |
| 2020 | B | 80 | 1 | 1 |
| 2020 | C | 50 | 3 | 2 |
| 2021 | A | 90 | 1 | 1 |
| 2021 | C | 60 | 2 | 2 |

The rank restarts in each year (partition). A and B tie in 2020; `RANK` then skips to 3, `DENSE_RANK` continues with 2.

```mermaid
flowchart TB
    T["reviews<br/>434,373 rows"] --> G["CTE per_year:<br/>GROUP BY year, product"]
    G --> W["CTE ranked:<br/>RANK() OVER (PARTITION BY year<br/>ORDER BY n DESC)"]
    W --> F["outer query:<br/>WHERE rnk &lt;= 3"]
    F --> J["JOIN products<br/>for title and store"]
```

### Why it matters

Rankings within groups, running totals, changes against the previous period and moving averages are standard analytical questions. Without window functions they need self-joins or loops in Python; with them they are one readable query that the database executes over all rows.

### How it works in Python

The practice query ranks products by reviews per year (also in [`workbooks/sql/03-rank-products-per-year.sql`](../workbooks/sql/03-rank-products-per-year.sql)):

```sql
WITH per_year AS (                    -- step 1: one row per year and product
    SELECT EXTRACT(YEAR FROM date)::int AS year, parent_asin, COUNT(*) AS n
    FROM reviews
    GROUP BY year, parent_asin
), ranked AS (                        -- step 2: rank inside each year
    SELECT *, RANK() OVER (PARTITION BY year ORDER BY n DESC) AS rnk
    FROM per_year
)
SELECT r.year, r.n, LEFT(p.title, 30) AS top_product
FROM ranked AS r
JOIN products AS p USING (parent_asin)
WHERE r.rnk = 1 AND r.year >= 2019    -- filter on the window result one level up
ORDER BY r.year;
-- 2019 | 526 | Nerdwax Stop Slipping Glasses
-- 2020 | 824 | Buttonsmith Black Adult Cotton
-- 2021 | 946 | JUNP Hydration Electrolyte Pow
```

Running totals and year-over-year changes:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW reviews AS SELECT * FROM 'case-study/data/train.parquet'")
print(con.sql("""
    WITH yearly AS (
        SELECT EXTRACT(YEAR FROM date)::int AS year, COUNT(*) AS n
        FROM reviews GROUP BY year
    )
    SELECT year, n,
           SUM(n) OVER (ORDER BY year)                              AS running_total,
           n - LAG(n) OVER (ORDER BY year)                          AS change,
           ROUND(AVG(n) OVER (ORDER BY year ROWS 2 PRECEDING))      AS moving_avg_3y
    FROM yearly
    ORDER BY year DESC
    LIMIT 3
""").df())
#    year      n  running_total  change  moving_avg_3y
# 0  2021  68456       434373.0   -7448        67318.0
# 1  2020  75904       365917.0   18310        61787.0
# 2  2019  57594       290013.0    5732        55523.0
```

![Reviews per year with the running total and the year-over-year change computed by window functions](figures/reviews-per-year-window.png)

The same window logic in pandas, for comparison: `cumsum()` is a running `SUM`, `diff()` is `n - LAG(n)`, and `groupby(...).rank(method="min", ascending=False)` is `RANK() OVER (PARTITION BY ...)`.

```python
import pandas as pd

reviews = pd.read_parquet("case-study/data/train.parquet", columns=["parent_asin", "date"])
per_year = (reviews.assign(year=reviews["date"].dt.year)
            .groupby(["year", "parent_asin"]).size().rename("n").reset_index())
per_year["rnk"] = per_year.groupby("year")["n"].rank(method="min", ascending=False)
print(per_year.query("rnk == 1 and year >= 2019")[["year", "parent_asin", "n"]].to_string(index=False))
# year parent_asin   n
# 2019  B00O0CK2UM 526
# 2020  B08K2MYBZY 824
# 2021  B0CB33QW6H 946
```

### In practice

- **E-commerce reporting** lists the top products per category and month; Amazon's own "Best Sellers" pages are rankings within categories that are recomputed regularly.
- **Finance teams** compute month-over-month growth and cumulative revenue with `LAG` and running sums for management reports.
- **Health-services research** with claims data uses `LAG` over the admissions of each patient to measure the time to readmission, the basis of readmission indicators such as those of the US Hospital Readmissions Reduction Program.

> [!WARNING]
> `RANK`, `DENSE_RANK` and `ROW_NUMBER` differ only when there are ties, so a bug may go unnoticed for a long time. Decide explicitly how ties should be handled, and add a second sort key (for example `ORDER BY n DESC, parent_asin`) when you need a reproducible order.

> [!CAUTION]
> `WHERE RANK() OVER (...) = 1` is an error: `WHERE` is evaluated before window functions. Compute the rank in a CTE and filter in the next step.

## Access from Python with SQLAlchemy and pandas

### Concept

**SQLAlchemy** is the standard Python toolkit for relational databases. `create_engine(url)` creates an **engine**, an object that opens and reuses connections. The URL names the **dialect** (which database) and the **driver** (which Python package talks to it), followed by user, password, host, port and database:

```
postgresql+psycopg://course:course@localhost:5432/course
└─dialect─┘ └driver┘ └user┘ └pw─┘ └─host──┘ └port┘└─db─┘
```

`pandas.read_sql(query, engine)` runs a query and returns a DataFrame; `DataFrame.to_sql(name, engine)` writes a DataFrame into a table. Values that change between runs are passed as **bound parameters**: in `text("... WHERE date >= :start")` the placeholder `:start` is filled from `params={"start": "2019-01-01"}`. The driver sends the values separately from the SQL text, so they can never be interpreted as SQL.

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

The database does the filtering, joining and aggregation on all rows; Python receives a small result for plotting and modelling. Parameters protect against **SQL injection**, an attack in which text entered by a user changes the meaning of a query. With string formatting, a "product id" such as `x' OR '1'='1` would return every row.

### How it works in Python

With PostgreSQL (after starting the server, see the next section):

```python
# requires a running PostgreSQL server and the environment variable DATABASE_URL
import os

import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])   # never hard-code passwords
query = text("""
    SELECT EXTRACT(YEAR FROM date)::int AS year, COUNT(*) AS n_reviews, AVG(rating) AS avg_rating
    FROM reviews
    WHERE verified_purchase = :verified AND date >= :start
    GROUP BY year ORDER BY year
""")
yearly = pd.read_sql(query, engine, params={"verified": True, "start": "2019-01-01"})
print(yearly.round(2))   # 2019: 54031, 4.08 | 2020: 70229, 3.95 | 2021: 62090, 3.88
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
    con.exec_driver_sql("CREATE TABLE reviews AS SELECT * FROM 'case-study/data/train.parquet'")

yearly = pd.read_sql(text("""
    SELECT EXTRACT(YEAR FROM date)::int AS year, COUNT(*) AS n_reviews, AVG(rating) AS avg_rating
    FROM reviews
    WHERE verified_purchase = :verified AND date >= :start
    GROUP BY year ORDER BY year
"""), engine, params={"verified": True, "start": "2019-01-01"})
print(yearly.round(2))
#    year  n_reviews  avg_rating
# 0  2019      54031        4.08
# 1  2020      70229        3.95
# 2  2021      62090        3.88

with engine.connect() as con:    # a single value, without pandas
    n = con.execute(text("SELECT COUNT(*) FROM reviews WHERE rating = :r"), {"r": 3}).scalar_one()
print(n)                                                             # 32482
```

### In practice

- **Analysts** in most companies pull aggregated data from a data warehouse into notebooks with exactly this pattern; the heavy work stays in the database.
- **Web applications** written in Python, such as the FastAPI service in Session 16, use SQLAlchemy to read and write their database.
- **SQL injection** is part of the "Injection" category of the OWASP Top 10 web application security risks; the 2008 breach of the card processor Heartland Payment Systems, one of the largest of its time, started with an SQL injection.

> [!CAUTION]
> Never build SQL with f-strings from values that come from users, files or the web: `f"... WHERE parent_asin = '{asin}'"`. Use bound parameters. Table and column names cannot be parameters; if they must vary, check them against a fixed list.

> [!TIP]
> `pd.read_sql` loads the whole result into memory. For large results, aggregate in SQL first, or read in pieces with `chunksize=` and process each piece.

## Loading data reproducibly with an ingestion script and constraints

### Concept

**Ingestion** is the step that brings data from their source into your database. It is **reproducible** if running the same script on the same source always gives the same tables, without manual steps in between. A good ingestion script:

1. downloads or reads the raw data from a documented source;
2. applies the same transformations every time (in the case study: hashing user ids, removing duplicates, splitting by date);
3. creates the tables with **keys and constraints**, so that invalid rows are rejected;
4. loads parent tables before child tables (products before reviews), because the foreign key requires the parent row to exist;
5. checks the result: row counts against the source, and no rejected rows.

The case-study script `case-study/prepare_data.py` does steps 1, 2 and 4 and can write the tables to PostgreSQL with `--postgres`. Because `pandas.to_sql` creates tables without keys, the keys are added afterwards with [`workbooks/sql/02-add-constraints.sql`](../workbooks/sql/02-add-constraints.sql); alternatively, the case-study notebook creates the schema first with [`01-schema.sql`](../workbooks/sql/01-schema.sql) and then loads the rows.

```mermaid
stateDiagram-v2
    [*] --> Downloaded: download raw files
    Downloaded --> Built: build tables (dedupe, split)
    Built --> Loaded: load products, then reviews
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

uv run --with pandas --with pyarrow --with sqlalchemy --with "psycopg[binary]" \
    python case-study/prepare_data.py --postgres $DATABASE_URL
psql $DATABASE_URL -f sessions/03-sql-and-polars/workbooks/sql/02-add-constraints.sql   # or from Python
```

> [!NOTE]
> `psql` expects a plain `postgresql://` URL; for psql, write `postgresql://course:course@localhost:5432/course`. SQLAlchemy needs the `+psycopg` part to choose the driver.

The core of a schema-first loader, here with DuckDB so that it runs anywhere. The constraint rejects an invalid row, and the error message names the rule:

```python
import duckdb

con = duckdb.connect()
con.execute("""
    CREATE TABLE products (parent_asin TEXT PRIMARY KEY, title TEXT NOT NULL, price DOUBLE CHECK (price > 0));
    CREATE TABLE reviews (
        review_id   TEXT PRIMARY KEY,
        parent_asin TEXT NOT NULL REFERENCES products (parent_asin),
        rating      SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5)
    );
""")
con.execute("INSERT INTO products SELECT parent_asin, title, price FROM 'case-study/data/products.parquet'")
con.execute("INSERT INTO reviews SELECT review_id, parent_asin, rating FROM 'case-study/data/train.parquet'")
print(con.sql("SELECT (SELECT COUNT(*) FROM products), (SELECT COUNT(*) FROM reviews)").fetchone())
# (60274, 434373): the counts match the source files

try:
    con.execute("INSERT INTO reviews VALUES ('r999999', 'B000050FEQ', 6)")
except duckdb.ConstraintException as e:
    print(str(e).splitlines()[0])
# Constraint Error: CHECK constraint failed on table reviews with expression CHECK((rating BETWEEN 1 AND 5))
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

Some facts can be computed (row counts, date range, shares of missing values, label distribution); others need judgement (who is under-represented, which questions the data cannot answer). For the review data, a limitation is **selection**: only customers who chose to write a review are represented, and very satisfied and very dissatisfied customers write reviews more often than others.

### Why it matters

Without documentation, knowledge about a dataset lives in the heads of the people who prepared it. Later users repeat mistakes (such as using a column that leaks the target, see Session 9), use data outside their licence, or draw conclusions about populations the data do not cover. A data card also forces the team to look at the data before modelling.

### How it works in Python

The computed part of the card comes from a few queries:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW reviews AS SELECT * FROM 'case-study/data/train.parquet'")
con.execute("CREATE VIEW products AS SELECT * FROM 'case-study/data/products.parquet'")
facts = con.sql("""
    SELECT COUNT(*) AS n_reviews,
           COUNT(DISTINCT parent_asin) AS n_products_reviewed,
           MIN(date)::date AS first_review, MAX(date)::date AS last_review,
           ROUND(AVG(verified_purchase::int), 3) AS share_verified,
           ROUND(AVG((label = 'neu')::int), 3) AS share_neutral,
           (SELECT ROUND(AVG((price IS NULL)::int), 3) FROM products) AS share_price_missing
    FROM reviews
""").df().T
print(facts)
#                                        0
# n_reviews                434373
# n_products_reviewed       55359
# first_review         2001-02-05 00:00:00
# last_review          2021-12-31 00:00:00
# share_verified            0.906
# share_neutral             0.075
# share_price_missing       0.825
```

### In practice

- **Hugging Face** shows a dataset card on the page of every dataset on its Hub; the card of a dataset is the first thing users see.
- **Google** introduced *Data Cards* for its own datasets and published the template and the experience from using it (Pushkarna et al., 2022).
- **Public open-data portals**, for example GovData in Germany, publish metadata following the DCAT-AP standard: source, licence, update frequency and contact for every dataset.

> [!IMPORTANT]
> The review data have no stated licence from the McAuley Lab, and the course does not redistribute them (see `case-study/README.md`). A data card must record such terms; "found on the internet" is not a licence.

## Check your understanding

1. Rewrite "top three products per year by number of reviews" as two CTEs and an outer query. In which step does the window function appear, and in which step the filter?
2. When do `RANK()` and `ROW_NUMBER()` give different results?
3. Why is `text("... WHERE store = :store")` with `params={"store": s}` safer than an f-string?
4. The script writes the tables with `to_sql(if_exists="replace")`. What is missing afterwards, and how do you add it?
5. Name two facts of a data card that can be computed by a query and two that need human judgement.

## Further reading

- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92. https://arxiv.org/abs/1803.09010
- The PostgreSQL Global Development Group. *PostgreSQL documentation: Window functions* (tutorial section 3.5). https://www.postgresql.org/docs/current/tutorial-window.html
- SQLAlchemy. *SQLAlchemy Unified Tutorial* (version 2.0). https://docs.sqlalchemy.org/en/20/tutorial/
- Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data cards: Purposeful and transparent dataset documentation for responsible AI. *Proceedings of FAccT 2022*. https://arxiv.org/abs/2204.01075
