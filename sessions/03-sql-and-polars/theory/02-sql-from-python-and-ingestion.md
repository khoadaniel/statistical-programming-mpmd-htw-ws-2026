# Advanced SQL, Python access and reproducible loading

This page covers the second block. The housing analyst from block 1 now asks follow-up questions that plain `GROUP BY` cannot answer: which district is the most expensive *for each room type*, how did the number of reviews develop month by month, and how does a single listing compare with its district? Common table expressions and window functions answer them. Then the analysis moves into a shared team database: access from Python with SQLAlchemy and pandas, loading the data reproducibly with an ingestion script and constraints, and documenting the dataset with a data card. Together these steps turn a folder of files into a shared, documented team database, which is the team-project task for this week.

```mermaid
flowchart LR
    RAW["Raw files<br/>(Inside Airbnb)"] --> SCRIPT["Ingestion script<br/>prepare_airbnb.py"]
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

- `PARTITION BY room_type` restricts the window to rows of the same room type (like a group);
- `ORDER BY median_price DESC` sorts the rows inside the window;
- the function decides what is computed: `RANK()` numbers the rows (ties share a rank and leave a gap), `DENSE_RANK()` numbers without gaps, `ROW_NUMBER()` numbers without ties, `SUM(n) OVER (ORDER BY month)` is a **running total**, `AVG(x) OVER (PARTITION BY district)` puts the district average next to each listing, and `LAG(n)` returns the value of the previous row.

Window functions are evaluated after `WHERE`, `GROUP BY` and `HAVING`. A filter on their result, such as "rank 1 only", must therefore be placed one query level higher, typically in the next CTE or in the outer query.

A worked example by hand. Median short-stay prices per room type and district (invented numbers):

| room_type | district | median_price | `RANK() OVER (PARTITION BY room_type ORDER BY median_price DESC)` | `DENSE_RANK()` |
|---|---|---|---|---|
| Entire home/apt | Mitte | 210 | 1 | 1 |
| Entire home/apt | Pankow | 210 | 1 | 1 |
| Entire home/apt | Neukölln | 160 | 3 | 2 |
| Private room | Mitte | 115 | 1 | 1 |
| Private room | Neukölln | 95 | 2 | 2 |

The rank restarts for each room type (partition). Mitte and Pankow tie for entire homes; `RANK` then skips to 3, `DENSE_RANK` continues with 2.

```mermaid
flowchart TB
    T["listings<br/>12,776 rows"] --> S["CTE short_stays:<br/>WHERE minimum_nights &lt; 28"]
    S --> G["CTE per_district:<br/>GROUP BY room_type, district<br/>HAVING COUNT(*) &gt;= 30"]
    G --> W["CTE ranked:<br/>RANK() OVER (PARTITION BY<br/>room_type ORDER BY median DESC)"]
    W --> F["outer query:<br/>WHERE rnk &lt;= 3"]
```

### Why it matters

Rankings within groups, running totals, changes against the previous period, moving averages and "this row compared with its group" are standard analytical questions. Without window functions they need self-joins or loops in Python; with them they are one readable query that the database executes over all rows.

### How it works in Python

The practice query ranks the districts by the median price of a short stay, separately for entire homes and private rooms (in [`workbooks/sql/03-rank-districts-by-price.sql`](../workbooks/sql/03-rank-districts-by-price.sql); here the top three):

```sql
WITH short_stays AS (                      -- step 1: the listings the question is about
    SELECT district, room_type, price
    FROM listings
    WHERE minimum_nights < 28 AND price IS NOT NULL
      AND room_type IN ('Entire home/apt', 'Private room')
), per_district AS (                       -- step 2: one row per room type and district
    SELECT room_type, district, COUNT(*) AS n_listings,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price
    FROM short_stays
    GROUP BY room_type, district
    HAVING COUNT(*) >= 30
), ranked AS (                             -- step 3: rank inside each room type
    SELECT *, RANK() OVER (PARTITION BY room_type ORDER BY median_price DESC) AS rnk
    FROM per_district
)
SELECT room_type, rnk, district, n_listings, median_price
FROM ranked
WHERE rnk <= 3                             -- filter one level up
ORDER BY room_type, rnk;
-- Entire home/apt | 1 | Mitte                    | 1116 | 212.5
-- Entire home/apt | 2 | Pankow                   |  816 | 197.525
-- Entire home/apt | 3 | Friedrichshain-Kreuzberg |  951 | 187.2
-- Private room    | 1 | Mitte                    |  367 | 117.0
-- Private room    | 2 | Charlottenburg-Wilm.     |  278 | 103.13
-- Private room    | 3 | Friedrichshain-Kreuzberg |  350 | 100.0
```

Mitte leads in both room types, but the rest of the order differs: Pankow is second for entire homes and only fourth for private rooms. A single ranking over all listings would have mixed two markets.

Running totals and month-over-month changes of the reviews, the course's proxy for demand, around the first COVID-19 lockdown:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW reviews_monthly AS SELECT * FROM 'case-study/data/airbnb/reviews_monthly.parquet'")
print(con.sql("""
    WITH monthly AS (
        SELECT month::date AS month, SUM(n_reviews) AS n
        FROM reviews_monthly
        WHERE month BETWEEN '2019-12-01' AND '2020-06-01'
        GROUP BY month
    )
    SELECT month, n,
           SUM(n) OVER (ORDER BY month)                         AS running_total,
           n - LAG(n) OVER (ORDER BY month)                     AS change,
           ROUND(AVG(n) OVER (ORDER BY month ROWS 2 PRECEDING)) AS moving_avg_3m
    FROM monthly
    ORDER BY month
""").df())
#        month       n  running_total  change  moving_avg_3m
# 0 2019-12-01  4070.0         4070.0     NaN         4070.0
# 1 2020-01-01  3971.0         8041.0   -99.0         4021.0
# 2 2020-02-01  4055.0        12096.0    84.0         4032.0
# 3 2020-03-01  2512.0        14608.0 -1543.0         3513.0
# 4 2020-04-01   285.0        14893.0 -2227.0         2284.0
# 5 2020-05-01   612.0        15505.0   327.0         1136.0
# 6 2020-06-01  1547.0        17052.0   935.0          815.0
```

`LAG` has no previous row for December, so the change is `NULL` (shown as `NaN` in pandas). The moving average uses fewer than three months at the start. In April 2020, Berlin's listings received 285 reviews instead of about 4,000 (Session 12 models this break).

![Reviews per year 2012-2025 with the year-over-year change computed by a window function](figures/reviews-per-year-window.png)

A window function can also place each listing next to its group without collapsing the rows. Here, the share of free nights in the coming year per listing, compared with the district average:

```python
import duckdb

con = duckdb.connect()
for table in ["listings", "calendar"]:
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM 'case-study/data/airbnb/{table}.parquet'")
print(con.sql("""
    WITH per_listing AS (                         -- one row per listing: share of free nights
        SELECT l.id, l.district, AVG(c.available::int) AS share_free
        FROM listings AS l JOIN calendar AS c ON c.listing_id = l.id
        GROUP BY l.id, l.district
    ), compared AS (                              -- window: each listing next to its district
        SELECT *, AVG(share_free) OVER (PARTITION BY district) AS district_avg,
               share_free - AVG(share_free) OVER (PARTITION BY district) AS diff_to_district
        FROM per_listing
    )
    SELECT district, ROUND(MIN(district_avg), 3) AS avg_share_free,
           ROUND(AVG((share_free = 0)::int), 3) AS share_fully_blocked,
           COUNT(*) FILTER (WHERE diff_to_district > 0.5) AS far_freer_than_district
    FROM compared
    GROUP BY district
    ORDER BY avg_share_free
    LIMIT 3
""").df().to_string(index=False))
#                 district  avg_share_free  share_fully_blocked  far_freer_than_district
#                 Neukölln           0.283                0.427                      203
# Friedrichshain-Kreuzberg           0.378                0.319                      441
#   Tempelhof - Schöneberg           0.384                0.246                      113
```

In Neukölln, 43 % of the listings have not a single free night in the coming year. That is not a sign of booming demand: a fully blocked calendar usually means the listing is not offered at all, which is why most of these listings show no price (Session 4).

The same window logic in pandas, for comparison: `groupby(...).rank(method="min", ascending=False)` is `RANK() OVER (PARTITION BY ...)`, `cumsum()` is a running `SUM`, `diff()` is `n - LAG(n)`, and `groupby(...).transform("mean")` is `AVG(...) OVER (PARTITION BY ...)`.

```python
import pandas as pd

listings = pd.read_parquet("case-study/data/airbnb/listings.parquet", columns=["district", "room_type", "price", "minimum_nights"])
short = listings[(listings["minimum_nights"] < 28) & listings["price"].notna()
                 & listings["room_type"].isin(["Entire home/apt", "Private room"])]
per_district = short.groupby(["room_type", "district"])["price"].agg(["size", "median"]).reset_index()
per_district = per_district[per_district["size"] >= 30]                                       # HAVING
per_district["rnk"] = per_district.groupby("room_type")["median"].rank(method="min", ascending=False)
print(per_district.query("rnk <= 2").sort_values(["room_type", "rnk"]).to_string(index=False))
#       room_type             district  size  median  rnk
# Entire home/apt                Mitte  1116 212.500  1.0
# Entire home/apt               Pankow   816 197.525  2.0
#    Private room                Mitte   367 117.000  1.0
#    Private room Charlottenburg-Wilm.   278 103.130  2.0

reviews = pd.read_parquet("case-study/data/airbnb/reviews_monthly.parquet")
yearly = reviews[reviews["month"] < "2026-01-01"].groupby(reviews["month"].dt.year)["n_reviews"].sum()
print(pd.DataFrame({"n": yearly, "change": yearly.diff(), "running_total": yearly.cumsum()}).loc[2019:2022])
#            n   change  running_total
# month
# 2019   53965  14640.0         156497
# 2020   26499 -27466.0         182996
# 2021   35447   8948.0         218443
# 2022   70471  35024.0         288914
```

### In practice

- **Tourism statistics** are often published as rankings, for example the regions with the most overnight stays per year; each such list is a ranking within a partition (year).
- **Finance teams** compute month-over-month growth and cumulative revenue with `LAG` and running sums for management reports.
- **Health-services research** with claims data uses `LAG` over the admissions of each patient to measure the time to readmission, the basis of readmission indicators such as those of the US Hospital Readmissions Reduction Program.

> [!WARNING]
> `RANK`, `DENSE_RANK` and `ROW_NUMBER` differ only when there are ties, so a bug may go unnoticed for a long time. Decide explicitly how ties should be handled, and add a second sort key (for example `ORDER BY median_price DESC, district`) when you need a reproducible order.

> [!CAUTION]
> `WHERE RANK() OVER (...) = 1` is an error: `WHERE` is evaluated before window functions. Compute the rank in a CTE and filter in the next step.

## Access from Python with SQLAlchemy and pandas

### Concept

**SQLAlchemy** is the standard Python toolkit for relational databases. `create_engine(url)` creates an **engine**, an object that opens and reuses connections. The URL names the **dialect** (which database) and the **driver** (which Python package talks to it), followed by user, password, host, port and database:

```
postgresql+psycopg://course:course@localhost:5432/course
└─dialect─┘ └driver┘ └user┘ └pw─┘ └─host──┘ └port┘└─db─┘
```

`pandas.read_sql(query, engine)` runs a query and returns a DataFrame; `DataFrame.to_sql(name, engine)` writes a DataFrame into a table. Values that change between runs are passed as **bound parameters**: in `text("... WHERE district = :district")` the placeholder `:district` is filled from `params={"district": "Neukölln"}`. The driver sends the values separately from the SQL text, so they can never be interpreted as SQL.

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

The database does the filtering, joining and aggregation on all rows; Python receives a small result for plotting and modelling. Parameters protect against **SQL injection**, an attack in which text entered by a user changes the meaning of a query. With string formatting, a "district" such as `x' OR '1'='1` would return every row; in a web form for district statistics, that is a data leak.

### How it works in Python

With PostgreSQL (after starting the server, see the next section):

```python
# requires a running PostgreSQL server and the environment variable DATABASE_URL
import os

import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])   # never hard-code passwords
query = text("""
    SELECT room_type, COUNT(*) AS n_listings,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price
    FROM listings
    WHERE district = :district AND minimum_nights < :max_nights AND price IS NOT NULL
    GROUP BY room_type ORDER BY n_listings DESC
""")
by_room = pd.read_sql(query, engine, params={"district": "Neukölln", "max_nights": 28})
print(by_room)   # Entire home/apt 350, 160.5 | Private room 199, 93.0 | Shared room 7, 47.0
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
    con.exec_driver_sql("CREATE TABLE listings AS SELECT * FROM 'case-study/data/airbnb/listings.parquet'")

by_room = pd.read_sql(text("""
    SELECT room_type, COUNT(*) AS n_listings,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY price) AS median_price
    FROM listings
    WHERE district = :district AND minimum_nights < :max_nights AND price IS NOT NULL
    GROUP BY room_type ORDER BY n_listings DESC
"""), engine, params={"district": "Neukölln", "max_nights": 28})
print(by_room)
#          room_type  n_listings  median_price
# 0  Entire home/apt         350         160.5
# 1     Private room         199          93.0
# 2      Shared room           7          47.0

with engine.connect() as con:    # a single value, without pandas
    n = con.execute(text("SELECT COUNT(*) FROM listings WHERE district = :d AND license_status = :s"),
                    {"d": "Neukölln", "s": "registration number"}).scalar_one()
print(n)                                                             # 541
```

### In practice

- **Analysts** in most companies pull aggregated data from a data warehouse into notebooks with exactly this pattern; the heavy work stays in the database.
- **Web applications** written in Python, such as the service of Session 16, use SQLAlchemy to read and write their database.
- **SQL injection** is part of the "Injection" category of the OWASP Top 10 web application security risks; the 2008 breach of the card processor Heartland Payment Systems, one of the largest of its time, started with an SQL injection.

> [!CAUTION]
> Never build SQL with f-strings from values that come from users, files or the web: `f"... WHERE district = '{district}'"`. Use bound parameters. Table and column names cannot be parameters; if they must vary, check them against a fixed list.

> [!TIP]
> `pd.read_sql` loads the whole result into memory. For large results, aggregate in SQL first, or read in pieces with `chunksize=` and process each piece. `SELECT * FROM calendar` sends 4.7 million rows to Python; the share of free nights per district is twelve rows.

## Loading data reproducibly with an ingestion script and constraints

### Concept

**Ingestion** is the step that brings data from their source into your database. It is **reproducible** if running the same script on the same source always gives the same tables, without manual steps in between. A good ingestion script:

1. downloads or reads the raw data from a documented source;
2. applies the same transformations every time (in the case study: converting the price text to a number, parsing dates, masking names in the registration field, aggregating reviews per month);
3. creates the tables with **keys and constraints**, so that invalid rows are rejected;
4. loads parent tables before child tables (listings before calendar and reviews), because the foreign key requires the parent row to exist;
5. checks the result: row counts against the source, and no rejected rows, or rejected rows that are counted and documented.

The case-study script `case-study/prepare_airbnb.py` does steps 1 and 2 and can write the tables `listings`, `calendar`, `reviews_monthly` and `weather_daily` to PostgreSQL with `--postgres`. Because `pandas.to_sql` creates tables without keys, the keys are added afterwards with [`workbooks/sql/02-add-constraints.sql`](../workbooks/sql/02-add-constraints.sql); alternatively, the case-study notebook creates the schema first with [`01-schema.sql`](../workbooks/sql/01-schema.sql) and then loads the rows.

```mermaid
stateDiagram-v2
    [*] --> Downloaded: download raw files
    Downloaded --> Built: build tables (parse, mask, aggregate)
    Built --> Loaded: load listings, then calendar and reviews
    Loaded --> Constrained: add keys and checks
    Constrained --> Verified: counts match, rejected rows documented
    Loaded --> Failed: constraint violated
    Failed --> Built: fix the data or the rule
    Verified --> [*]
```

### Why it matters

Manual loading ("I imported the CSV in a graphical tool and fixed a few rows") cannot be repeated or reviewed, and nobody knows later which version of the data an analysis used. A script can be rerun when Inside Airbnb publishes the next quarterly snapshot, reviewed in a pull request and run by every team member. Constraints turn silent data problems into loud errors at load time.

### How it works in Python

Start PostgreSQL in Docker (one command; the container keeps running in the background):

```bash
docker run --name course-db -e POSTGRES_USER=course -e POSTGRES_PASSWORD=course \
    -e POSTGRES_DB=course -p 5432:5432 -d postgres:17
export DATABASE_URL=postgresql+psycopg://course:course@localhost:5432/course

uv run python case-study/prepare_airbnb.py --postgres $DATABASE_URL   # about 1.5 minutes for the 4.7 M calendar rows
psql postgresql://course:course@localhost:5432/course -f sessions/03-sql-and-polars/workbooks/sql/02-add-constraints.sql
```

> [!NOTE]
> `psql` expects a plain `postgresql://` URL; SQLAlchemy needs the `+psycopg` part to choose the driver. Without `psql` on your machine, run the file from Python as workbook 10 does, or inside the container: `docker exec -i course-db psql -U course course < sessions/03-sql-and-polars/workbooks/sql/02-add-constraints.sql`.

The core of a schema-first loader, here with DuckDB so that it runs anywhere. Two constraints of [`01-schema.sql`](../workbooks/sql/01-schema.sql) reject rows of the snapshot, and each error message names the rule:

```python
import duckdb

con = duckdb.connect()
con.execute(open("sessions/03-sql-and-polars/workbooks/sql/01-schema.sql").read())   # keys and constraints
COLS = ("id, host_id, district, neighbourhood, latitude, longitude, room_type, accommodates, bedrooms, price, "
        "minimum_nights, maximum_nights, availability_365, number_of_reviews, first_review, last_review, "
        "review_scores_rating, license_status")
SRC = "case-study/data/airbnb"
try:
    con.execute(f"INSERT INTO listings SELECT {COLS} FROM '{SRC}/listings.parquet'")
except duckdb.ConstraintException as e:
    print(str(e).splitlines()[0][:100])
# Constraint Error: CHECK constraint failed on table listings with expression CHECK((maximum_nights <

# decision 1, documented in the data card: the placeholder maximum stay becomes NULL
con.execute(f"INSERT INTO listings SELECT {COLS.replace('maximum_nights', 'NULLIF(maximum_nights, 2147483647)')} "
            f"FROM '{SRC}/listings.parquet'")
try:
    con.execute(f"INSERT INTO calendar SELECT listing_id, date, available, minimum_nights FROM '{SRC}/calendar.parquet'")
except duckdb.ConstraintException as e:
    print(str(e).splitlines()[0][:100])
# Constraint Error: Violates foreign key constraint because key "id: 1502751432634721106" does not exi
print(con.sql("SELECT COUNT(*) FROM calendar").fetchone())   # (0,): the whole statement was rolled back

# decision 2: load only calendar rows of known listings, and count what was left out
con.execute(f"""INSERT INTO calendar SELECT listing_id, date, available, minimum_nights
                FROM '{SRC}/calendar.parquet' WHERE listing_id IN (SELECT id FROM listings)""")
print(con.sql(f"""SELECT (SELECT COUNT(*) FROM listings), (SELECT COUNT(*) FROM calendar),
                         (SELECT COUNT(*) FROM '{SRC}/calendar.parquet') - (SELECT COUNT(*) FROM calendar)""").fetchone())
# (12776, 4663240, 28835): 28,835 calendar rows of 79 vanished listings left out, and we know exactly which
```

### In practice

- **GitLab** published a detailed post-mortem of a 2017 incident in which production PostgreSQL data were deleted by mistake and several backup procedures turned out not to work; restoring depended on a copy that happened to exist. Scripted, tested loading and restoring is the lesson.
- **Public Health England** lost about 16,000 positive COVID-19 test results in October 2020 because a manual step used an old Excel file format whose row limit was exceeded; an automated pipeline with row-count checks would have caught it.
- **Data engineering teams** use tools such as dbt, which runs SQL transformations from version-controlled files and checks `unique`, `not_null` and `relationships` rules after each run.

> [!WARNING]
> `to_sql(..., if_exists="replace")` drops the table and recreates it **without** keys, constraints or indexes. After a reload, run the constraints script again, or load with `if_exists="append"` into a table that you created with its schema. Once `02-add-constraints.sql` has added the foreign keys, `listings` cannot simply be dropped: a second run of `prepare_airbnb.py --postgres` stops with "cannot drop table listings because other objects depend on it". Drop the tables first (`DROP TABLE calendar, reviews_monthly, weather_daily, listings CASCADE;`), then reload and add the constraints again.

> [!TIP]
> Sending 4.7 million rows with plain `to_sql` takes about a minute and a half. PostgreSQL's `COPY` command is much faster: in the course team's test, workbook 10 loaded all four tables with `COPY` in about 35 seconds, most of it spent preparing the rows in pandas. The workbook shows how to use it from pandas, following the example in the pandas documentation (`to_sql(..., method=)` with a callable).

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
      missing values
      personal data
    Processing
      script, filters
      rejected rows
    Uses
      intended
      to avoid
    Limitations
```

Some facts can be computed (row counts, date ranges, shares of missing values); others need judgement (who is under-represented, which questions the data cannot answer). For the Berlin listings, three limitations need judgement:

- **Survivorship.** `reviews_monthly` contains the reviews of listings that are still online in June 2026. Listings that were deleted earlier took their reviews with them, so the growth of reviews since 2015 overstates the growth of the market.
- **Availability is not demand.** A night that is not available is booked *or* blocked by the host; the calendar cannot tell them apart.
- **Asking prices.** The price is what a host asks for the next free night, not what guests paid, and Airbnb shifts each location by up to about 150 metres to protect the host.

### Why it matters

Without documentation, knowledge about a dataset lives in the heads of the people who prepared it. Later users repeat mistakes (such as using a column that is derived from the target, see Session 9), use data outside their licence, or draw conclusions about populations the data do not cover. A data card also forces the team to look at the data before modelling.

### How it works in Python

The computed part of the card comes from a few queries:

```python
import duckdb

con = duckdb.connect()
for table in ["listings", "calendar", "reviews_monthly"]:
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM 'case-study/data/airbnb/{table}.parquet'")
facts = con.sql("""
    SELECT COUNT(*) AS n_listings,
           COUNT(DISTINCT host_id) AS n_hosts,
           COUNT(DISTINCT district) AS n_districts,
           MIN(last_scraped)::date AS scraped_from, MAX(last_scraped)::date AS scraped_to,
           ROUND(AVG((price IS NULL)::int), 3) AS share_price_missing,
           ROUND(AVG((minimum_nights >= 28)::int), 3) AS share_medium_term,
           ROUND(AVG((license_status = 'registration number')::int), 3) AS share_registered,
           (SELECT MIN(month)::date FROM reviews_monthly) AS first_review_month,
           (SELECT MAX(date)::date FROM calendar) AS calendar_until
    FROM listings
""").df().T
print(facts)
#                                        0
# n_listings                         12776
# n_hosts                             8182
# n_districts                           12
# scraped_from         2026-06-26 00:00:00
# scraped_to           2026-07-03 00:00:00
# share_price_missing                0.339
# share_medium_term                  0.351
# share_registered                   0.351
# first_review_month   2009-06-01 00:00:00
# calendar_until       2027-07-02 00:00:00
```

The two equal shares are a coincidence (4,480 listings each), and a revealing one: the groups hardly overlap, since only 131 medium-term listings show a registration number. Such observations belong in the card.

### In practice

- **Hugging Face** shows a dataset card on the page of every dataset on its Hub; the card of a dataset is the first thing users see.
- **Google** introduced *Data Cards* for its own datasets and published the template and the experience from using it (Pushkarna et al., 2022).
- **Public open-data portals**, for example GovData in Germany, publish metadata following the DCAT-AP standard: source, licence, update frequency and contact for every dataset.

> [!IMPORTANT]
> Inside Airbnb publishes its data under the Creative Commons Attribution 4.0 licence (CC BY 4.0): reuse is allowed with attribution. The data are scraped from public listing pages and contain personal data, which `prepare_airbnb.py` removes (host names, review texts, names typed into the registration field). A data card records both the licence and these steps, and the course reports results in aggregate, never naming hosts; "found on the internet" is not a licence.

## Check your understanding

1. Rewrite "the three cheapest districts for private rooms" as CTEs and an outer query. In which step does the window function appear, and in which step the filter?
2. When do `RANK()` and `ROW_NUMBER()` give different results?
3. Why is `text("... WHERE district = :district")` with `params={"district": d}` safer than an f-string?
4. The script writes the tables with `to_sql(if_exists="replace")`. What is missing afterwards, and why can the foreign key from `calendar` to `listings` not simply be added as a valid constraint?
5. Name two facts of the listings' data card that can be computed by a query and two that need human judgement.

## Further reading

- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92. https://arxiv.org/abs/1803.09010
- The PostgreSQL Global Development Group. *PostgreSQL documentation: Window functions* (tutorial section 3.5). https://www.postgresql.org/docs/current/tutorial-window.html
- SQLAlchemy. *SQLAlchemy Unified Tutorial* (version 2.0). https://docs.sqlalchemy.org/en/20/tutorial/
- Inside Airbnb. *Data assumptions*. https://insideairbnb.com/data-assumptions/
