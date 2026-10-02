# The relational model and basic SQL

This page covers the first block of the session. Suppose a housing analyst of the city asks three questions: how many Airbnb listings are there in each district, what does a night cost there, and which listings are actually booked? The answers are spread over several tables: one row per listing, one row per listing and night of the coming year, one row per listing and month with the number of reviews. A relational database keeps such tables linked by keys, and SQL is the language analysts use to get answers out of them. Almost every organisation keeps its operational data in relational databases, and in job advertisements for data roles SQL is among the most frequently requested skills after Python.

All examples use the course case study, the Berlin snapshot of Inside Airbnb (26 June 2026): `listings` (12,776 listings with district, room type, price, minimum stay, reviews and registration status), `calendar` (availability of each listing for each of the next 365 nights, 4.7 million rows) and `reviews_monthly` (number of reviews per listing and month since 2009).

```mermaid
flowchart LR
    Q["Business question"] --> S["SQL query"]
    S --> DB[("PostgreSQL<br/>listings, calendar,<br/>reviews_monthly")]
    DB --> R["Small result table"]
    R --> P["pandas / Polars<br/>in Python"]
    P --> A["Chart, model, report"]
```

> [!NOTE]
> You can run every query on this page without a database server: DuckDB, an embedded database, reads the Parquet files of the case study directly (see [How it works in Python](#how-it-works-in-python)). PostgreSQL itself is set up in block 2.

## The relational model: tables, primary and foreign keys, relationships

### Concept

A **relational database** stores data in **tables** (also called relations). Each **row** is one record; each **column** has a name and a **data type** (text, integer, decimal number, date, true/false). The set of table definitions is the **schema**. The idea goes back to Edgar F. Codd (1970), who proposed that data should be described by tables and queried by their content, not by the way they are stored on disk.

A **primary key** is a column, or a combination of columns, whose value identifies each row uniquely and is never empty. In the case study, `id` identifies a listing. The calendar has a **composite key**: the pair (`listing_id`, `date`) identifies a row, because each listing appears once per night.

A **foreign key** is a column that refers to the primary key of another table. `calendar.listing_id` and `reviews_monthly.listing_id` point to `listings.id`. This expresses a **relationship**: one listing has many calendar nights and many review months, each of which belongs to exactly one listing (a *one-to-many* relationship). The database can enforce it: a calendar row for a listing that does not exist is rejected.

Storing each fact once, in one table, is called **normalisation**. The district of a listing is stored once in `listings`, not repeated in each of its 365 calendar rows. If a listing is assigned to another district, one row changes, and no copy can contradict another.

A small example that you can follow by hand:

| listings |  |  |  |
|---|---|---|---|
| **id** (PK) | district | room_type | price |
| 1 | Mitte | Entire home/apt | 120 |
| 2 | Neukölln | Private room | 45 |
| 3 | Mitte | Entire home/apt | NULL |

| reviews_monthly |  |  |
|---|---|---|
| **listing_id** (PK, FK) | **month** (PK) | n_reviews |
| 1 | 2026-04 | 3 |
| 1 | 2026-05 | 2 |
| 2 | 2026-05 | 1 |

Listing 1 has two months with reviews, listing 2 one, listing 3 none. `NULL` marks a value that is unknown: listing 3 shows no price. (The ids here are invented; the real ones have up to 19 digits.)

```mermaid
classDiagram
    direction LR
    class listings {
        id : bigint  PK
        host_id : bigint
        district : text
        room_type : text
        price : double
        minimum_nights : integer
        license_status : text
    }
    class calendar {
        listing_id : bigint  PK, FK
        date : date  PK
        available : boolean
        minimum_nights : integer
    }
    class reviews_monthly {
        listing_id : bigint  PK, FK
        month : date  PK
        n_reviews : integer
    }
    listings "1" --> "0..*" calendar : has nights
    listings "1" --> "0..*" reviews_monthly : has reviews
```

### Why it matters

Keys make the relationship between tables explicit, and the database uses them to guard data quality at the moment data are written: a calendar row for an unknown listing, or the same night of a listing twice, is rejected immediately instead of being discovered months later in an analysis. The database also builds an **index** (a sorted lookup structure) on every primary key, which makes lookups and joins fast. Without normalisation, the same fact is stored many times, and copies drift apart.

### How it works in Python

The case-study tables arrive as Parquet files. The following check tests the key properties with pandas: the primary keys are unique, and every foreign-key value has a partner.

```python
import pandas as pd

listings = pd.read_parquet("case-study/data/airbnb/listings.parquet", columns=["id", "district"])
calendar = pd.read_parquet("case-study/data/airbnb/calendar.parquet")
reviews = pd.read_parquet("case-study/data/airbnb/reviews_monthly.parquet")

print(listings["id"].is_unique, calendar.duplicated(["listing_id", "date"]).any())   # True False
# foreign keys: does every calendar and review row refer to an existing listing?
for name, table in [("calendar", calendar), ("reviews_monthly", reviews)]:
    unknown = ~table["listing_id"].isin(listings["id"])
    print(name, unknown.sum(), table.loc[unknown, "listing_id"].nunique())
# calendar 28835 79
# reviews_monthly 499 57
# one-to-many: review months per listing
print(reviews.groupby("listing_id").size().describe()[["mean", "50%", "max"]].round(1).to_dict())
# {'mean': 22.1, '50%': 11.0, 'max': 163.0}
```

The foreign key fails for 79 listings in the calendar (28,835 rows, exactly 365 per listing) and 57 listings in the review table. Inside Airbnb scrapes listings, calendars and reviews separately, and the listings were scraped between 26 June and 3 July 2026: a listing that went offline between two scrapes appears in one file but not in the other. A real foreign key would have rejected these rows, which is exactly the point: the database makes you decide what to do with them (keep them in a separate table, or drop them and document it) instead of letting a join lose them silently.

The same rules written as SQL create the tables with **constraints**. `CHECK` adds a rule for the values; `REFERENCES` declares the foreign key.

```sql
CREATE TABLE demo_listings (
    id        BIGINT PRIMARY KEY,                                              -- one row per listing
    district  TEXT NOT NULL,
    room_type TEXT NOT NULL CHECK (room_type IN ('Entire home/apt', 'Private room', 'Hotel room', 'Shared room')),
    price     DOUBLE PRECISION CHECK (price > 0)                               -- NULL allowed
);
CREATE TABLE demo_reviews (
    listing_id BIGINT NOT NULL REFERENCES demo_listings (id),                  -- foreign key
    month      DATE NOT NULL,
    n_reviews  INTEGER NOT NULL CHECK (n_reviews > 0),
    PRIMARY KEY (listing_id, month)                                            -- composite key
);
INSERT INTO demo_listings VALUES (1, 'Mitte', 'Entire home/apt', 120), (2, 'Neukölln', 'Private room', 45),
                                 (3, 'Mitte', 'Entire home/apt', NULL);
INSERT INTO demo_reviews VALUES (1, '2026-04-01', 3), (1, '2026-05-01', 2), (2, '2026-05-01', 1);
INSERT INTO demo_reviews VALUES (9, '2026-05-01', 1);           -- ERROR: violates foreign key constraint
INSERT INTO demo_reviews VALUES (1, '2026-05-01', 4);           -- ERROR: duplicate key (listing 1, May)
INSERT INTO demo_listings VALUES (4, 'Mitte', 'Castle', 300);   -- ERROR: violates check constraint
```

### In practice

- **Stack Exchange Data Explorer** lets anyone query the public Stack Overflow database with SQL; questions, answers, users and votes are separate tables linked by identifiers such as `PostId` and `UserId`.
- **Wikimedia** offers public read-only copies of the Wikipedia databases through the Quarry service; pages, revisions and users are related tables, and volunteers answer research questions about Wikipedia with SQL.
- **Inside Airbnb** publishes each city as separate files (listings, calendar, reviews) that are linked only by the listing id; every analysis that combines them, including the course case study, relies on that key.

> [!WARNING]
> A file is not a database. A Parquet or CSV file has no keys and no constraints: nothing stops a duplicated `id` or a calendar row for a listing that is not in the listings file. When you work with files, check the key properties yourself (as in the Python block above) and repeat the check after each update.

## PostgreSQL as the course database

### Concept

A **database management system** (DBMS) is the software that stores tables, checks constraints, executes queries and lets many users work at the same time without corrupting the data. **PostgreSQL** is a free, open-source relational DBMS that has been developed since 1986 (as POSTGRES at the University of California, Berkeley). It follows the SQL standard closely, supports window functions, JSON columns, full-text search and, through the `pgvector` extension, vector search, which the course uses again in Session 15.

PostgreSQL runs as a **server**: a separate program that listens on a network port (5432 by default). Clients, such as the command-line tool `psql`, a graphical tool such as DBeaver, or Python with SQLAlchemy, connect to it with a host, port, database name, user and password.

**DuckDB** is a different kind of database: an **embedded** database that runs inside your Python process, like SQLite, but optimised for analytical queries over many rows. It needs no server and reads Parquet and CSV files directly. The course uses DuckDB for quick queries over files and as a fallback when PostgreSQL is not available.

| | PostgreSQL | DuckDB | SQLite |
|---|---|---|---|
| Runs as | server (shared) | inside your process | inside your process |
| Typical use | operational data, team database | analytics on files, notebooks | apps, mobile phones |
| Many writers at once | yes | no | limited |
| Reads Parquet directly | no (extension needed) | yes | no |

### Why it matters

A team project needs one place where the current version of the data lives, with constraints that every member's code must respect. A server database provides that. Learning PostgreSQL also transfers directly to work: cloud services such as Amazon RDS, Google Cloud SQL and Azure Database offer managed PostgreSQL, and data warehouses such as Amazon Redshift use a PostgreSQL-like SQL dialect.

### How it works in Python

Start a PostgreSQL server in Docker with one command (details in [block 2](02-sql-from-python-and-ingestion.md#loading-data-reproducibly-with-an-ingestion-script-and-constraints)), then connect from Python. Without a server, DuckDB answers the same SQL from the files:

```python
import duckdb

con = duckdb.connect()   # in-memory database, no server
for table in ["listings", "calendar", "reviews_monthly"]:
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM 'case-study/data/airbnb/{table}.parquet'")
print(con.sql("SELECT COUNT(*) FROM listings").fetchone(), con.sql("SELECT COUNT(*) FROM calendar").fetchone())
# (12776,) (4692075,)

# with PostgreSQL (requires a running server, see block 2):
# from sqlalchemy import create_engine
# engine = create_engine("postgresql+psycopg://course:course@localhost:5432/course")
```

### In practice

- In the **Stack Overflow Developer Survey** of 2023 and 2024, PostgreSQL was the database used most often by professional developers.
- **Instagram** documented in its engineering blog how it stored and sharded its data across many PostgreSQL servers as the service grew.
- **DuckDB** was developed at the Centrum Wiskunde & Informatica (CWI) in Amsterdam, the Dutch national research institute for mathematics and computer science, for analytical work inside data-science tools.

> [!TIP]
> Never write passwords into notebooks or scripts that you share or commit. Put the connection URL into an environment variable (`DATABASE_URL`) and read it with `os.environ["DATABASE_URL"]`.

## Basic queries: SELECT, WHERE, ORDER BY, LIMIT

### Concept

A query describes the result you want, not the steps to compute it; the database chooses how. The basic clauses are:

- `SELECT` names the columns (or expressions) of the result;
- `FROM` names the table;
- `WHERE` gives a condition that each row must satisfy;
- `ORDER BY` sorts the result (`DESC` for descending);
- `LIMIT` keeps only the first *n* rows.

Conditions combine with `AND`, `OR` and `NOT`. `IN (...)` tests membership in a list, `BETWEEN a AND b` a range, and `LIKE` / `ILIKE` a text pattern, where `%` stands for any sequence of characters (`ILIKE` ignores upper and lower case).

Although we write `SELECT` first, the database evaluates the clauses in a different **logical order**. This explains, for example, why a column alias defined in `SELECT` cannot be used in `WHERE`.

```mermaid
flowchart LR
    F["1 FROM / JOIN"] --> W["2 WHERE"] --> G["3 GROUP BY"] --> H["4 HAVING"] --> S["5 SELECT"] --> O["6 ORDER BY"] --> L["7 LIMIT"]
```

By hand: in the small tables above, `SELECT id FROM listings WHERE district = 'Mitte' ORDER BY id` keeps listings 1 and 3 and returns them in this order. `WHERE price < 100` keeps only listing 2: listing 3 has no price, and a comparison with `NULL` is never true.

### Why it matters

Filtering in the database moves only the rows you need into Python. With millions of calendar rows, `SELECT *` followed by filtering in pandas wastes memory and time. Writing the condition explicitly also makes the question precise: "affordable short stays for a family" becomes `room_type = 'Entire home/apt' AND minimum_nights < 28 AND accommodates >= 4 AND price <= 150`.

### How it works in Python

```sql
SELECT id, neighbourhood, price, number_of_reviews, review_scores_rating
FROM listings
WHERE district = 'Neukölln' AND room_type = 'Entire home/apt'
  AND minimum_nights < 28 AND price < 100 AND number_of_reviews >= 50    -- all conditions must hold
ORDER BY review_scores_rating DESC, number_of_reviews DESC             -- best rated first; reviews break ties
LIMIT 3;
-- 31382644 | Schillerpromenade | 92.5 | 374 | 4.91
--   237670 | Buckow Nord       | 68.5 |  91 | 4.84
--   190448 | Buckow Nord       | 85.0 |  87 | 4.77
```

Listing titles are not shown on purpose: hosts sometimes write their own name into the title, and an analysis does not need it.

From Python with DuckDB, a query result becomes a pandas DataFrame with `.df()`:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW listings AS SELECT * FROM 'case-study/data/airbnb/listings.parquet'")

print(con.sql("""
    SELECT COUNT(*) FROM listings
    WHERE room_type = 'Entire home/apt' AND minimum_nights < 28 AND price BETWEEN 50 AND 100
""").fetchone())                                                                              # (336,)
print(con.sql("SELECT COUNT(*) FROM listings WHERE name ILIKE '%balcony%' OR name ILIKE '%balkon%'").fetchone())  # (711,)
longest = con.sql("""
    SELECT id, district, room_type, minimum_nights
    FROM listings ORDER BY minimum_nights DESC NULLS LAST, id LIMIT 3
""").df()
print(longest)
#         id district        room_type  minimum_nights
# 0  6670861  Neukölln  Entire home/apt          1125.0
# 1   584757    Pankow     Private room          1000.0
# 2  6704144    Pankow     Private room          1000.0
```

`NULLS LAST` matters here: three listings have no minimum stay, and the databases differ in where they sort `NULL` by default (PostgreSQL puts it first in descending order). A minimum stay of 1,125 nights, three years, is not a holiday rental; Session 4 treats such values.

### In practice

- **Property portals** turn the search form (district, number of rooms, maximum rent) into exactly such filter conditions before they sort the result by date or price.
- The **European Medicines Agency's** EudraVigilance database of suspected side effects is queried by drug, period and outcome before any statistical signal detection is done.
- **Data engineers** check every nightly load with small queries such as `SELECT COUNT(*) FROM orders WHERE order_date = CURRENT_DATE - 1`.

> [!CAUTION]
> Without `ORDER BY`, the order of rows is not defined. `LIMIT 10` without `ORDER BY` returns *some* ten rows, which may change between runs or database versions.

## Aggregation with GROUP BY and HAVING

### Concept

An **aggregate function** reduces many values to one: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, and for the median `percentile_cont(0.5) WITHIN GROUP (ORDER BY price)` (PostgreSQL and DuckDB; DuckDB also accepts `MEDIAN(price)`). `GROUP BY` splits the rows into groups with equal values of the grouping columns and computes the aggregates per group; the result has one row per group. Every column in `SELECT` must either be a grouping column or be inside an aggregate.

`WHERE` filters **rows before** grouping; `HAVING` filters **groups after** aggregation, for example "only districts with at least 300 listings". `FILTER (WHERE ...)` restricts a single aggregate to some rows.

A **share** is the average of a 0/1 variable. `AVG((license_status = 'registration number')::int)` is the share of listings with a registration number, because `::int` turns true/false into 1/0.

By hand: grouping the three example listings by `district` gives Mitte with `COUNT(*) = 2` and `COUNT(price) = 1`, and Neukölln with 1 and 1. `AVG(price)` for Mitte is 120, not 60: the missing price is ignored, not counted as zero. `HAVING COUNT(*) >= 2` keeps only Mitte.

### Why it matters

Almost every business question is an aggregate: listings per district, median price per room type, reviews per month, the share of registered listings per host type. `HAVING` with a minimum count prevents tiny groups from topping a ranking by chance: a district with five listings can have any median.

### How it works in Python

```sql
SELECT district,
       COUNT(*) AS n_listings,
       COUNT(price) FILTER (WHERE minimum_nights < 28) AS n_short_stays,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY price) FILTER (WHERE minimum_nights < 28) AS median_price
FROM listings
GROUP BY district
HAVING COUNT(*) >= 300            -- filter groups, not rows
ORDER BY median_price DESC
LIMIT 4;
-- Mitte                    | 2826 | 1537 | 187.0
-- Pankow                   | 1950 | 1033 | 174.0
-- Friedrichshain-Kreuzberg | 2652 | 1337 | 160.0
-- Charlottenburg-Wilm.     | 1432 |  760 | 149.3
```

The median price is computed for short stays only (`minimum_nights < 28`). A third of the listings require a month or more, and their price field means something else (Session 4); mixing them in would compare different products.

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW listings AS SELECT * FROM 'case-study/data/airbnb/listings.parquet'")
by_type = con.sql("""
    SELECT room_type,
           COUNT(*) AS n,
           ROUND(AVG((license_status = 'registration number')::int), 3) AS share_registered,
           COUNT(*) FILTER (WHERE minimum_nights >= 28) AS n_medium_term
    FROM listings
    GROUP BY room_type
    ORDER BY n DESC
""").df()
print(by_type)
#          room_type     n  share_registered  n_medium_term
# 0  Entire home/apt  8846             0.336           3157
# 1     Private room  3754             0.401           1310
# 2       Hotel room    89             0.000              8
# 3      Shared room    87             0.034              5
```

Hotel rooms and shared rooms (hostel beds) almost never show a registration number: the Berlin registration rule concerns private flats, while hotels and hostels are commercial businesses regulated differently, so a missing number is not a violation there. One aggregate table, read with knowledge of the domain, prevents a wrong headline.

### In practice

- **Eurostat and national statistical offices** publish tourism statistics, such as nights spent per region and month, computed from the reports of individual accommodation businesses; each published cell is a `GROUP BY` result.
- **Inside Airbnb** shows the number of listings per neighbourhood and the share of each room type on its city pages; they are aggregates of the same listings table.
- **Hospital comparisons** such as Care Compare of the US Centers for Medicare & Medicaid Services do not publish a measure when a hospital has too few cases, which is the idea of `HAVING COUNT(*) >= k`.

> [!WARNING]
> `COUNT(*)` counts rows; `COUNT(column)` counts non-NULL values of that column; `AVG(column)` silently ignores NULLs. On the listings table, `COUNT(*)` is 12,776 but `COUNT(price)` only 8,441, and `AVG(price)` (€160.7) is the average over the listings that show a price, not over all listings.

## INNER and LEFT JOIN; NULL values in joins and aggregates

### Concept

A **join** combines rows of two tables whose key columns match. An **inner join** (`JOIN ... ON a.key = b.key`) returns one row per matching pair; rows without a partner disappear. Joining `listings` with `reviews_monthly` returns one row per listing and review month, with the district and room type of the listing added. `USING (key)` is a short form when the key has the same name in both tables. Short **aliases** (`l`, `r`) say which table a column comes from.

A **left join** keeps every row of the left table; where no partner exists, the columns of the right table are `NULL`. Combined with `WHERE right.key IS NULL`, it finds rows **without** a partner, an **anti-join**: for example, listings that have never been reviewed.

![Inner and left join of a small listings and monthly reviews table](figures/join-types.png)

**NULL** means "unknown". Any comparison with NULL is unknown, not true or false, so `price = NULL` never matches; use `price IS NULL`. In `GROUP BY`, all NULLs form one group.

By hand, with the tables above: `listings JOIN reviews_monthly` gives 3 rows (listing 1 twice, listing 2 once); `listings LEFT JOIN reviews_monthly` gives 4 rows, the fourth being listing 3 with `month = NULL` and `n_reviews = NULL`.

### Why it matters

Normalised data must be joined before they can be analysed. The join type decides which rows survive and therefore which population a number describes. Many wrong numbers in reports come from an unintended inner join, which drops unmatched rows silently, or from a join on a key that is not unique, which duplicates rows.

### How it works in Python

```sql
-- reviews in 2025 per district: each review month gets the district of its listing
SELECT l.district, SUM(r.n_reviews) AS reviews_2025, COUNT(DISTINCT l.id) AS listings_reviewed
FROM listings AS l
JOIN reviews_monthly AS r ON r.listing_id = l.id
WHERE r.month BETWEEN '2025-01-01' AND '2025-12-01'
GROUP BY l.district
ORDER BY reviews_2025 DESC
LIMIT 3;
-- Mitte                    | 32541 | 1387
-- Friedrichshain-Kreuzberg | 31179 | 1403
-- Pankow                   | 21587 | 1035

-- share of nights still free in July and August 2026, per district (listings joined with calendar)
SELECT l.district, COUNT(*) AS nights, ROUND(AVG(c.available::int), 3) AS share_available
FROM listings AS l
JOIN calendar AS c ON c.listing_id = l.id
WHERE c.date BETWEEN '2026-07-01' AND '2026-08-31'
GROUP BY l.district
ORDER BY share_available
LIMIT 3;
-- Neukölln                 |  79532 | 0.257
-- Pankow                   | 119050 | 0.320
-- Friedrichshain-Kreuzberg | 161730 | 0.321

-- anti-join: listings without any review
SELECT COUNT(*) AS listings_without_review
FROM listings AS l
LEFT JOIN reviews_monthly AS r ON r.listing_id = l.id
WHERE r.listing_id IS NULL;
-- 2573
```

A night that is not available is either booked or blocked by the host; the calendar does not say which. "Only a quarter of the summer nights in Neukölln are still free" is therefore an upper bound on demand, not a measure of it.

```python
import duckdb

con = duckdb.connect()
for table in ["listings", "reviews_monthly"]:
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM 'case-study/data/airbnb/{table}.parquet'")

# who are the listings without any review?
print(con.sql("""
    SELECT l.room_type, COUNT(*) AS n,
           ROUND(AVG((l.minimum_nights >= 28)::int), 3) AS share_medium_term
    FROM listings AS l
    LEFT JOIN reviews_monthly AS r ON r.listing_id = l.id
    WHERE r.listing_id IS NULL
    GROUP BY l.room_type
    ORDER BY n DESC
    LIMIT 2
""").df())
#          room_type     n  share_medium_term
# 0  Entire home/apt  1912              0.823
# 1     Private room   648              0.664

# the other direction: review rows whose listing is not in the listings table
print(con.sql("""
    SELECT COUNT(*) AS n_rows, COUNT(DISTINCT r.listing_id) AS n_listings, SUM(r.n_reviews) AS n_reviews
    FROM reviews_monthly AS r
    LEFT JOIN listings AS l ON l.id = r.listing_id
    WHERE l.id IS NULL
""").fetchall())                                                           # [(499, 57, 4932)]
print(con.sql("SELECT COUNT(*) FROM reviews_monthly JOIN listings ON listings.id = reviews_monthly.listing_id").fetchone())
# (226058,): the inner join silently drops 499 of the 226,557 review months
```

Most listings without a review are medium-term rentals (78 %, against 24 % among the reviewed listings): people who rent a flat for three months rarely leave a review, or the listing is new. The inner join loses the 4,932 reviews of the 57 vanished listings without any warning; only the count against the 226,557 rows of the table shows it.

### In practice

- **Marketing analysts** join transactions with customer attributes to compare customer segments; a customer without transactions disappears in an inner join and must be kept with a left join if "inactive customers" are part of the question.
- **Epidemiological record linkage**, for example linking cancer registries with mortality registers, keeps unmatched records and analyses them separately, because who fails to link is itself informative.
- **Data quality checks** in finance and retail use anti-joins to find records that refer to a deleted customer or a discontinued product, as the review rows of vanished listings do here.

> [!CAUTION]
> A join with a one-to-many relationship repeats the rows of the "one" side. After `listings JOIN reviews_monthly`, a listing with 100 review months appears 100 times, so `AVG(l.price)` is no longer the average price of a listing but an average weighted by review months: €182.2 instead of €190.2 for the reviewed short stays. Aggregate the "many" side first (in a CTE, block 2), or check `COUNT(*)` against `COUNT(DISTINCT key)`; in pandas, use `merge(..., validate="many_to_one")`.

> [!WARNING]
> A condition on the right table in `WHERE` turns a left join back into an inner join: `listings l LEFT JOIN reviews_monthly r ... WHERE r.month >= '2026-01-01'` keeps only the 6,009 listings with a review in 2026, because `r.month` is NULL for all others. Put such conditions into the `ON` clause if unmatched rows must stay: `LEFT JOIN reviews_monthly r ON r.listing_id = l.id AND r.month >= '2026-01-01'` keeps all 12,776 listings.

## Check your understanding

1. Which column is the primary key of `listings`, and which columns form the key of `calendar`? What would the database do with a calendar row for a listing that is not in `listings`?
2. Explain the difference between `WHERE` and `HAVING` with an example from the listings.
3. Why does `SELECT COUNT(*) FROM listings WHERE price = NULL` return 0, and how do you count the listings without a price?
4. A colleague joins `listings` with `reviews_monthly` using an inner join and reports the number of listings per district. Which listings are missing from the result, and does it matter for her question?
5. Why is `AVG(l.price)` after `listings JOIN reviews_monthly` not the average price of a listing?

## Further reading

- Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377–387. https://doi.org/10.1145/362384.362685
- The PostgreSQL Global Development Group. *PostgreSQL documentation: Tutorial* (Part I, "The SQL Language"). https://www.postgresql.org/docs/current/tutorial.html
- Software Carpentry. *Databases and SQL*. https://swcarpentry.github.io/sql-novice-survey/
- DuckDB Foundation. *SQL introduction* (DuckDB documentation). https://duckdb.org/docs/stable/sql/introduction
