# The relational model and basic SQL

This page covers the first block of the session: how a relational database organises data in tables linked by keys, why the course uses PostgreSQL, and the core of SQL: selecting, filtering, sorting, aggregating and joining. Almost every organisation keeps its operational data in relational databases, and SQL is the language analysts use to get data out of them. In job advertisements for data roles, SQL is among the most frequently requested skills after Python. All examples use the two tables of the course case study, `reviews` (one row per review) and `products` (one row per product).

```mermaid
flowchart LR
    Q["Business question"] --> S["SQL query"]
    S --> DB[("PostgreSQL<br/>reviews, products")]
    DB --> R["Small result table"]
    R --> P["pandas / Polars<br/>in Python"]
    P --> A["Chart, model, report"]
```

> [!NOTE]
> You can run every query on this page without a database server: DuckDB, an embedded database, reads the Parquet files of the case study directly (see [How it works in Python](#how-it-works-in-python)). PostgreSQL itself is set up in block 2.

## The relational model: tables, primary and foreign keys, relationships

### Concept

A **relational database** stores data in **tables** (also called relations). Each **row** is one record; each **column** has a name and a **data type** (text, integer, decimal number, date, true/false). The set of table definitions is the **schema**. The idea goes back to Edgar F. Codd (1970), who proposed that data should be described by tables and queried by their content, not by the way they are stored on disk.

A **primary key** is a column, or a combination of columns, whose value identifies each row uniquely and is never empty. In the case study, `review_id` identifies a review and `parent_asin` (Amazon's product identifier) identifies a product.

A **foreign key** is a column that refers to the primary key of another table. `reviews.parent_asin` points to `products.parent_asin`. This expresses a **relationship**: one product has many reviews, each review belongs to exactly one product (a *one-to-many* relationship). The database can enforce it: a review of a product that does not exist is rejected.

Storing each fact once, in one table, is called **normalisation**. The store name of a product is stored once in `products`, not repeated in each of its 3,000 reviews. If the store is renamed, one row changes, and no copy can contradict another.

A small example that you can follow by hand:

| products |  |  |
|---|---|---|
| **parent_asin** (PK) | title | price |
| A1 | Vitamin D3 drops | 12.99 |
| B2 | Pill organiser | NULL |

| reviews |  |  |
|---|---|---|
| **review_id** (PK) | parent_asin (FK) | rating |
| r1 | A1 | 5 |
| r2 | A1 | 2 |
| r3 | B2 | 4 |

Product A1 has two reviews, B2 has one. `NULL` marks a value that is unknown: the price of B2 was not recorded.

```mermaid
classDiagram
    direction LR
    class products {
        parent_asin : text  PK
        title : text
        store : text
        price : numeric
        train_avg_rating : numeric
    }
    class reviews {
        review_id : text  PK
        parent_asin : text  FK
        rating : smallint
        text : text
        date : timestamp
        label : text
    }
    products "1" --> "0..*" reviews : has
```

### Why it matters

Keys make the relationship between tables explicit, and the database uses them to guard data quality at the moment data are written: a rating of 6 or a review of an unknown product is rejected immediately instead of being discovered months later in an analysis. The database also builds an **index** (a sorted lookup structure) on every primary key, which makes lookups and joins fast. Without normalisation, the same fact is stored many times, and copies drift apart.

### How it works in Python

The case-study tables arrive as Parquet files. The following check confirms the two key properties with pandas: the primary key is unique, and every foreign key value has a partner.

```python
import pandas as pd

reviews = pd.read_parquet("case-study/data/train.parquet")
products = pd.read_parquet("case-study/data/products.parquet")

print(reviews["review_id"].is_unique, products["parent_asin"].is_unique)   # True True
# foreign key: does every review refer to an existing product?
print(reviews["parent_asin"].isin(products["parent_asin"]).all())          # True
# one-to-many: reviews per product
print(reviews.groupby("parent_asin").size().describe()[["mean", "50%", "max"]].round(1))
# mean 7.8 | 50% 2.0 | max 2997.0
```

The same rules written as SQL create the tables with **constraints**. `CHECK` adds a rule for the values; `REFERENCES` declares the foreign key.

```sql
CREATE TABLE shop_products (
    parent_asin TEXT PRIMARY KEY,                 -- one row per product
    title       TEXT NOT NULL,
    price       NUMERIC(8, 2) CHECK (price >= 0)  -- NULL allowed: price unknown
);
CREATE TABLE shop_reviews (
    review_id   TEXT PRIMARY KEY,
    parent_asin TEXT NOT NULL REFERENCES shop_products (parent_asin),  -- foreign key
    rating      SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5)
);
INSERT INTO shop_products VALUES ('A1', 'Vitamin D3 drops', 12.99), ('B2', 'Pill organiser', NULL);
INSERT INTO shop_reviews VALUES ('r1', 'A1', 5), ('r2', 'A1', 2), ('r3', 'B2', 4);
INSERT INTO shop_reviews VALUES ('r4', 'Z9', 3);   -- ERROR: violates foreign key constraint
INSERT INTO shop_reviews VALUES ('r5', 'A1', 6);   -- ERROR: violates check constraint
```

### In practice

- **Stack Exchange Data Explorer** lets anyone query the public Stack Overflow database with SQL; questions, answers, users and votes are separate tables linked by identifiers such as `PostId` and `UserId`.
- **Wikimedia** offers public read-only copies of the Wikipedia databases through the Quarry service; pages, revisions and users are related tables, and volunteers answer research questions about Wikipedia with SQL.
- **Hospital information systems** store patients, admissions and diagnoses in separate tables; a patient appears once in the patient table and many times in the admissions table, linked by a patient identifier.

> [!WARNING]
> A file is not a database. A Parquet or CSV file has no keys and no constraints: nothing stops a duplicated `review_id` or a rating of 6. When you work with files, check the key properties yourself (as in the Python block above) and repeat the check after each update.

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
con.execute("CREATE VIEW reviews AS SELECT * FROM 'case-study/data/train.parquet'")
con.execute("CREATE VIEW products AS SELECT * FROM 'case-study/data/products.parquet'")
print(con.sql("SELECT COUNT(*) AS n FROM reviews").fetchone())   # (434373,)

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

By hand: in the small tables above, `SELECT review_id FROM reviews WHERE rating >= 4 ORDER BY review_id` keeps r1 (5) and r3 (4) and returns them in the order r1, r3.

### Why it matters

Filtering in the database moves only the rows you need into Python. With millions of rows, `SELECT *` followed by filtering in pandas wastes memory and time. Writing the condition explicitly also makes the question precise: "recent negative reviews" becomes `rating <= 2 AND date >= '2021-01-01'`.

### How it works in Python

```sql
SELECT review_id, rating, helpful_vote, LEFT(title, 30) AS title
FROM reviews
WHERE rating = 1 AND verified_purchase      -- both conditions must hold
ORDER BY helpful_vote DESC                  -- most helpful first
LIMIT 3;
-- r232284 | 1 | 400 | Unreliable!
-- r220303 | 1 | 399 | Tired it, didn't work for me.
-- r200692 | 1 | 386 | I want to live!
```

From Python with DuckDB, a query result becomes a pandas DataFrame with `.df()`:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW reviews AS SELECT * FROM 'case-study/data/train.parquet'")

print(con.sql("""
    SELECT COUNT(*) AS n FROM reviews
    WHERE rating <= 2 AND date >= '2021-01-01'
""").fetchone())                                                   # (15882,)
print(con.sql("SELECT COUNT(*) FROM reviews WHERE title ILIKE '%refund%'").fetchone())  # (147,)
top = con.sql("SELECT review_id, helpful_vote FROM reviews ORDER BY helpful_vote DESC LIMIT 3").df()
print(top)
#   review_id  helpful_vote
# 0   r016260          7326
# 1   r076389          5907
# 2   r007437          1343
```

### In practice

- **Customer-service teams** at online retailers filter recent negative reviews with many helpful votes to find product problems first.
- The **European Medicines Agency's** EudraVigilance database of suspected side effects is queried by drug, period and outcome before any statistical signal detection is done.
- **Data engineers** check every nightly load with small queries such as `SELECT COUNT(*) FROM orders WHERE order_date = CURRENT_DATE - 1`.

> [!CAUTION]
> Without `ORDER BY`, the order of rows is not defined. `LIMIT 10` without `ORDER BY` returns *some* ten rows, which may change between runs or database versions.

## Aggregation with GROUP BY and HAVING

### Concept

An **aggregate function** reduces many values to one: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`. `GROUP BY` splits the rows into groups with equal values of the grouping columns and computes the aggregates per group; the result has one row per group. Every column in `SELECT` must either be a grouping column or be inside an aggregate.

`WHERE` filters **rows before** grouping; `HAVING` filters **groups after** aggregation, for example "only products with at least 1000 reviews".

A **share** is the average of a 0/1 variable. `AVG((rating <= 2)::int)` is the share of 1–2 star reviews, because `::int` turns true/false into 1/0.

By hand: grouping the three example reviews by `parent_asin` gives A1 with `COUNT(*) = 2`, `AVG(rating) = 3.5`, and B2 with 1 and 4.0. `HAVING COUNT(*) >= 2` keeps only A1.

### Why it matters

Almost every business question is an aggregate: reviews per month, average rating per store, share of complaints per product. `HAVING` with a minimum count prevents tiny groups from topping a ranking by chance: a product with one 5-star review has a perfect average.

### How it works in Python

```sql
SELECT rating, COUNT(*) AS n_reviews
FROM reviews
GROUP BY rating
ORDER BY rating;
-- 1: 58129 | 2: 25349 | 3: 32482 | 4: 51043 | 5: 267370

SELECT parent_asin, COUNT(*) AS n_reviews, ROUND(AVG(rating), 2) AS avg_rating
FROM reviews
GROUP BY parent_asin
HAVING COUNT(*) >= 1000          -- filter groups, not rows
ORDER BY avg_rating
LIMIT 3;
-- B0CCWPQL6X | 1029 | 2.92
-- B00RTFT08W | 1069 | 3.10
-- B0077L8YFI | 1966 | 3.19
```

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW reviews AS SELECT * FROM 'case-study/data/train.parquet'")
yearly = con.sql("""
    SELECT EXTRACT(YEAR FROM date)::int AS year,
           COUNT(*) AS n,
           ROUND(AVG((rating <= 2)::int), 3) AS share_neg
    FROM reviews
    WHERE date >= '2019-01-01'
    GROUP BY year
    ORDER BY year
""").df()
print(yearly)
#    year      n  share_neg
# 0  2019  57594      0.188
# 1  2020  75904      0.218
# 2  2021  68456      0.232
```

### In practice

- **Retail dashboards** report sales and returns per store and week; each tile is a `GROUP BY` query on a transaction table.
- **Statistical offices** such as Destatis publish counts and rates per region and year computed from individual-level registers.
- **Hospital comparisons** such as Care Compare of the US Centers for Medicare & Medicaid Services do not publish a measure when a hospital has too few cases, which is the idea of `HAVING COUNT(*) >= k`.

> [!WARNING]
> `COUNT(*)` counts rows; `COUNT(column)` counts non-NULL values of that column; `AVG(column)` silently ignores NULLs. On the products table, `COUNT(*)` is 60,274 but `COUNT(price)` is 10,535, and `AVG(price)` is the average of the known prices only.

## INNER and LEFT JOIN; NULL values in joins and aggregates

### Concept

A **join** combines rows of two tables whose key columns match. An **inner join** (`JOIN ... ON a.key = b.key`) returns one row per matching pair; rows without a partner disappear. Because each review has exactly one product, joining reviews with products keeps one row per review and adds the product columns. `USING (parent_asin)` is a short form when the key has the same name in both tables. Short **aliases** (`r`, `p`) say which table a column comes from.

A **left join** keeps every row of the left table; where no partner exists, the columns of the right table are `NULL`. Combined with `WHERE right.key IS NULL`, it finds rows **without** a partner, an **anti-join**: for example, products that were never reviewed.

![Inner and left join of a small products and reviews table](figures/join-types.png)

**NULL** means "unknown". Any comparison with NULL is unknown, not true or false, so `price = NULL` never matches; use `price IS NULL`. In `GROUP BY`, all NULLs form one group, which is why a ranking of stores contains a row with store `NULL`.

By hand, with the tables above plus a product C3 without reviews: `products JOIN reviews` gives 3 rows (r1, r2, r3); `products LEFT JOIN reviews` gives 4 rows, the fourth being C3 with `review_id = NULL`.

### Why it matters

Normalised data must be joined before they can be analysed. The join type decides which rows survive and therefore which population a number describes. Many wrong numbers in reports come from an unintended inner join, which drops unmatched rows silently, or from a join on a key that is not unique, which duplicates rows.

### How it works in Python

```sql
-- average rating per store: each review gets the columns of its product
SELECT p.store, COUNT(*) AS n_reviews, ROUND(AVG(r.rating), 2) AS avg_rating
FROM reviews AS r
JOIN products AS p ON p.parent_asin = r.parent_asin
GROUP BY p.store
HAVING COUNT(*) >= 1000
ORDER BY avg_rating DESC, n_reviews DESC
LIMIT 3;
-- ASUTRA | 2997 | 4.74
-- Essential Depot | 2123 | 4.74
-- Pure Acres Farm | 1317 | 4.74

-- anti-join: products without any review in the training period
SELECT COUNT(*) AS products_without_reviews
FROM products AS p
LEFT JOIN reviews AS r ON r.parent_asin = p.parent_asin
WHERE r.review_id IS NULL;
-- 4915
```

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW reviews AS SELECT * FROM 'case-study/data/train.parquet'")
con.execute("CREATE VIEW products AS SELECT * FROM 'case-study/data/products.parquet'")

# practice question: how many products have no price, and do their reviews differ?
print(con.sql("""
    SELECT p.price IS NULL AS price_missing,
           COUNT(DISTINCT r.parent_asin) AS n_products,
           COUNT(*) AS n_reviews,
           ROUND(AVG(r.rating), 2) AS avg_rating
    FROM reviews AS r
    LEFT JOIN products AS p USING (parent_asin)
    GROUP BY price_missing
    ORDER BY price_missing
""").df())
#    price_missing  n_products  n_reviews  avg_rating
# 0          False        9107     137937        4.18
# 1           True       46252     296436        3.95

n_groups = con.sql("""SELECT COUNT(*) FROM (SELECT p.store FROM reviews r JOIN products p USING (parent_asin)
                      GROUP BY p.store HAVING COUNT(*) >= 1000)""").fetchone()
print(n_groups)                                                        # (30,) one of them is store NULL
```

The reviews of products without a price have a lower average rating (3.95 against 4.18). Whether the missing price is related to other properties of a product is the starting question of Session 4.

### In practice

- **Marketing analysts** join transactions with customer attributes to compare customer segments; a customer without transactions disappears in an inner join and must be kept with a left join if "inactive customers" are part of the question.
- **Epidemiological record linkage**, for example linking cancer registries with mortality registers, keeps unmatched records and analyses them separately, because who fails to link is itself informative.
- **Data quality checks** in finance and retail use anti-joins to find orders that refer to a deleted customer or a product that no longer exists.

> [!CAUTION]
> A join on a key that is not unique on the "one" side multiplies rows. If `products` contained the same `parent_asin` twice, every review of that product would appear twice after the join and all counts would be inflated. Check uniqueness before joining (`COUNT(*)` against `COUNT(DISTINCT key)`), or in pandas use `merge(..., validate="many_to_one")`.

> [!WARNING]
> A condition on the right table in `WHERE` turns a left join back into an inner join: `LEFT JOIN products p ... WHERE p.price > 20` drops all rows where `p.price` is NULL. Put such conditions into the `ON` clause if unmatched rows must stay.

## Check your understanding

1. Which column is the primary key of `reviews`, and which column is a foreign key? What would the database do with a review whose `parent_asin` does not exist in `products`?
2. Explain the difference between `WHERE` and `HAVING` with an example from the review data.
3. Why does `SELECT COUNT(*) FROM products WHERE price = NULL` return 0?
4. A colleague joins `reviews` with `products` using an inner join and reports the average price of reviewed products. Which products are missing from the result, and does it matter for her question?
5. Which logical step of a query is evaluated first: `SELECT` or `WHERE`? What follows for column aliases?

## Further reading

- Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377–387. https://doi.org/10.1145/362384.362685
- The PostgreSQL Global Development Group. *PostgreSQL documentation: Tutorial* (Part I, "The SQL Language"). https://www.postgresql.org/docs/current/tutorial.html
- Software Carpentry. *Databases and SQL*. https://swcarpentry.github.io/sql-novice-survey/
- DuckDB Foundation. *SQL introduction* (DuckDB documentation). https://duckdb.org/docs/stable/sql/introduction
