# Session 3 · Relational databases and SQL with PostgreSQL

> **Guiding question.** How do we store, query and combine data in a relational database?

## 1 · Relational databases and first queries

[Open notebook](workbooks/01-lecture-relational-databases-and-first-queries.ipynb)

- Tables, rows, columns and PostgreSQL's data types; real dates and date arithmetic, true/false, strict types, `CAST` and `::`.
- Primary keys, foreign keys and relationships, checked on the course data; normalisation, star and snowflake schemas.
- PostgreSQL as a database server on your own computer: connections, `information_schema`, `psql`, database clients.
- `CREATE TABLE` with constraints, `INSERT ... RETURNING`, transactions and `COPY`.
- `SELECT`, `WHERE`, `ORDER BY`, `LIMIT` and the logical order of the clauses; text patterns with `LIKE`, `ILIKE` and regular expressions.
- NULL values and three-valued logic.
- Worked example: a weekend flat for a family.

## 2 · Aggregation, joins and SQL from Python

[Open notebook](workbooks/02-lecture-aggregation-joins-and-sql-from-python.ipynb)

- `GROUP BY` and `HAVING`; shares as averages of 0/1 values.
- Inner, left, right, full and cross joins; anti-joins, and the duplicated rows that one-to-many joins can produce.
- Subqueries and common table expressions, including recursive ones.
- Window functions: `PARTITION BY`, frames with `ROWS BETWEEN`, rankings, running totals, `LAG`, `LEAD` and moving averages.
- Python's `psycopg`: connections, cursors, parameters, transactions, pandas, scripts, and SQL written by a language model.
- Worked example: district price rankings and reviews per district.

## Team project until the next session

Download `ebti-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)). Write a loading script that stores the decisions from `case-study/data/ebti/decisions.parquet` in a PostgreSQL database of the team (for example a database `ebti` on the local server), then count them in SQL per start year (`start_date`), language and member state (`issuing_country`).

## Setup

- Download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- Build the course database once: `uv run python sessions/03-sql/workbooks/make_airbnb_postgres.py` starts a local PostgreSQL server (data in `case-study/data/airbnb/pgdata/`, not committed) and loads the tables into the database `airbnb`. The notebooks run this step themselves; `--force` rebuilds the tables, for example after exercise 2.9.
- Optional: a database client such as [DBeaver](https://dbeaver.io/) or [pgAdmin](https://www.pgadmin.org/) to look at the tables while the server runs.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
