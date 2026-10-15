# Session 3 · Relational databases and SQL with SQLite

> **Guiding question.** How do we store, query and combine data in a relational database?

## 1 · Relational databases and first queries

[Open notebook](workbooks/01-lecture-relational-databases-and-first-queries.ipynb)

- Tables, rows, columns and SQLite's data types; dates as text, true/false as 1/0; `STRICT` tables and `CAST`.
- Primary keys, foreign keys and relationships, checked on the course data; normalisation and star schemas.
- SQLite as a database in one file: connections, `sqlite_master`, database browsers.
- `CREATE TABLE` with constraints, `INSERT`, and the SQLite equivalents of `COPY`.
- `SELECT`, `WHERE`, `ORDER BY`, `LIMIT` and the logical order of the clauses.
- NULL values and three-valued logic.
- Worked example: a weekend flat for a family.

## 2 · Aggregation, joins and SQL from Python

[Open notebook](workbooks/02-lecture-aggregation-joins-and-sql-from-python.ipynb)

- `GROUP BY` and `HAVING`; shares as averages of 0/1 values.
- Inner, left, right, full and cross joins; anti-joins and one-to-many traps.
- Subqueries and common table expressions, including recursive ones.
- Window functions: rankings, running totals, `LAG`, `LEAD` and moving averages.
- Python's `sqlite3`: connections, cursors, parameters, transactions, pandas, scripts, and SQL written by a language model.
- Worked example: district price rankings and reviews per district.

## Team project until the next session

Download `ebti.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)). Write a loading script that stores the training decisions `case-study/data/ebti/train.parquet` in the team's SQLite database, then count them in SQL per start year (`start_date`), language and member state (`issuing_country`).

## Setup

- Download `airbnb.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- Build the course database once (it is written to `case-study/data/airbnb/airbnb.sqlite`, which is not committed): `uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py`. The notebooks run this step themselves if the file is missing; `--force` rebuilds it, for example after exercise 2.9.
- Optional: [DB Browser for SQLite](https://sqlitebrowser.org/) to look at the file.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
