# Session 3 · Relational databases and SQL with SQLite

> **Guiding question.** How do we store, query and combine data in a relational database?

## 1 · Relational databases and first queries

[Open notebook](workbooks/01-lecture-relational-databases-and-first-queries.ipynb)

- Tables, rows, columns and SQLite's data types; dates as text, true/false as 1/0.
- Primary keys, foreign keys and relationships, checked on the course data.
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
- Window functions: rankings, running totals, `LAG` and moving averages.
- Python's `sqlite3`: connections, cursors, parameters, transactions, pandas, scripts.
- Worked example: district price rankings and reviews per district.

## Team project until the next session

Download the customs decisions with the provided script (see [case-study/README.md](../../case-study/README.md)), load them into the team's SQLite database with a loading script, and count them per year, language and member state in SQL.

## Setup

- Build the course database once (it is written to `case-study/data/airbnb.sqlite`, which is not committed): `uv run python case-study/prepare_airbnb.py` (if `case-study/data/airbnb/` is missing), then `uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py`. The notebooks run the second step themselves if the file is missing; `--force` rebuilds it, for example after exercise 2.9.
- Optional: [DB Browser for SQLite](https://sqlitebrowser.org/) to look at the file.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
