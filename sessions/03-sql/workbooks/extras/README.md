# Session 3 · Extras and notebook provenance

Optional material for self-study and the sources curated into the two block notebooks, with their provenance.

## Notebooks

| File | Content | Block | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|---|
| [02-loading-with-constraints.ipynb](02-loading-with-constraints.ipynb) | A strict course database with keys and checks for all three tables, rejected rows, documented loading decisions, counts against the source files; works in its own schema of the local PostgreSQL server | 1, 2 | course team | CC BY 4.0 | (course-written) | Curated into block 1, section "Creating tables and loading data", subsection "Schema first" |

## Authors and licences

- **UC Berkeley Data 100 (Fall 2023 course notes):** the figures in `images/`, and the sections of the block notebooks marked "Adapted from Data 100". BSD 3-Clause, Copyright (c) 2022 UC Berkeley Data 100. Source: [DS-100/fa23-course-notes](https://github.com/DS-100/fa23-course-notes) (`sql_I/sql_I.qmd`, `sql_II/sql_II.qmd`, commit `88328f2`).
- **Course team:** extra 02, the block notebooks, the scripts and SQL files in `workbooks/`, and the figures in `../../figures/`. CC BY 4.0.

Keep the original attribution and licence notices when reusing these sources. The block notebooks mark each curated section with an "Adapted from" line.

## Data

The course database `airbnb` on the local PostgreSQL server is built by [`../make_airbnb_postgres.py`](../make_airbnb_postgres.py) from the Inside Airbnb Berlin snapshot of 26 June 2026 (CC BY 4.0, [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/)); the server keeps its data in `case-study/data/airbnb/pgdata/`. The lecturer prepared these data (personal data removed, 20 % of the listings held back as a test set), and you download them as `airbnb-v1.zip` from the course drive (see [case-study/README.md](../../../../case-study/README.md)). Exercise 2.9 uses the daily Berlin weather from the [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) (CC BY 4.0), which comes in the same download. Neither is stored in this folder.

## Figures

| File | Content | Source | Licence |
|---|---|---|---|
| `images/cats.png`, `cross.png`, `inner.png`, `left.png`, `right.png`, `full.png`, `student.png`, `inequality.png`, `multidimensional.png`, `star.png` | Join types and star schemas, used by the block notebooks | [DS-100/fa23-course-notes, `sql_II/images/`](https://github.com/DS-100/fa23-course-notes/tree/main/sql_II/images), downloaded 2026-10-14 | BSD 3-Clause |
| [`../../figures/join-types.png`](../../figures/join-types.png), [`../../figures/reviews-per-year-window.png`](../../figures/reviews-per-year-window.png) | Inner and left join of small listings and reviews tables; reviews per year with the year-over-year change. Made by [`../../figures/make_figures.py`](../../figures/make_figures.py) from the Inside Airbnb Berlin reviews | course team | CC BY 4.0 |

## Supporting files of the block notebooks

- [`../make_airbnb_postgres.py`](../make_airbnb_postgres.py): starts the local PostgreSQL server and loads the case-study Parquet files into the database `airbnb` with `COPY`.
- [`../district_prices.py`](../district_prices.py): command-line script, short-stay listings and average price per room type in one district.
- [`../sql/01-schema.sql`](../sql/01-schema.sql), [`../sql/02-rank-districts-by-price.sql`](../sql/02-rank-districts-by-price.sql), [`../sql/03-reviews-running-total.sql`](../sql/03-reviews-running-total.sql): strict schema with keys and constraints; districts ranked by price; running totals of reviews.

## References cited in the block notebooks

- Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377–387. https://doi.org/10.1145/362384.362685
- Inside Airbnb. *Data assumptions*. https://insideairbnb.com/data-assumptions/
- OWASP. *SQL injection prevention cheat sheet*. https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
- Pixeltable. *pixeltable-pgserver*: PostgreSQL as a Python package. https://github.com/pixeltable/pixeltable-pgserver
- Psycopg. *Psycopg 3 documentation*. https://www.psycopg.org/psycopg3/docs/
- Software Carpentry. *Databases and SQL*. https://swcarpentry.github.io/sql-novice-survey/
- The PostgreSQL Global Development Group. *PostgreSQL documentation*: tutorial, data types, date/time functions, pattern matching, COPY, window functions, `WITH` queries. https://www.postgresql.org/docs/current/

More practice outside the course material: [SQLBolt](https://sqlbolt.com/), the [SQL Murder Mystery](https://mystery.knightlab.com/) of Knight Lab, and [PostgreSQL Exercises](https://pgexercises.com/).
