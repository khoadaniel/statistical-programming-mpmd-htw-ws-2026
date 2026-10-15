# Session 3 · Extras and notebook provenance

Optional notebooks for self-study and the sources curated into the two block notebooks, with their provenance.

## Notebooks

| File | Content | Block | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|---|
| [01-data100-sql-basics.ipynb](01-data100-sql-basics.ipynb) | Databases against CSV files, schema, `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `OFFSET`, `GROUP BY` on toy tables | 1 | [DS-100/fa23-course-notes, `sql_I/sql_I.qmd`](https://raw.githubusercontent.com/DS-100/fa23-course-notes/main/sql_I/sql_I.qmd) (commit `88328f2`) | [BSD 3-Clause](https://github.com/DS-100/fa23-course-notes/blob/main/LICENSE) | 2026-10-14 | Converted from Quarto (`.qmd`) to `.ipynb` (front matter dropped, callout boxes turned into quotes); `%load_ext sql` replaced by `import sqlite3`, the `%%sql` connection cell by `sqlite3.connect(...)` and each `%%sql` query by `pd.read_sql_query("""...""", con)`; a provenance line added at the top; executed in the course environment. Text unchanged. Curated into block 1, sections "What a relational database is", "Creating tables and loading data" and "Queries", and block 2, section "Aggregation with GROUP BY and HAVING" |
| [02-loading-with-constraints.ipynb](02-loading-with-constraints.ipynb) | A strict course database with keys and checks for all three tables, rejected rows, documented loading decisions, counts against the source files; writes only to a temporary folder | 1, 2 | course team | CC BY 4.0 | (course-written) | Renamed from `01-case-study-loading-with-constraints.ipynb`; content unchanged. Curated into block 1, section "Creating tables and loading data", subsection "Schema first" |
| [03-data100-sql-having-cleaning-joins.ipynb](03-data100-sql-having-cleaning-joins.ipynb) | `HAVING` against `WHERE`, `LIKE`, `CAST`, `CASE`, star schemas and all join types with figures | 2 | [DS-100/fa23-course-notes, `sql_II/sql_II.qmd`](https://raw.githubusercontent.com/DS-100/fa23-course-notes/main/sql_II/sql_II.qmd) (commit `88328f2`) | [BSD 3-Clause](https://github.com/DS-100/fa23-course-notes/blob/main/LICENSE) | 2026-10-14 | Converted as extra 01; the `imdbmini.db` cells (`LIKE`, `CAST`, `CASE`) are skipped with a message when the IMDb extract is missing (see below); executed in the course environment. Text unchanged. Curated into block 1, sections "What a relational database is", "Primary keys, foreign keys and relationships" and "Queries", and block 2, sections "Aggregation with GROUP BY and HAVING" and "Joins" |
| [04-claude-text-to-sql.ipynb](04-claude-text-to-sql.ipynb) | A language model writes SQL from a question and the schema of a small SQLite table | 2 | [anthropics/claude-cookbooks, `misc/how_to_make_sql_queries.ipynb`](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/misc/how_to_make_sql_queries.ipynb) | [MIT](https://github.com/anthropics/claude-cookbooks/blob/main/LICENSE) | 2026-10-14 | Renamed from `how_to_make_sql_queries.ipynb`; content unchanged. Needs the `anthropic` package (not in the course environment) and an API key, and writes `test_db.db` into the folder it runs in: run it from a temporary folder. Curated into block 2, section "Running SQL from Python", subsection "SQL written by a language model" |

Two remarks on the Data 100 notes, which were written for the SQLite version of 2023: SQLite supports `RIGHT JOIN` and `FULL OUTER JOIN` since version 3.39 (2022), although the notes say it does not; and the text of extra 01 mentions a missing value in the `Dragon` table that the bundled `basic_examples.db` does not contain. The block notebooks use the current behaviour.

## Authors and licences

- **UC Berkeley Data 100 (Fall 2023 course notes):** extras 01 and 03, the database `data/basic_examples.db` and the figures in `images/`. BSD 3-Clause, Copyright (c) 2022 UC Berkeley Data 100.
- **Anthropic, Claude cookbooks:** extra 04, MIT.
- **Course team:** extra 02, the block notebooks, the scripts and SQL files in `workbooks/`, and the figures in `../../figures/`. CC BY 4.0.

Keep the original attribution and licence notices when reusing these sources. The block notebooks mark each curated section with an "Adapted from" line.

## Data files

| File | Content | Source | Licence | Downloaded |
|---|---|---|---|---|
| [data/basic_examples.db](data/basic_examples.db) | SQLite database with the toy tables `Dragon`, `Dish` and `Scene` (28 KB), used by extras 01 and 03 | [DS-100/fa23-course-notes, `sql_I/data/basic_examples.db`](https://github.com/DS-100/fa23-course-notes/raw/main/sql_I/data/basic_examples.db) | BSD 3-Clause | 2026-10-14 |
| `data/imdbmini.db` (not included) | IMDb extract used by a few cells of extra 03 (10 MB) | [DS-100/fa23-course-notes, `sql_II/data/imdbmini.db`](https://github.com/DS-100/fa23-course-notes/raw/main/sql_II/data/imdbmini.db) | IMDb data are for personal and non-commercial use only ([IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)), so the file is not stored in the course repository. To run those cells, download it into `extras/data/` yourself | (not downloaded) |

The course database `case-study/data/airbnb/airbnb.sqlite` is built by [`../make_airbnb_sqlite.py`](../make_airbnb_sqlite.py) from the Inside Airbnb Berlin snapshot of 26 June 2026 (CC BY 4.0, [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/)), as prepared by the lecturer (personal data removed, 20 % of the listings held back as a test set) and downloaded as `airbnb.zip` from the course drive (see [case-study/README.md](../../../../case-study/README.md)). Exercise 2.9 uses the daily Berlin weather from the [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) (CC BY 4.0), which comes in the same download. Neither is stored in this folder.

## Figures

| File | Content | Source | Licence |
|---|---|---|---|
| `images/cats.png`, `cross.png`, `inner.png`, `left.png`, `right.png`, `full.png`, `student.png`, `inequality.png`, `multidimensional.png`, `star.png` | Join types and star schemas, used by extra 03 and the block notebooks | [DS-100/fa23-course-notes, `sql_II/images/`](https://github.com/DS-100/fa23-course-notes/tree/main/sql_II/images), downloaded 2026-10-14 | BSD 3-Clause |
| [`../../figures/join-types.png`](../../figures/join-types.png), [`../../figures/reviews-per-year-window.png`](../../figures/reviews-per-year-window.png) | Inner and left join of small listings and reviews tables; reviews per year with the year-over-year change. Made by [`../../figures/make_figures.py`](../../figures/make_figures.py) from the Inside Airbnb Berlin reviews | course team | CC BY 4.0 |

## Supporting files of the block notebooks

- [`../make_airbnb_sqlite.py`](../make_airbnb_sqlite.py): builds `case-study/data/airbnb/airbnb.sqlite` from the case-study Parquet files.
- [`../district_prices.py`](../district_prices.py): command-line script, short-stay listings and average price per room type in one district.
- [`../sql/01-schema.sql`](../sql/01-schema.sql), [`../sql/02-rank-districts-by-price.sql`](../sql/02-rank-districts-by-price.sql), [`../sql/03-reviews-running-total.sql`](../sql/03-reviews-running-total.sql): strict schema with keys and constraints; districts ranked by price; running totals of reviews.

## References cited in the block notebooks

- Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377–387. https://doi.org/10.1145/362384.362685
- Inside Airbnb. *Data assumptions*. https://insideairbnb.com/data-assumptions/
- Library of Congress. *Recommended formats statement: datasets*. https://www.loc.gov/preservation/resources/rfs/data.html
- OWASP. *SQL injection prevention cheat sheet*. https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
- Python Software Foundation. *sqlite3: DB-API 2.0 interface for SQLite databases*. https://docs.python.org/3/library/sqlite3.html
- SQLite. *Datatypes in SQLite*; *STRICT tables*; *Window functions*; *The WITH clause*; *Most widely deployed and used database engine*. https://www.sqlite.org/docs.html
- Software Carpentry. *Databases and SQL*. https://swcarpentry.github.io/sql-novice-survey/
- The PostgreSQL Global Development Group. *PostgreSQL documentation* (COPY, window functions). https://www.postgresql.org/docs/current/

More practice outside the course material: [SQLBolt](https://sqlbolt.com/) and the [SQL Murder Mystery](https://mystery.knightlab.com/) of Knight Lab.
