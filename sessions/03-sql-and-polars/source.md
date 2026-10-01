# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/01-sql-select-where-order.ipynb](workbooks/01-sql-select-where-order.ipynb) | SQL via jupysql+DuckDB: SELECT, WHERE, ORDER BY, LIMIT, DISTINCT | [DS-100/course-notes (UC Berkeley Data 100)](https://raw.githubusercontent.com/DS-100/course-notes/main/content/sql_I/sql_I.ipynb) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-09-26 | Renamed from `ds100_sql-I.ipynb`; content unchanged. Uses `data/example_duck.db` |
| [workbooks/02-sql-first-query.ipynb](workbooks/02-sql-first-query.ipynb) | jupysql+DuckDB intro: SELECT/WHERE/ORDER BY on UCI bank marketing data | [ploomber/sql](https://raw.githubusercontent.com/ploomber/sql/main/colabs/intro-to-sql/making-your-first-query.ipynb) | [Apache-2.0](https://github.com/ploomber/sql/blob/main/LICENSE) | 2026-09-26 | Renamed from `ploomber-sql_01-first-query.ipynb`. 2026-10-01: import path `sys.path.insert(0, "../../")` changed to `"."` so that `banking.py` in the same folder is found |
| [workbooks/03-sql-aggregate-functions.ipynb](workbooks/03-sql-aggregate-functions.ipynb) | COUNT/SUM/AVG/MIN/MAX, GROUP BY, HAVING | [ploomber/sql](https://raw.githubusercontent.com/ploomber/sql/main/colabs/intro-to-sql/aggregate-functions-in-sql.ipynb) | [Apache-2.0](https://github.com/ploomber/sql/blob/main/LICENSE) | 2026-09-26 | Renamed from `ploomber-sql_02-aggregate-functions.ipynb`. 2026-10-01: import path changed to `"."` (as above) |
| [workbooks/04-sql-groupby-joins-ctes.ipynb](workbooks/04-sql-groupby-joins-ctes.ipynb) | GROUP BY, HAVING, LIKE, CAST, CASE, INNER/OUTER JOIN, CTEs (IMDb) | [DS-100/course-notes (UC Berkeley Data 100)](https://raw.githubusercontent.com/DS-100/course-notes/main/content/sql_II/sql_II.ipynb) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-09-26 | Renamed from `ds100_sql-II.ipynb`; content unchanged. Uses `data/basic_examples.db`; downloads `imdb_duck.db` from Google Drive with `gdown` on first run |
| [workbooks/05-sql-joins.ipynb](workbooks/05-sql-joins.ipynb) | INNER/LEFT/RIGHT/FULL joins on Czech bank (Berka) tables | [ploomber/sql](https://raw.githubusercontent.com/ploomber/sql/main/colabs/intro-to-sql/joining-data-in-sql.ipynb) | [Apache-2.0](https://github.com/ploomber/sql/blob/main/LICENSE) | 2026-09-26 | Renamed from `ploomber-sql_03-joins.ipynb`. 2026-10-01: import path changed to `"."`; data URL `https://tinyurl.com/jb-bank-m` (redirects to `sorry.vse.cz`, which no longer resolves) replaced by the Internet Archive copy `https://web.archive.org/web/20210508083251id_/https://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip` |
| [workbooks/07-sql-window-functions.ipynb](workbooks/07-sql-window-functions.ipynb) | Window functions: RANK/DENSE_RANK OVER (PARTITION BY), moving averages, ROLLUP | [ploomber/sql](https://raw.githubusercontent.com/ploomber/sql/main/colabs/advanced-querying-techniques/advanced-aggregations.ipynb) | [Apache-2.0](https://github.com/ploomber/sql/blob/main/LICENSE) | 2026-09-26 | Renamed from `ploomber-sql_04-window-functions.ipynb` (was `06-...` until 2026-10-01). 2026-10-01: import path and Berka data URL changed as in 05 |
| [workbooks/08-postgres-with-python.ipynb](workbooks/08-postgres-with-python.ipynb) | PostgreSQL in Docker + SQLAlchemy/jupysql: load pandas data, JOIN/CTE queries, profiling | [ploomber/sql](https://raw.githubusercontent.com/ploomber/sql/main/video-material/postgres/postgres-intro.ipynb) | [Apache-2.0](https://github.com/ploomber/sql/blob/main/LICENSE) | 2026-09-26 | Renamed from `ploomber-sql_05-postgres-intro.ipynb` (was `07-...`). 2026-10-01: import path `"../../../"` changed to `"."`; Berka data URL changed as in 05 |
| [workbooks/09-jupysql-postgres-connect.ipynb](workbooks/09-jupysql-postgres-connect.ipynb) | How to connect jupysql to PostgreSQL (Docker), load data, query and plot | [ploomber/jupysql](https://raw.githubusercontent.com/ploomber/jupysql/master/doc/integrations/postgres-connect.ipynb) | [Apache-2.0](https://github.com/ploomber/jupysql/blob/master/LICENSE) | 2026-09-26 | Renamed from `jupysql_postgres-connect.ipynb` (was `08-...`); content unchanged |
| [workbooks/11-polars-eda-with-sql-equivalents.ipynb](workbooks/11-polars-eda-with-sql-equivalents.ipynb) | Polars basics (read_csv, sort, with_columns, filter, Series) with the equivalent SQL for each step; boxplots and histograms | [DS-100/course-notes, EDA I (UC Berkeley Data 100, Polars version)](https://raw.githubusercontent.com/DS-100/course-notes/main/content/new_eda_1/new_eda_1.ipynb) | [BSD-3-Clause](https://github.com/DS-100/course-notes/blob/main/LICENSE) | 2026-10-01 | Renamed from `new_eda_1.ipynb`; content unchanged. Images in `images/` were not copied (the notebook shows their file names instead) |
| [workbooks/12-polars-getting-started.ipynb](workbooks/12-polars-getting-started.ipynb) | Polars DataFrame, Series, LazyFrame, `explain`/`show_graph`, select/filter, with_columns, method chaining, larger-than-RAM data | [PacktPublishing/Polars-Cookbook, Chapter 1](https://raw.githubusercontent.com/PacktPublishing/Polars-Cookbook/main/Chapter01/ch01.ipynb) (Yuki Kakegawa, *Polars Cookbook*, Packt 2024) | [MIT](https://github.com/PacktPublishing/Polars-Cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `ch01.ipynb`. Data paths `'../data/'` changed to `'data/'` (6 cells). `show_graph` needs the Graphviz `dot` program; the last two cells need the Chicago taxi file (several GB, not included) |
| [workbooks/13-polars-transformations.ipynb](workbooks/13-polars-transformations.ipynb) | Aggregations, group_by, horizontal aggregations, window functions with `over`, UDFs vs expressions, `SQLContext` | [PacktPublishing/Polars-Cookbook, Chapter 4](https://raw.githubusercontent.com/PacktPublishing/Polars-Cookbook/main/Chapter04/ch04.ipynb) | [MIT](https://github.com/PacktPublishing/Polars-Cookbook/blob/main/LICENSE) | 2026-10-01 | Renamed from `ch04.ipynb`. Data paths `'../data/'` changed to `'data/'` (7 cells) |
| [workbooks/banking.py](workbooks/banking.py) | Download helpers for the bank data used by 02, 03, 05, 07, 08 | [ploomber/sql](https://github.com/ploomber/sql) | [Apache-2.0](https://github.com/ploomber/sql/blob/main/LICENSE) | 2026-09-26 | Unchanged |

### Data files in `workbooks/data/`

| File | Used by | Source | Licence | Downloaded |
|---|---|---|---|---|
| `example_duck.db`, `basic_examples.db` | 01, 04 | [DS-100/course-notes](https://github.com/DS-100/course-notes/tree/main/content/sql_I/data) | BSD-3-Clause (repository) | 2026-09-26 |
| `elections.csv`, `pivoted-ucb-data.csv`, `pivoted-ucb-data-w-enrollment.csv` | 11 | [DS-100/course-notes, new_eda_1/data](https://github.com/DS-100/course-notes/tree/main/content/new_eda_1/data) (compiled from University of California public admissions statistics) | BSD-3-Clause (repository) | 2026-10-01 |
| `titanic_dataset.csv` | 12 | [PacktPublishing/Polars-Cookbook/data](https://github.com/PacktPublishing/Polars-Cookbook/tree/main/data) (originally the Kaggle Titanic dataset) | MIT (repository); the original data carry no stated licence | 2026-10-01 |
| `contoso_sales.csv`, `pokemon.csv` | 13 | [PacktPublishing/Polars-Cookbook/data](https://github.com/PacktPublishing/Polars-Cookbook/tree/main/data) (generated with SQLBI's Contoso Data Generator; Pokémon statistics from a gist by Ritchie Vink) | MIT (repository); the Pokémon file has no separate licence | 2026-10-01 |

Data downloaded at run time (not stored in the repository): UCI Bank Marketing data (02, 03; [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing), CC-BY-4.0); PKDD'99 Berka financial data (05, 07, 08; released for the PKDD'99 Discovery Challenge, no licence stated; archived copy); IMDb extract (04; prepared by Data 100); NYC TLC yellow taxi trips, January 2021 (09; NYC Taxi and Limousine Commission open data).

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-relational-model-and-sql.md](theory/01-relational-model-and-sql.md) | Block 1: relational model, PostgreSQL, SELECT/WHERE/ORDER BY/LIMIT, GROUP BY/HAVING, joins and NULL | CC-BY-4.0 |
| [theory/02-sql-from-python-and-ingestion.md](theory/02-sql-from-python-and-ingestion.md) | Block 2: CTEs and window functions, SQLAlchemy and pandas, ingestion with constraints, data cards | CC-BY-4.0 |
| [theory/03-polars-and-choosing-a-tool.md](theory/03-polars-and-choosing-a-tool.md) | Block 3: limits of pandas, Polars (expressions, lazy, optimiser, streaming, Parquet), SQL/pandas/Polars side by side, choosing a tool | CC-BY-4.0 |
| [theory/figures/make_figures.py](theory/figures/make_figures.py) | Script for `join-types.png`, `reviews-per-year-window.png`, `pandas-polars-benchmark.png` | CC-BY-4.0 |
| [workbooks/06-case-study-sql-first-questions.ipynb](workbooks/06-case-study-sql-first-questions.ipynb) | Practice block 1: first SQL questions on the reviews with DuckDB (average rating per store, products without a price) | CC-BY-4.0 |
| [workbooks/10-case-study-postgres-reviews.ipynb](workbooks/10-case-study-postgres-reviews.ipynb) | Practice block 2: load reviews into PostgreSQL (Docker) or DuckDB with constraints, window-function ranking, data card | CC-BY-4.0 |
| [workbooks/14-case-study-pandas-vs-polars.ipynb](workbooks/14-case-study-pandas-vs-polars.ipynb) | Practice block 3: the same aggregation on all training reviews in SQL, pandas and Polars; code, runtime and peak memory | CC-BY-4.0 |
| [workbooks/sql/01-schema.sql](workbooks/sql/01-schema.sql), [02-add-constraints.sql](workbooks/sql/02-add-constraints.sql), [03-rank-products-per-year.sql](workbooks/sql/03-rank-products-per-year.sql) | Schema with keys and constraints; constraints for tables written by `prepare_data.py --postgres`; window-function ranking | CC-BY-4.0 |
| [workbooks/data-card-template.md](workbooks/data-card-template.md) | Data card template | CC-BY-4.0 |

Author of own material: course team.

## Citations

- Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377–387. https://doi.org/10.1145/362384.362685
- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92. https://arxiv.org/abs/1803.09010
- Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data cards: Purposeful and transparent dataset documentation for responsible AI. *FAccT 2022*. https://arxiv.org/abs/2204.01075
- Raasveldt, M., & Mühleisen, H. (2019). DuckDB: an embeddable analytical database. *SIGMOD 2019*, 1981–1984. https://doi.org/10.1145/3299869.3320212
- McKinney, W. (2017). *Apache Arrow and the "10 Things I Hate About pandas"*. https://wesmckinney.com/blog/apache-arrow-pandas-internals/
- Kakegawa, Y. (2024). *Polars Cookbook*. Packt. Code: https://github.com/PacktPublishing/Polars-Cookbook
- Heavey, K. *Modern Polars*. https://kevinheavey.github.io/modern-polars/ (CC-BY-4.0)
- UC Berkeley Data 100. *Course notes*. https://ds100.org/course-notes/ ; https://github.com/DS-100/course-notes
- Ploomber. *SQL course* and *JupySQL documentation*. https://github.com/ploomber/sql ; https://jupysql.ploomber.io/
- The PostgreSQL Global Development Group. *PostgreSQL documentation*. https://www.postgresql.org/docs/current/
- Polars developers. *Polars user guide*. https://docs.pola.rs/user-guide/
- SQLAlchemy. *SQLAlchemy 2.0 documentation*. https://docs.sqlalchemy.org/en/20/
- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging language and items for retrieval and recommendation. arXiv:2403.03952 (course dataset).
- Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Berka, P. (1999). Guide to the financial data set. PKDD'99 Discovery Challenge.
