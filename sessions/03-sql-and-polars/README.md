# Session 3 · Relational databases, SQL and Polars

> [!NOTE]
> **Guiding question.** How do we store, query and combine data reliably, and how do we process large tables efficiently in Python?

**Learning outcomes.** You are able to

- describe a relational schema with primary and foreign keys and write SQL queries that filter, join and aggregate data
- load data reproducibly into PostgreSQL and continue the analysis in Python
- process large tables efficiently with Polars and choose between SQL, pandas and Polars

## Session plan

**0:00–0:45 · The relational model and basic SQL** ([theory page](theory/01-relational-model-and-sql.md))

- [The relational model: tables, primary and foreign keys, relationships](theory/01-relational-model-and-sql.md#the-relational-model-tables-primary-and-foreign-keys-relationships)
- [PostgreSQL as the course database](theory/01-relational-model-and-sql.md#postgresql-as-the-course-database)
- [Basic queries: SELECT, WHERE, ORDER BY, LIMIT](theory/01-relational-model-and-sql.md#basic-queries-select-where-order-by-limit)
- [Aggregation with GROUP BY and HAVING](theory/01-relational-model-and-sql.md#aggregation-with-group-by-and-having)
- [INNER and LEFT JOIN; NULL values in joins and aggregates](theory/01-relational-model-and-sql.md#inner-and-left-join-null-values-in-joins-and-aggregates)
- *Practice:* answer first questions about the reviews in SQL; average rating per store; join reviews with product prices and count products without a price → [workbook 06](workbooks/06-case-study-sql-first-questions.ipynb)

**1:00–1:45 · Advanced SQL, Python access and reproducible loading** ([theory page](theory/02-sql-from-python-and-ingestion.md))

- [Common table expressions and window functions](theory/02-sql-from-python-and-ingestion.md#common-table-expressions-and-window-functions)
- [Access from Python with SQLAlchemy and pandas](theory/02-sql-from-python-and-ingestion.md#access-from-python-with-sqlalchemy-and-pandas)
- [Loading data reproducibly with an ingestion script and constraints](theory/02-sql-from-python-and-ingestion.md#loading-data-reproducibly-with-an-ingestion-script-and-constraints)
- [Documenting a dataset (data card)](theory/02-sql-from-python-and-ingestion.md#documenting-a-dataset-data-card)
- *Practice:* case study: load reviews and products into PostgreSQL with the provided script; rank products by reviews per year with a window function; write a short data card → [workbook 10](workbooks/10-case-study-postgres-reviews.ipynb), [SQL files](workbooks/sql/), [data card template](workbooks/data-card-template.md)

**2:00–2:45 · Large tables in Python: Polars and choosing a tool** ([theory page](theory/03-polars-and-choosing-a-tool.md))

- [Limits of pandas: memory, single-threaded execution, eager evaluation](theory/03-polars-and-choosing-a-tool.md#limits-of-pandas-memory-single-threaded-execution-eager-evaluation)
- [Polars: expressions, lazy queries and the query optimiser, streaming, Parquet](theory/03-polars-and-choosing-a-tool.md#polars-expressions-lazy-queries-and-the-query-optimiser-streaming-parquet)
- [The same query in SQL, pandas and Polars](theory/03-polars-and-choosing-a-tool.md#the-same-query-in-sql-pandas-and-polars)
- [Choosing a tool: SQL database, pandas or Polars](theory/03-polars-and-choosing-a-tool.md#choosing-a-tool-sql-database-pandas-or-polars)
- *Practice:* case study: run the same aggregation on all training reviews in pandas and Polars and compare code, runtime and memory use → [workbook 14](workbooks/14-case-study-pandas-vs-polars.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-relational-model-and-sql.md](theory/01-relational-model-and-sql.md) | Relational model, PostgreSQL, basic queries, aggregation, joins and NULL | 1 | core |
| [theory/02-sql-from-python-and-ingestion.md](theory/02-sql-from-python-and-ingestion.md) | CTEs, window functions, SQLAlchemy, ingestion, data card | 2 | core |
| [theory/03-polars-and-choosing-a-tool.md](theory/03-polars-and-choosing-a-tool.md) | Limits of pandas, Polars, three tools side by side, choosing a tool | 3 | core |
| [workbooks/01-sql-select-where-order.ipynb](workbooks/01-sql-select-where-order.ipynb) | Data 100: SELECT, WHERE, ORDER BY, LIMIT with jupysql and DuckDB | 1 | core |
| [workbooks/02-sql-first-query.ipynb](workbooks/02-sql-first-query.ipynb) | Ploomber: a first query on bank marketing data | 1 | optional |
| [workbooks/03-sql-aggregate-functions.ipynb](workbooks/03-sql-aggregate-functions.ipynb) | Ploomber: aggregate functions, GROUP BY, HAVING | 1 | optional |
| [workbooks/04-sql-groupby-joins-ctes.ipynb](workbooks/04-sql-groupby-joins-ctes.ipynb) | Data 100: GROUP BY, CASE, joins and CTEs on IMDb data | 1–2 | core |
| [workbooks/05-sql-joins.ipynb](workbooks/05-sql-joins.ipynb) | Ploomber: INNER, LEFT, RIGHT and FULL joins | 1 | optional |
| [workbooks/06-case-study-sql-first-questions.ipynb](workbooks/06-case-study-sql-first-questions.ipynb) | **Case study**: first SQL questions on the reviews (DuckDB) | 1 | core |
| [workbooks/07-sql-window-functions.ipynb](workbooks/07-sql-window-functions.ipynb) | Ploomber: RANK, DENSE_RANK, moving averages, ROLLUP | 2 | optional |
| [workbooks/08-postgres-with-python.ipynb](workbooks/08-postgres-with-python.ipynb) | Ploomber: PostgreSQL in Docker with SQLAlchemy and jupysql | 2 | optional |
| [workbooks/09-jupysql-postgres-connect.ipynb](workbooks/09-jupysql-postgres-connect.ipynb) | JupySQL: connecting to PostgreSQL, loading and plotting | 2 | optional |
| [workbooks/10-case-study-postgres-reviews.ipynb](workbooks/10-case-study-postgres-reviews.ipynb) | **Case study**: load reviews into PostgreSQL with constraints, rank products per year, data card | 2 | core |
| [workbooks/sql/](workbooks/sql/) | `01-schema.sql`, `02-add-constraints.sql`, `03-rank-products-per-year.sql` | 2 | core |
| [workbooks/data-card-template.md](workbooks/data-card-template.md) | Data card template for the case study and the team project | 2 | core |
| [workbooks/11-polars-eda-with-sql-equivalents.ipynb](workbooks/11-polars-eda-with-sql-equivalents.ipynb) | Data 100: first steps in Polars, each with the equivalent SQL | 3 | core |
| [workbooks/12-polars-getting-started.ipynb](workbooks/12-polars-getting-started.ipynb) | Polars Cookbook ch. 1: DataFrame, LazyFrame, query plans, expressions | 3 | core |
| [workbooks/13-polars-transformations.ipynb](workbooks/13-polars-transformations.ipynb) | Polars Cookbook ch. 4: group_by, window functions with `over`, UDFs, SQL in Polars | 3 | optional |
| [workbooks/14-case-study-pandas-vs-polars.ipynb](workbooks/14-case-study-pandas-vs-polars.ipynb) | **Case study**: the same aggregation in SQL, pandas and Polars; runtime and memory | 3 | core |

Origins and licences of third-party files: [source.md](source.md).

## Before and after the session

**Preparation**

- Run the case-study script once so that `case-study/data/` exists (see [case-study/README.md](../../case-study/README.md)).
- Install [Docker Desktop](https://docs.docker.com/get-docker/) if you can, and check that `docker run hello-world` works. Without Docker, the case-study notebooks fall back to DuckDB.
- Work through the first half of [SQLBolt](https://sqlbolt.com/) (lessons 1–6), about 45 minutes.

**Team project until the next session.** Identify the project's data sources and load a first extract into the team database.

**Further reading (optional)**

- [PostgreSQL tutorial](https://www.postgresql.org/docs/current/tutorial.html), official documentation, Part I.
- [pandas: Comparison with SQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html).
- [Polars user guide: Coming from pandas](https://docs.pola.rs/user-guide/migration/pandas/).
- [PostgreSQL Exercises](https://pgexercises.com/) for joins, aggregation and window functions.

## Setup

Everything runs in the course environment from the repository root (`uv sync`, then `uv run jupyter lab`); it contains pandas, Polars, DuckDB, duckdb-engine, jupysql, SQLAlchemy and psycopg.

Extra packages for some third-party workbooks:

```bash
uv run --with gdown --with python-dotenv jupyter lab   # 04 downloads IMDb data with gdown; 08 reads a .env file
```

Graphviz (the `dot` program) is needed only for `show_graph()` in workbook 12 (`brew install graphviz` or https://graphviz.org/download/); `explain()` works without it.

**PostgreSQL.** One command starts a server in Docker:

```bash
docker run --name course-db -e POSTGRES_USER=course -e POSTGRES_PASSWORD=course \
    -e POSTGRES_DB=course -p 5432:5432 -d postgres:17
export DATABASE_URL=postgresql+psycopg://course:course@localhost:5432/course
```

If port 5432 is taken (for example by a local PostgreSQL installation), use `-p 5433:5432` and port 5433 in the URL. Workbooks 08 and 09 start their own containers named `postgres` with other credentials; workbook 09 ends by stopping and removing the containers it finds with `docker container ls --filter ancestor=postgres`, which can include your own course database; skip those cells if you want to keep it.

> [!TIP]
> No Docker, no PostgreSQL? Leave `DATABASE_URL` unset. Workbook 10 then uses DuckDB in a local file, and all SQL in workbooks 06, 10 and 14 runs unchanged.
