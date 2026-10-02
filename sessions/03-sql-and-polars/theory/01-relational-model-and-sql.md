# The relational model and basic SQL

This page covers the first block of the session: how a relational database organises data in tables linked by keys, why the course uses PostgreSQL, and the core of SQL: selecting, filtering, sorting, aggregating and joining. Almost every organisation keeps its operational data in relational databases, and SQL is the language analysts use to get data out of them. In job advertisements for data roles, SQL is among the most frequently requested skills after Python. All examples use the tables of the course case study: `decisions` (one row per Binding Tariff Information decision, the training data 2017–2023) and `nomenclature` (one row per four-digit heading of the Harmonized System, with its English description, chapter and section).

```mermaid
flowchart LR
    Q["Business question"] --> S["SQL query"]
    S --> DB[("PostgreSQL<br/>decisions, nomenclature")]
    DB --> R["Small result table"]
    R --> P["pandas / Polars<br/>in Python"]
    P --> A["Chart, model, report"]
```

> [!NOTE]
> You can run every query on this page without a database server: DuckDB, an embedded database, reads the Parquet files of the case study directly (see [How it works in Python](#how-it-works-in-python)). PostgreSQL itself is set up in block 2.

## The relational model: tables, primary and foreign keys, relationships

### Concept

A **relational database** stores data in **tables** (also called relations). Each **row** is one record; each **column** has a name and a **data type** (text, integer, decimal number, date, true/false). The set of table definitions is the **schema**. The idea goes back to Edgar F. Codd (1970), who proposed that data should be described by tables and queried by their content, not by the way they are stored on disk.

A **primary key** is a column, or a combination of columns, whose value identifies each row uniquely and is never empty. In the case study, `bti_reference` identifies a decision and `heading` (the four-digit code, such as `9503` for toys) identifies a row of the nomenclature.

A **foreign key** is a column that refers to the primary key of another table. `decisions.heading` points to `nomenclature.heading`. This expresses a **relationship**: one heading has many decisions, each decision is classified under exactly one heading (a *one-to-many* relationship). The database can enforce it: a decision with a heading that does not exist in the nomenclature is rejected.

Storing each fact once, in one table, is called **normalisation**. The English description of heading 9503 is stored once in `nomenclature`, not repeated in each of its almost 9,000 decisions. If the description is corrected, one row changes, and no copy can contradict another.

A small example that you can follow by hand:

| nomenclature |  |  |
|---|---|---|
| **heading** (PK) | heading_description | chapter |
| 0102 | Bovine animals; live | 01 |
| 6404 | Footwear with textile uppers | 64 |
| 9503 | Toys | 95 |

| decisions |  |  |
|---|---|---|
| **bti_reference** (PK) | heading (FK) | keywords |
| DE-1 | 9503 | TOYS, PLUSH |
| DE-2 | 9503 | NULL |
| FR-1 | 6404 | SNEAKERS |

Heading 9503 has two decisions, 6404 one, 0102 none. `NULL` marks a value that is unknown: no keywords were recorded for DE-2. (The references and keywords here are invented; the real ones are longer.)

```mermaid
classDiagram
    direction LR
    class nomenclature {
        heading : text  PK
        heading_description : text
        chapter : text
        section : text
        section_name : text
    }
    class decisions {
        bti_reference : text  PK
        heading : text  FK
        issuing_country : text
        language : text
        start_date : date
        description : text
        keywords : text
    }
    nomenclature "1" --> "0..*" decisions : classifies
```

### Why it matters

Keys make the relationship between tables explicit, and the database uses them to guard data quality at the moment data are written: a heading with three digits or a decision for a heading that does not exist is rejected immediately instead of being discovered months later in an analysis. The database also builds an **index** (a sorted lookup structure) on every primary key, which makes lookups and joins fast. Without normalisation, the same fact is stored many times, and copies drift apart.

### How it works in Python

The case-study tables arrive as Parquet files. The following check tests the two key properties with pandas: the primary key is unique, and every foreign-key value has a partner.

```python
import pandas as pd

decisions = pd.read_parquet("case-study/data/train.parquet", columns=["bti_reference", "heading"])
nomenclature = pd.read_parquet("case-study/data/nomenclature.parquet")

print(decisions["bti_reference"].is_unique, nomenclature["heading"].is_unique)   # True True
# foreign key: does every decision refer to an existing heading?
known = decisions["heading"].isin(nomenclature["heading"])
print(known.all(), decisions.loc[~known, "heading"].value_counts().to_dict())  # False {'8803': 51}
# one-to-many: decisions per heading
print(decisions.groupby("heading").size().describe()[["mean", "50%", "max"]].round(1).to_dict())
# {'mean': 277.9, '50%': 44.0, 'max': 12852.0}
```

The foreign key fails for 51 decisions. They use heading 8803 (parts of aircraft), which the 2022 revision of the Harmonized System deleted; such parts now belong to the new heading 8807. The nomenclature table is the 2022 version, the decisions go back to 2017. A real foreign key would have rejected these rows, which is exactly the point: the database makes you decide what to do with them (keep an extra table of old headings, map 8803 to 8807, or drop the rows) instead of letting a join lose them silently.

The same rules written as SQL create the tables with **constraints**. `CHECK` adds a rule for the values; `REFERENCES` declares the foreign key.

```sql
CREATE TABLE tariff_headings (
    heading     TEXT PRIMARY KEY CHECK (length(heading) = 4),   -- one row per heading
    description TEXT NOT NULL
);
CREATE TABLE tariff_decisions (
    bti_reference TEXT PRIMARY KEY,
    heading       TEXT NOT NULL REFERENCES tariff_headings (heading),   -- foreign key
    keywords      TEXT                                                  -- NULL allowed
);
INSERT INTO tariff_headings VALUES ('0102', 'Bovine animals; live'), ('6404', 'Footwear with textile uppers'),
                                   ('9503', 'Toys');
INSERT INTO tariff_decisions VALUES ('DE-1', '9503', 'TOYS, PLUSH'), ('DE-2', '9503', NULL), ('FR-1', '6404', 'SNEAKERS');
INSERT INTO tariff_decisions VALUES ('NL-1', '8803', 'AIRCRAFT PARTS');   -- ERROR: violates foreign key constraint
INSERT INTO tariff_headings VALUES ('950', 'Toys?');                     -- ERROR: violates check constraint
```

### In practice

- **Stack Exchange Data Explorer** lets anyone query the public Stack Overflow database with SQL; questions, answers, users and votes are separate tables linked by identifiers such as `PostId` and `UserId`.
- **Wikimedia** offers public read-only copies of the Wikipedia databases through the Quarry service; pages, revisions and users are related tables, and volunteers answer research questions about Wikipedia with SQL.
- **The EU customs tariff** itself is relational: the Commission's TARIC database links each code to duty rates, measures and legal acts through code identifiers, and the EBTI database links each decision to its code.

> [!WARNING]
> A file is not a database. A Parquet or CSV file has no keys and no constraints: nothing stops a duplicated `bti_reference` or a heading that no longer exists. When you work with files, check the key properties yourself (as in the Python block above) and repeat the check after each update.

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
con.execute("CREATE VIEW decisions AS SELECT * FROM 'case-study/data/train.parquet'")
con.execute("CREATE VIEW nomenclature AS SELECT * FROM 'case-study/data/nomenclature.parquet'")
print(con.sql("SELECT COUNT(*) AS n FROM decisions").fetchone())   # (309529,)

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

By hand: in the small tables above, `SELECT bti_reference FROM decisions WHERE heading = '9503' ORDER BY bti_reference` keeps DE-1 and DE-2 and returns them in this order.

### Why it matters

Filtering in the database moves only the rows you need into Python. With millions of rows, `SELECT *` followed by filtering in pandas wastes memory and time. Writing the condition explicitly also makes the question precise: "textile decisions from the first COVID-19 year" becomes `chapter = '63' AND start_date BETWEEN '2020-01-01' AND '2020-12-31'`.

### How it works in Python

```sql
SELECT bti_reference, issuing_country, start_date, LEFT(keywords, 30) AS keywords
FROM decisions
WHERE heading = '9503' AND language = 'en'     -- both conditions must hold
ORDER BY start_date DESC, bti_reference        -- newest first; reference breaks ties
LIMIT 3;
-- XIBTI000000-2023-BTI157 | XI | 2023-11-22 | CARDS,EDUCATIONAL,EDUCATIONAL
-- XIBTI505051787          | XI | 2023-10-16 | EDUCATIONAL,FOR CHILDREN,FOR E
-- XIBTI505051885          | XI | 2023-10-16 | EDUCATIONAL,FOR CHILDREN,FOR E
```

The newest English toy decisions come from `XI`, the code for Northern Ireland: after Brexit, decisions of the United Kingdom (`GB`) are no longer EU decisions, but Northern Ireland still follows the EU customs rules for goods.

From Python with DuckDB, a query result becomes a pandas DataFrame with `.df()`:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW decisions AS SELECT * FROM 'case-study/data/train.parquet'")

print(con.sql("""
    SELECT COUNT(*) AS n FROM decisions
    WHERE chapter = '63' AND start_date BETWEEN '2020-01-01' AND '2020-12-31'
""").fetchone())                                                             # (1498,)
print(con.sql("SELECT COUNT(*) FROM decisions WHERE keywords ILIKE '%face mask%'").fetchone())   # (200,)
longest = con.sql("""
    SELECT bti_reference, language, LENGTH(description) AS n_chars
    FROM decisions ORDER BY n_chars DESC LIMIT 3
""").df()
print(longest)
#      bti_reference language  n_chars
# 0  DKBTI20-0944546       da     8621
# 1  DKBTI23-0238810       da     7134
# 2  DKBTI20-0944545       da     6403
```

### In practice

- **Customs officers and trade-compliance teams** filter the EBTI database by heading, keyword and date before classifying a new product; the public consultation page of the database offers search fields for exactly such conditions.
- The **European Medicines Agency's** EudraVigilance database of suspected side effects is queried by drug, period and outcome before any statistical signal detection is done.
- **Data engineers** check every nightly load with small queries such as `SELECT COUNT(*) FROM orders WHERE order_date = CURRENT_DATE - 1`.

> [!CAUTION]
> Without `ORDER BY`, the order of rows is not defined. `LIMIT 10` without `ORDER BY` returns *some* ten rows, which may change between runs or database versions.

## Aggregation with GROUP BY and HAVING

### Concept

An **aggregate function** reduces many values to one: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`. `GROUP BY` splits the rows into groups with equal values of the grouping columns and computes the aggregates per group; the result has one row per group. Every column in `SELECT` must either be a grouping column or be inside an aggregate.

`WHERE` filters **rows before** grouping; `HAVING` filters **groups after** aggregation, for example "only countries with at least 10,000 decisions".

A **share** is the average of a 0/1 variable. `AVG((language = 'en')::int)` is the share of decisions written in English, because `::int` turns true/false into 1/0.

By hand: grouping the three example decisions by `heading` gives 9503 with `COUNT(*) = 2` and `COUNT(keywords) = 1`, and 6404 with 1 and 1. `HAVING COUNT(*) >= 2` keeps only 9503.

### Why it matters

Almost every business question is an aggregate: decisions per month, per country, per heading; the share of a language per year. `HAVING` with a minimum count prevents tiny groups from topping a ranking by chance: a country with three decisions can have a share of 100 % for anything.

### How it works in Python

```sql
SELECT issuing_country, COUNT(*) AS n_decisions, COUNT(DISTINCT heading) AS n_headings
FROM decisions
GROUP BY issuing_country
HAVING COUNT(*) >= 10000          -- filter groups, not rows
ORDER BY n_decisions DESC;
-- DE | 172492 | 1004
-- FR |  48909 |  882
-- NL |  11933 |  540
-- GB |  11616 |  553
-- PL |  11259 |  571
```

Germany issues more than half of all decisions. This matters for every later model: a classifier trained on these data learns mostly from German descriptions.

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW decisions AS SELECT * FROM 'case-study/data/train.parquet'")
yearly = con.sql("""
    SELECT EXTRACT(YEAR FROM start_date)::int AS year,
           COUNT(*) AS n,
           ROUND(AVG((language = 'en')::int), 3) AS share_en,
           COUNT(*) FILTER (WHERE issuing_country = 'GB') AS n_gb
    FROM decisions
    WHERE start_date >= '2019-01-01'
    GROUP BY year
    ORDER BY year
""").df()
print(yearly)
#    year      n  share_en  n_gb
# 0  2019  48013     0.077  2885
# 1  2020  41697     0.076  2503
# 2  2021  40897     0.021     0
# 3  2022  39217     0.014     0
# 4  2023  43316     0.013     0
```

The share of English descriptions drops from 7.6 % to 2.1 % in 2021, when the United Kingdom left the EU customs union: a change in the data that has nothing to do with the products (Session 16 calls it drift).

### In practice

- **Eurostat and national statistical offices** publish trade statistics per product code, partner country and month, computed from individual customs declarations; each published cell is a `GROUP BY` result.
- **Statistical offices** such as Destatis publish counts and rates per region and year computed from individual-level registers.
- **Hospital comparisons** such as Care Compare of the US Centers for Medicare & Medicaid Services do not publish a measure when a hospital has too few cases, which is the idea of `HAVING COUNT(*) >= k`.

> [!WARNING]
> `COUNT(*)` counts rows; `COUNT(column)` counts non-NULL values of that column; `AVG(column)` silently ignores NULLs. On the decisions table, `COUNT(*)` is 309,529 but `COUNT(keywords)` is 308,256 and `COUNT(invalidation_reason)` only 45,223: most decisions simply expired after three years and have no invalidation reason.

## INNER and LEFT JOIN; NULL values in joins and aggregates

### Concept

A **join** combines rows of two tables whose key columns match. An **inner join** (`JOIN ... ON a.key = b.key`) returns one row per matching pair; rows without a partner disappear. Because each decision has exactly one heading, joining decisions with the nomenclature keeps one row per decision (if its heading exists) and adds the English description, chapter and section. `USING (heading)` is a short form when the key has the same name in both tables. Short **aliases** (`d`, `n`) say which table a column comes from.

A **left join** keeps every row of the left table; where no partner exists, the columns of the right table are `NULL`. Combined with `WHERE right.key IS NULL`, it finds rows **without** a partner, an **anti-join**: for example, headings for which no decision was ever issued.

![Inner and left join of a small nomenclature and decisions table](figures/join-types.png)

**NULL** means "unknown". Any comparison with NULL is unknown, not true or false, so `keywords = NULL` never matches; use `keywords IS NULL`. In `GROUP BY`, all NULLs form one group, which is why a count per `invalidation_reason` contains one large row with reason `NULL`.

By hand, with the tables above: `nomenclature JOIN decisions` gives 3 rows (DE-1, DE-2, FR-1); `nomenclature LEFT JOIN decisions` gives 4 rows, the fourth being heading 0102 with `bti_reference = NULL`.

### Why it matters

Normalised data must be joined before they can be analysed. The join type decides which rows survive and therefore which population a number describes. Many wrong numbers in reports come from an unintended inner join, which drops unmatched rows silently, or from a join on a key that is not unique, which duplicates rows.

### How it works in Python

```sql
-- decisions per section: each decision gets the columns of its heading
SELECT n.section, LEFT(n.section_name, 40) AS section_name, COUNT(*) AS n_decisions
FROM decisions AS d
JOIN nomenclature AS n ON n.heading = d.heading
GROUP BY n.section, n.section_name
ORDER BY n_decisions DESC
LIMIT 3;
-- XVI | Machinery and mechanical appliances; ele | 68648
-- XX  | Miscellaneous manufactured articles      | 32408
-- XV  | Base metals and articles of base metal   | 32266

-- anti-join: headings without any decision in 2017-2023
SELECT COUNT(*) AS headings_without_decisions
FROM nomenclature AS n
LEFT JOIN decisions AS d ON d.heading = n.heading
WHERE d.bti_reference IS NULL;
-- 116
```

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW decisions AS SELECT * FROM 'case-study/data/train.parquet'")
con.execute("CREATE VIEW nomenclature AS SELECT * FROM 'case-study/data/nomenclature.parquet'")

# which chapters have the most headings without any decision?
print(con.sql("""
    SELECT n.chapter, LEFT(MIN(n.chapter_description), 30) AS chapter_description,
           COUNT(*) AS headings_without_decisions
    FROM nomenclature AS n
    LEFT JOIN decisions AS d ON d.heading = n.heading
    WHERE d.bti_reference IS NULL
    GROUP BY n.chapter
    ORDER BY headings_without_decisions DESC, n.chapter
    LIMIT 3
""").df())
#   chapter             chapter_description  headings_without_decisions
# 0      26  Ores, slag and ash                                       15
# 1      51  Wool, fine or coarse animal ha                            9
# 2      28  Inorganic chemicals; organic a                            7

# the other direction: decisions whose heading is not in the nomenclature
print(con.sql("""
    SELECT d.heading, COUNT(*) AS n
    FROM decisions AS d
    LEFT JOIN nomenclature AS n ON n.heading = d.heading
    WHERE n.heading IS NULL
    GROUP BY d.heading
""").fetchall())                                                       # [('8803', 51)]
print(con.sql("SELECT COUNT(*) FROM decisions JOIN nomenclature USING (heading)").fetchone())
# (309478,): the inner join silently drops the 51 decisions of heading 8803
```

Raw materials such as ores and wool are rarely the subject of a decision: their classification is seldom in doubt. The inner join loses the 51 decisions of the deleted heading 8803 without any warning; only the count against the 309,529 rows of the table shows it.

### In practice

- **Marketing analysts** join transactions with customer attributes to compare customer segments; a customer without transactions disappears in an inner join and must be kept with a left join if "inactive customers" are part of the question.
- **Epidemiological record linkage**, for example linking cancer registries with mortality registers, keeps unmatched records and analyses them separately, because who fails to link is itself informative.
- **Data quality checks** in finance, retail and customs use anti-joins to find records that refer to a deleted customer, a discontinued product or, as here, a tariff code that no longer exists.

> [!CAUTION]
> A join on a key that is not unique on the "one" side multiplies rows. If `nomenclature` contained heading 9503 twice (for example once from HS 2017 and once from HS 2022), every toy decision would appear twice after the join and all counts would be inflated. Check uniqueness before joining (`COUNT(*)` against `COUNT(DISTINCT key)`), or in pandas use `merge(..., validate="many_to_one")`.

> [!WARNING]
> A condition on the right table in `WHERE` turns a left join back into an inner join: `nomenclature n LEFT JOIN decisions d ... WHERE d.language = 'de'` drops all headings without decisions, because `d.language` is NULL there. Put such conditions into the `ON` clause if unmatched rows must stay.

## Check your understanding

1. Which column is the primary key of `decisions`, and which column is a foreign key? What would the database do with a decision whose heading does not exist in `nomenclature`?
2. Explain the difference between `WHERE` and `HAVING` with an example from the decision data.
3. Why does `SELECT COUNT(*) FROM decisions WHERE keywords = NULL` return 0?
4. A colleague joins `decisions` with `nomenclature` using an inner join and reports the number of decisions per section. Which decisions are missing from the result, and does it matter for her question?
5. Which logical step of a query is evaluated first: `SELECT` or `WHERE`? What follows for column aliases?

## Further reading

- Codd, E. F. (1970). A relational model of data for large shared data banks. *Communications of the ACM*, 13(6), 377–387. https://doi.org/10.1145/362384.362685
- The PostgreSQL Global Development Group. *PostgreSQL documentation: Tutorial* (Part I, "The SQL Language"). https://www.postgresql.org/docs/current/tutorial.html
- Software Carpentry. *Databases and SQL*. https://swcarpentry.github.io/sql-novice-survey/
- DuckDB Foundation. *SQL introduction* (DuckDB documentation). https://duckdb.org/docs/stable/sql/introduction
