# Large tables in Python: Polars and choosing a tool

This page covers the third block. pandas is the default table library in Python and works well for most course data, but it reaches limits when tables grow: it holds everything in memory, uses one processor core for most operations and executes every step immediately. **Polars** is a newer DataFrame library that addresses these limits with expressions, lazy queries, a query optimiser and a streaming engine. We write the same query in SQL, pandas and Polars, and end with a guide for choosing between a database, pandas and Polars. Distributed systems such as Spark belong to the module on big-data technology and are only mentioned here.

```mermaid
flowchart LR
    subgraph Eager["pandas: eager"]
        A1["read all columns"] --> A2["new columns"] --> A3["groupby"] --> A4["filter groups"]
    end
    subgraph Lazy["Polars: lazy"]
        B1["scan: build a plan"] --> B2["optimise plan"] --> B3["collect: run in parallel"]
    end
```

## Limits of pandas: memory, single-threaded execution, eager evaluation

### Concept

Three properties of pandas matter when data grow:

1. **Memory.** A DataFrame lives completely in main memory (RAM). Intermediate results, such as the table after a `merge` or a new column, are additional copies. A common rule of thumb from the pandas author Wes McKinney was that pandas needed 5 to 10 times as much RAM as the size of the dataset. Text columns are especially costly: before pandas 3.0, each string was a separate Python object.
2. **Single-threaded execution.** Most pandas operations use one processor core, even when the laptop has 8 or 16.
3. **Eager evaluation.** Each line is executed immediately and completely. When you read a file with 14 columns and later use 3, pandas has already read all 14, unless you said so with `columns=`. pandas cannot look ahead at what you will do with the result.

A worked example with the case-study data: `train.parquet` is 147 MB on disk (compressed). In memory the 309,529 decisions take about 560 MB in pandas 2.3, almost four times the file size, because four of the 14 columns are long texts. The raw export of the European Commission, 1,051,034 decisions in CSV files of 1.2 GB, takes about 2.1 GB as a pandas DataFrame, and the aggregation of [workbook 14](../workbooks/14-case-study-pandas-vs-polars.ipynb) reached a peak of about 2.3 GB above the starting point of the Python process.

### Why it matters

The limits rarely matter for tens of thousands of rows. They matter when a team project uses several years of data, many text columns or repeated experiments: the notebook becomes slow, the kernel crashes with an out-of-memory error, and people start to sample the data for technical rather than statistical reasons.

### How it works in Python

```python
import pandas as pd

decisions = pd.read_parquet("case-study/data/train.parquet")
print(f"{decisions.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")     # 557 MB in memory
print(decisions.memory_usage(deep=True).sort_values(ascending=False).head(3).div(1e6).round(0))
# description                     257.0   <- the free-text columns dominate
# classification_justification    125.0
# keywords                         39.0   (exact values depend on the pandas version)

small = pd.read_parquet("case-study/data/train.parquet", columns=["issuing_country", "heading", "start_date"])
print(f"{small.memory_usage(deep=True).sum() / 1e6:.0f} MB with three columns")  # 35 MB with three columns
```

### In practice

- **pandas 3.0** (January 2026) stores text in a dedicated string type backed by Apache Arrow by default and made copy-on-write the only mode, both to reduce memory use and hidden copies.
- **Wes McKinney**, the creator of pandas, described these limits in his 2017 essay *Apache Arrow and the "10 Things I Hate About pandas"*, which motivated the Apache Arrow project on which Polars and DuckDB build.
- The **H2O.ai database-like ops benchmark** (continued by DuckDB Labs since 2023) compares group-by and join speed of DataFrame tools on tables from 10 million to 1 billion rows; pandas fails on the largest sizes on a single machine because of memory.

> [!TIP]
> The cheapest optimisation in pandas is to read only the columns you need (`columns=[...]`) and to use Parquet instead of CSV. Try it before switching tools.

## Polars: expressions, lazy queries and the query optimiser, streaming, Parquet

### Concept

**Polars** is a DataFrame library written in Rust with a Python interface (first released in 2020 by Ritchie Vink). It stores data in the **Apache Arrow** columnar format: each column is a contiguous block of memory, which is fast to scan and easy to share with other tools.

**Expressions.** In Polars you describe *what* to compute with expressions such as `pl.col("heading").n_unique()`, and pass them to methods such as `select`, `with_columns`, `filter`, `group_by(...).agg(...)`. An expression is a recipe, not a result; Polars can run many expressions in parallel on several cores.

| Task | pandas | Polars |
|---|---|---|
| choose columns | `df[["a", "b"]]` | `df.select("a", "b")` |
| new column | `df.assign(c=df["a"] * 2)` | `df.with_columns(c=pl.col("a") * 2)` |
| filter rows | `df[df["a"] > 3]` or `df.query("a > 3")` | `df.filter(pl.col("a") > 3)` |
| group | `df.groupby("g").agg(m=("a", "mean"))` | `df.group_by("g").agg(m=pl.col("a").mean())` |
| row index | yes, central | none (no index) |

**Lazy queries and the optimiser.** `pl.scan_parquet(path)` returns a **LazyFrame**: nothing is read yet. Each method call adds a step to a **query plan**. On `.collect()`, the **query optimiser** rewrites the plan before running it, for example:

- **projection pushdown**: read only the columns the query uses;
- **predicate pushdown**: apply filters while reading, so that rows that are not needed are never loaded;
- **common subexpression elimination**: compute a repeated expression once.

`lf.explain()` prints the optimised plan. This is the same idea that a SQL database uses: you describe the result, the system chooses the steps.

```mermaid
flowchart TB
    Q["Your query:<br/>scan, filter, group_by, agg"] --> P["Logical plan"]
    P --> O{"Optimiser"}
    O -->|"projection pushdown"| C["read 4 of 14 columns"]
    O -->|"predicate pushdown"| R["skip rows while reading"]
    C --> X["Physical plan<br/>(parallel)"]
    R --> X
    X --> E{"Engine"}
    E -->|"in-memory (default)"| M["Result"]
    E -->|"streaming"| S["Batches: data larger than RAM"]
    S --> M
```

**Streaming.** With `collect(engine="streaming")`, Polars processes the data in **batches** instead of loading the whole table, so queries can run on files larger than the available memory, as long as the *result* fits. `sink_parquet` writes a streaming result straight to a file.

**Parquet.** Parquet is a **columnar** file format: values of one column are stored together, compressed, with statistics (minimum, maximum) per block of rows. A reader can therefore read a few columns or skip blocks without reading the whole file. CSV, by contrast, must be parsed in full and carries no data types. Parquet was created in 2013 by engineers at Twitter and Cloudera and is now an Apache project.

### Why it matters

Lazy evaluation lets the library do what an experienced pandas user does by hand (read fewer columns, filter early) automatically and consistently. Parallel execution uses the cores you already have. Streaming removes the hard memory ceiling for many aggregations, without moving to a cluster.

### How it works in Python

```python
import polars as pl

lf = pl.scan_parquet("case-study/data/train.parquet")        # LazyFrame: nothing read yet
query = (
    lf.filter(pl.col("language") == "de")
      .group_by(pl.col("start_date").dt.year().alias("year"))
      .agg(n=pl.len(),
           n_headings=pl.col("heading").n_unique(),
           median_chars=pl.col("description").str.len_chars().median())
      .sort("year")
)
print(query.explain())          # optimised plan: note "PROJECT 4/14 COLUMNS" and the pushed-down SELECTION
result = query.collect()        # now the work is done, on all cores
print(result.tail(3))
# ┌──────┬───────┬────────────┬──────────────┐
# │ year ┆ n     ┆ n_headings ┆ median_chars │
# │ 2021 ┆ 23849 ┆ 743        ┆ 750.0        │
# │ 2022 ┆ 22953 ┆ 721        ┆ 766.0        │
# │ 2023 ┆ 25851 ┆ 764        ┆ 759.0        │
# └──────┴───────┴────────────┴──────────────┘

streamed = query.collect(engine="streaming")   # batch-wise; same result
print(streamed.equals(result))                 # True
```

Expressions compose. Several aggregations per group, a conditional column and a window expression (`.over()`, the Polars form of `OVER (PARTITION BY ...)`):

```python
import polars as pl

decisions = pl.read_parquet("case-study/data/train_sample.parquet")
out = (
    decisions.with_columns(
        n_words=pl.col("description").str.split(" ").list.len(),
        has_code=pl.when(pl.col("description").str.contains("<CODE>", literal=True))
                   .then(pl.lit("yes")).otherwise(pl.lit("no")),
        heading_size=pl.len().over("heading"),          # window expression: decisions per heading
    )
    .filter(pl.col("language").is_in(["de", "fr"]))
    .group_by("language", "has_code")
    .agg(n=pl.len(), median_words=pl.col("n_words").median(),
         median_heading_size=pl.col("heading_size").median())
    .sort("language", "has_code")
)
print(out)
# 4 rows: 3,231 German descriptions quote their own code (<CODE>), but only 3 French ones;
# French descriptions are about half as long in words (median 43 against 95)
```

### In practice

- **Berkeley's Data 100** course (University of California, Berkeley) switched its teaching from pandas to Polars in autumn 2026; workbook 11 is from its course notes.
- **Polars' published benchmarks** use the TPC-H queries of the Transaction Processing Performance Council, a standard benchmark for analytical databases, to compare Polars, pandas, DuckDB and other tools on the same hardware.
- **Parquet** is the default storage format of data-lake systems such as Delta Lake and Apache Iceberg, and of the data published by the Hugging Face Hub, which converts uploaded datasets to Parquet automatically.

> [!WARNING]
> Python functions inside Polars (`map_elements`, `map_batches` with a lambda) run row by row in Python, on one core, and the optimiser cannot see inside them. They are as slow as `apply` in pandas. Look for a built-in expression first (`str.*`, `dt.*`, `list.*`, `when/then`).

> [!CAUTION]
> Polars and pandas differ in details that change results: Polars has no index, `group_by` does not keep the group order unless `maintain_order=True`, missing values are `null` (not `NaN`), and empty text fields may be read as `""` by one tool and as missing by another (see the `INVALIDATION_REASON` example in workbook 14). Compare results when you port code.

## The same query in SQL, pandas and Polars

### Concept

SQL, pandas and Polars describe the same operations with different words. Knowing the correspondence lets you move a step to the tool where it fits best and check one result against another.

| Operation | SQL | pandas | Polars |
|---|---|---|---|
| read | `FROM decisions` | `pd.read_parquet` | `pl.scan_parquet` |
| filter rows | `WHERE` | boolean mask, `query` | `filter` |
| join | `LEFT JOIN ... USING` | `merge(how="left")` | `join(how="left")` |
| group and aggregate | `GROUP BY` + `AVG(...)` | `groupby(...).agg(...)` | `group_by(...).agg(...)` |
| filter groups | `HAVING` | filter after `agg` | `filter` after `agg` |
| window | `RANK() OVER (PARTITION BY g)` | `groupby(g)[c].rank()` | `pl.col(c).rank().over(g)` |
| NULL group | kept | dropped unless `dropna=False` | kept |

### Why it matters

Teams rarely use one tool only. Data are extracted with SQL, prepared in pandas or Polars, and the final numbers are reconciled against the database. Translating between the three is a daily task, and differences in NULL handling or ordering are a common source of disagreeing numbers.

### How it works in Python

The question: per issuing country and start year, the number of decisions and the average length of the description, for country-years with at least 500 decisions.

```python
import duckdb
import pandas as pd
import polars as pl

TRAIN = "case-study/data/train.parquet"

# 1. SQL (DuckDB on the file)
sql = duckdb.sql(f"""
    SELECT issuing_country, EXTRACT(YEAR FROM start_date)::int AS year,
           COUNT(*) AS n, AVG(length(description)) AS avg_chars
    FROM '{TRAIN}'
    GROUP BY issuing_country, year
    HAVING COUNT(*) >= 500
""").pl()

# 2. pandas
r = pd.read_parquet(TRAIN, columns=["issuing_country", "start_date", "description"])
pdf = (r.assign(year=r["start_date"].dt.year, n_chars=r["description"].str.len())
        .groupby(["issuing_country", "year"], as_index=False)
        .agg(n=("n_chars", "size"), avg_chars=("n_chars", "mean"))
        .query("n >= 500"))

# 3. Polars (lazy)
plf = (pl.scan_parquet(TRAIN)
         .group_by("issuing_country", pl.col("start_date").dt.year().alias("year"))
         .agg(n=pl.len(), avg_chars=pl.col("description").str.len_chars().mean())
         .filter(pl.col("n") >= 500)
         .collect())

print(len(sql), len(pdf), len(plf))                      # 74 74 74
print(sorted(sql["n"].to_list()) == sorted(plf["n"].to_list()) == sorted(pdf["n"].tolist()))  # True
print(plf.filter(pl.col("issuing_country") == "GB").sort("year"))
# GB appears for 2017-2020 only: after Brexit, British decisions are no longer EU decisions
```

The case-study notebook runs a larger version of this query, with distinct headings and the share of invalidated decisions, on the **raw export**: 1,051,034 decisions in 23 CSV files of together 1.2 GB ([workbook 14](../workbooks/14-case-study-pandas-vs-polars.ipynb)).

![Runtime and peak memory of the same aggregation on the raw EBTI export in DuckDB, pandas and Polars](figures/pandas-polars-benchmark.png)

On the course team's laptop (Apple silicon), pandas needed about 9.5 seconds, mostly to parse all 15 text columns; Polars in lazy mode needed about 0.2 seconds, a factor of about 50, because it parses only the 5 columns the query uses and works on all cores. DuckDB took about 0.4 seconds although its CSV reader had to run on one thread (the descriptions contain line breaks inside quoted fields). Memory told a different story: the peak resident memory was about 2.1 to 2.4 GB above the baseline for pandas *and* for every Polars version, and about 1.3 GB for DuckDB. Lazy and streaming execution did not lower the peak here; one likely reason is that Polars maps the CSV files into memory, and mapped pages count as resident memory. The absolute numbers will differ on your machine; the honest summary is "much faster, not smaller".

### In practice

- **Data teams** often prototype a transformation in a notebook (pandas or Polars) and move it into SQL in the warehouse once it is stable, so that it runs next to the data.
- The **pandas documentation** maintains a page "Comparison with SQL", and the **Polars user guide** has migration pages for pandas and SQL users, both of which are standard references for translating code.
- **Reconciliation checks** in finance compare totals computed in the source database with totals computed in the analysis code before a report is released.

> [!WARNING]
> Benchmarks measure one query on one machine with one version of each library. Do not generalise from a single timing; repeat runs, compare ratios, and check that all versions return the same result before comparing speed.

## Choosing a tool: SQL database, pandas or Polars

### Concept

The choice depends on where the data live, how large they are, who else needs them and what happens next.

```mermaid
flowchart TD
    A{"Do the data live in a database<br/>or must a team share them?"} -->|yes| SQL["Filter, join and aggregate in SQL<br/>(PostgreSQL)"]
    A -->|no, files| B{"Fits comfortably in RAM?<br/>(file size x 5 &lt; free RAM)"}
    SQL --> C{"Result small?"}
    C -->|yes| B
    B -->|"yes, < ~1 million rows"| D{"Next step needs pandas?<br/>(statsmodels, plotting, older code)"}
    D -->|yes| PD["pandas"]
    D -->|no| PL["Polars or pandas"]
    B -->|"no, or many millions of rows"| PL2["Polars lazy / streaming,<br/>or DuckDB on the files"]
    PL2 --> BIG{"Still too large for one machine?"}
    BIG -->|yes| SP["Distributed tools (Spark):<br/>module 3.2"]
```

Rules of thumb:

- **SQL database**: the data are shared, updated, or larger than a laptop's memory; the work is filtering, joining and aggregating; constraints should protect the data. Bring only the result to Python.
- **pandas**: the data fit easily in memory; the next step uses libraries that expect pandas (statsmodels formulas, seaborn, many tutorials); interactive exploration.
- **Polars**: the data are large for pandas, the pipeline is long, speed matters, or the data are many Parquet files; also a clean expression syntax for new code. Convert with `pl.from_pandas` and `.to_pandas()` at the boundary.
- **DuckDB**: SQL on local files without a server; fast for analytical queries; returns pandas or Polars.

scikit-learn accepts both pandas and Polars DataFrames as input, and since version 1.4 can return Polars output with `set_output(transform="polars")`.

### Why it matters

Choosing the wrong tool costs time in both directions: a team that loads 5 GB of CSV into pandas on every run waits minutes and crashes kernels; a team that builds a database for a 2 MB spreadsheet spends a week on infrastructure. Mixing tools at clear boundaries is normal and good practice.

### How it works in Python

Moving between the tools is cheap because they share the Arrow format:

```python
import duckdb
import polars as pl

lf = pl.scan_parquet("case-study/data/train.parquet")
by_year = lf.group_by(pl.col("start_date").dt.year().alias("year")).agg(n=pl.len()).sort("year").collect()

pdf = by_year.to_pandas()          # Polars -> pandas (for statsmodels, seaborn, ...)
back = pl.from_pandas(pdf)         # pandas -> Polars
print(duckdb.sql("SELECT MAX(n) AS peak FROM by_year").fetchone())   # (51482,): DuckDB reads the Polars frame
```

### In practice

- **Berkeley Data 100** teaches Polars and SQL side by side, writing each step in both (workbook 11).
- **Analytics teams** at many companies keep the warehouse in SQL (for example with dbt) and use Python only for modelling and charts, so that numbers in dashboards and analyses come from one source.
- **Research groups** that publish large datasets increasingly use Parquet and recommend DuckDB or Polars for local analysis, for example the Hugging Face Hub, which offers a DuckDB query interface for every dataset.

> [!IMPORTANT]
> For this course, all tools are acceptable in the team project. What counts is that the choice is deliberate and written down: where the data live, which tool does which step, and why.

## Check your understanding

1. Name the three limits of pandas discussed on this page and one way to reduce each within pandas.
2. What is the difference between `pl.read_parquet` and `pl.scan_parquet`? When does Polars read the data in each case?
3. What do projection pushdown and predicate pushdown do? Find both in the output of `explain()` for a query of your own.
4. In workbook 14, the rule `INVALIDATION_REASON IS NOT NULL` gives 100 % invalidated decisions in Polars but plausible shares in pandas and DuckDB. Why, and how do you write a rule that all three tools evaluate the same way?
5. Your project data are 30 CSV files of 2 GB each from a public portal, updated monthly. Sketch a tool chain and justify each choice.

## Further reading

- Polars developers. *Polars user guide*: "Lazy API", "Streaming" and "Coming from pandas". https://docs.pola.rs/user-guide/
- Heavey, K. *Modern Polars*: a side-by-side comparison of pandas and Polars (CC-BY-4.0). https://kevinheavey.github.io/modern-polars/
- McKinney, W. (2017). *Apache Arrow and the "10 Things I Hate About pandas"*. https://wesmckinney.com/blog/apache-arrow-pandas-internals/
- Raasveldt, M., & Mühleisen, H. (2019). DuckDB: an embeddable analytical database. *Proceedings of SIGMOD 2019*, 1981–1984. https://doi.org/10.1145/3299869.3320212
