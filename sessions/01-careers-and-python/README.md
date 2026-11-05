# Session 1 · Data careers, the final project challenge, NumPy and pandas

> **Guiding question.** What do data professionals do, what will we build in this course, and how do we work with data in Python?

## 1 · Data careers, the course and the challenge

[Open notebook](workbooks/01-lecture-data-careers.ipynb)

- Seven data roles along the path from data to product, their outputs and the skills they share.
- What a data platform is, self-service, infrastructure and DataOps; analytics engineers between data engineering and analysis.
- What Berlin job advertisements ask for, by role, and how to read one.
- Agentic engineering: how AI coding agents change data work and learning.
- Course organisation and assessment.
- The final project challenge: the customs heading of a product from its description.

## 2 · Notebooks, a Python recap and NumPy

[Open notebook](workbooks/02-lecture-notebooks-python-and-numpy.ipynb)

- The working environment: uv, VS Code and the course environment.
- Jupyter notebooks: cells, kernel, Markdown, order of execution.
- Python fundamentals: variables, simple types, collections, conditions and loops, functions, errors, modules.
- NumPy arrays: creation, `dtype`, `shape`, indexing, slicing and Boolean masks.
- Vectorised operations, broadcasting and aggregation along an axis.
- Random numbers and simulation; worked example: the Berlin booking calendar.

## 3 · pandas: Series and DataFrames, from basic to advanced

[Open notebook](workbooks/03-lecture-pandas.ipynb)

- Series and DataFrame: index, columns and `dtype`; reading a file; questions to ask of every new table.
- Selecting rows and columns with `loc`, `iloc` and Boolean masks; missing values.
- Creating and transforming columns: arithmetic, `.str`, `.dt`, `map` and `apply`.
- Sorting, `value_counts`, `groupby` with `agg`, and pivot tables.
- Combining tables with `merge` and `concat`.
- Method chaining, and tips for using AI coding assistants.
- Worked example: five questions about the Berlin listings.

## 4 · Reshaping tables: crosstab, melt, pivot and pivot_table

[Open notebook](workbooks/04-lecture-reshaping-tables.ipynb)

- Long and wide tables, and which tool goes which way.
- `crosstab`: counts of category pairs, totals with `margins`, shares with `normalize`, other summaries.
- `melt`: column names become values; its four parameters.
- `pivot`: values become column names; the duplicates rule; the round trip with `melt`.
- `pivot_table`: pivot that aggregates; `margins`, `fill_value`, several statistics; `pivot_table` as a `groupby`.
- Worked example: room types across the Berlin districts.

## 5 · Plotting with pandas

[Open notebook](workbooks/05-lecture-plotting-with-pandas.ipynb)

- The generic plot call `data.plot.kind(options)` and what goes on each axis.
- Histograms and box plots by group, scatter plots with colour, a map from longitude and latitude.
- Bar charts of counts, stacked bars from a cross table, line charts with a rolling mean.
- Several panels: `subplots` and `layout`, shared bins, `DataFrame.hist`, the scatter matrix.
- Styling: title, labels, size, log scale, colours, the Axes after the call, saving a figure.
- Worked example: where are the expensive listings?

## Team project until the next session

Teams are formed and read the [final project brief](../17-18-final-presentations/final-project.md).

## Setup

- The case-study data: download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)); `ebti-v1.zip` is needed from Session 3 on.
- The job advertisements of exercise 1.1 are handed out by the lecturer; they are not in the repository.
- To regenerate the figures: `uv run python sessions/01-careers-and-python/figures/make_figures.py`. To refresh the skills heat map with today's job ads, first run `uv run --with trafilatura python sessions/01-careers-and-python/figures/job_ad_skills.py` (about 250 page requests to the Data Berlin board).

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
