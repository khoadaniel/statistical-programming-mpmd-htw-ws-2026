# Session 1 · Data careers, the final project challenge, NumPy and pandas

> **Guiding question.** What do data professionals do, what will we build in this course, and how do we work with data in Python?

## 1 · Data careers, the course and the challenge

[Open notebook](workbooks/01-lecture-data-careers.ipynb)

- Data roles, their outputs and the skills they share
- What Berlin job advertisements ask for, by role, and how to read one
- Agentic engineering: how AI coding agents change data work and learning
- Course organisation and assessment
- The final project challenge: the customs heading of a product from its description

## 2 · Notebooks, a Python recap and NumPy

[Open notebook](workbooks/02-lecture-notebooks-python-and-numpy.ipynb)

- The working environment: uv, VS Code and the course environment
- Jupyter notebooks: cells, kernel, Markdown, order of execution
- A short recap of Python: types, collections, conditions, loops, functions, errors
- NumPy arrays: creation, `dtype`, `shape`, indexing, slicing and Boolean masks
- Vectorised operations, broadcasting and aggregation along an axis
- Random numbers and simulation; worked example: the Berlin booking calendar

## 3 · pandas: Series and DataFrames, from basic to advanced

[Open notebook](workbooks/03-lecture-pandas.ipynb)

- Series and DataFrame: index, columns and `dtype`; reading a file
- Selecting rows and columns with `loc`, `iloc` and Boolean masks; missing values
- Creating and transforming columns: arithmetic, `.str`, `.dt`, `map` and `apply`
- Sorting, `value_counts`, `groupby` with `agg`, and pivot tables
- Combining tables with `merge` and `concat`
- Method chaining, and tips for using AI coding assistants
- Worked example: five questions about the Berlin listings

## Team project until the next session

Teams of three are formed and read the project brief and the leaderboard rules.

## Setup

- The job advertisements of exercise 1.1 are handed out by the lecturer; they are not in the repository.
- To regenerate the figures: `uv run python sessions/01-careers-and-python/figures/make_figures.py`. To refresh the skills heat map with today's job ads, first run `uv run --with trafilatura python sessions/01-careers-and-python/figures/job_ad_skills.py` (about 250 page requests to the Data Berlin board).

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
