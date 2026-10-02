# Session 1 · Introduction: data science careers, Python for analysis and Python for software engineering

> [!NOTE]
> **Guiding question.** What do data professionals do, and how does Python for analysis differ from Python for building applications?

**Learning outcomes.** After this session you are able to

- describe the main data roles, their tasks and the skills they require
- set up a reproducible Python environment and analyse tabular data in a notebook
- explain the difference between analysis code in notebooks and application code in modules and classes

## Session plan

**0:00–0:45 · Data careers and the course** ([theory](theory/01-data-careers.md))

- [Data roles: data analyst, data scientist, ML and AI engineer, data engineer](theory/01-data-careers.md#data-roles)
- [Tasks, skills and entry routes of each role](theory/01-data-careers.md#tasks-skills-and-entry-routes)
- [Course organisation, assessment, the running case study and the leaderboard](theory/01-data-careers.md#course-organisation-assessment-case-study-and-leaderboard)
- *Practice:* [compare three job advertisements](theory/01-data-careers.md#practice-compare-three-job-advertisements) (analyst, data scientist, ML engineer) and list the skills they share and those that differ

**1:00–1:45 · Python for analysis** ([theory](theory/02-python-for-analysis.md))

- [Python for analysis and its working environment (uv, Jupyter, VS Code)](theory/02-python-for-analysis.md#python-for-analysis-and-its-working-environment)
- [A refresher of the fundamentals (types, lists and dictionaries, conditions, loops, functions)](theory/02-python-for-analysis.md#a-refresher-of-the-fundamentals)
- [Tabular data with pandas in a notebook](theory/02-python-for-analysis.md#tabular-data-with-pandas-in-a-notebook)
- [Tips for using AI coding assistants](theory/02-python-for-analysis.md#tips-for-using-ai-coding-assistants)
- *Practice:* case study: download the Berlin Airbnb listings with the provided script and answer five questions about them in a notebook (how many listings, where, of what type, at what price, with how many reviews): [workbooks/12-case-study-five-questions.ipynb](workbooks/12-case-study-five-questions.ipynb)

**2:00–2:45 · From notebook to application** ([theory](theory/03-notebook-to-application.md))

- [Python for software engineering: from notebook to application](theory/03-notebook-to-application.md#python-for-software-engineering-from-notebook-to-application)
- [Scripts, modules and packages](theory/03-notebook-to-application.md#scripts-modules-and-packages)
- [Project structure](theory/03-notebook-to-application.md#project-structure)
- [A first introduction to object-oriented programming (classes, objects, attributes, methods)](theory/03-notebook-to-application.md#a-first-introduction-to-object-oriented-programming)
- [When to use a notebook and when an application](theory/03-notebook-to-application.md#when-to-use-a-notebook-and-when-an-application)
- *Practice:* turn the notebook analysis into a module with a small class that loads and summarises the listings, in the [workspace](workspace/README.md)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-data-careers.md](theory/01-data-careers.md) | Data roles, skills, entry routes; course organisation, assessment, datasets, leaderboard | 1 | core |
| [theory/02-python-for-analysis.md](theory/02-python-for-analysis.md) | uv, Jupyter, VS Code; Python refresher; pandas; AI coding assistants | 2 | core |
| [theory/03-notebook-to-application.md](theory/03-notebook-to-application.md) | Scripts, modules, packages; src layout; first classes; notebook or application | 3 | core |
| [workbooks/01-jupyter-introduction.ipynb](workbooks/01-jupyter-introduction.ipynb) | How Jupyter notebooks work (*Think Python*) | 2 | core |
| [workbooks/02-python-variables.ipynb](workbooks/02-python-variables.ipynb) | Variables, dynamic typing, object references | 2 | optional (refresher) |
| [workbooks/03-python-scalar-types.ipynb](workbooks/03-python-scalar-types.ipynb) | int, float, str, bool, None | 2 | optional (refresher) |
| [workbooks/04-python-data-structures.ipynb](workbooks/04-python-data-structures.ipynb) | Lists, tuples, dictionaries, sets | 2 | core |
| [workbooks/05-python-control-flow.ipynb](workbooks/05-python-control-flow.ipynb) | if/elif/else, for and while loops | 2 | optional (refresher) |
| [workbooks/06-python-functions.ipynb](workbooks/06-python-functions.ipynb) | Defining functions, arguments, lambda | 2 | core |
| [workbooks/07-python-errors-and-exceptions.ipynb](workbooks/07-python-errors-and-exceptions.ipynb) | Errors, try/except, raising exceptions | 2 | optional (prepares S2) |
| [workbooks/08-pandas-introduction.ipynb](workbooks/08-pandas-introduction.ipynb) | Series, DataFrame, selection, filtering, grouping | 2 | core |
| [workbooks/09-pandas-indexing-and-selection.ipynb](workbooks/09-pandas-indexing-and-selection.ipynb) | loc, iloc, boolean masks | 2 | core |
| [workbooks/10-pandas-merge-and-join.ipynb](workbooks/10-pandas-merge-and-join.ipynb) | Joining tables with `pd.merge` | 2 | optional (prepares S3) |
| [workbooks/11-pandas-groupby.ipynb](workbooks/11-pandas-groupby.ipynb) | Aggregation, split-apply-combine | 2 | core |
| [workbooks/12-case-study-five-questions.ipynb](workbooks/12-case-study-five-questions.ipynb) | Case study: five questions about the Berlin Airbnb listings (own) | 2 | core (practice) |
| [workbooks/13-python-modules-and-packages.ipynb](workbooks/13-python-modules-and-packages.ipynb) | import forms, standard library, third-party packages | 3 | core |
| [workbooks/14-classes-and-functions.ipynb](workbooks/14-classes-and-functions.ipynb) | Programmer-defined types, attributes, objects (*Think Python*) | 3 | core |
| [workbooks/15-classes-and-methods.ipynb](workbooks/15-classes-and-methods.ipynb) | Methods, `__init__`, `__str__`, operator overloading (*Think Python*) | 3 | optional |
| [workspace/](workspace/README.md) | Package `listingtools` with the class `ListingTable`, tests and exercises | 3 | core (practice) |

Sources and licences: [source.md](source.md).

## Before and after the session

**Before.** Install [uv](https://docs.astral.sh/uv/getting-started/installation/), [VS Code](https://code.visualstudio.com/) and [Git](https://git-scm.com/downloads). Clone the course repository and generate the case-study data of Sessions 1–12 once from the repository root (downloads the Berlin snapshot of Inside Airbnb, about 100 MB, and the Berlin weather from Open-Meteo; writes about 6 MB of Parquet files to `case-study/data/airbnb/`):

```bash
uv run python case-study/prepare_airbnb.py
```

The EBTI customs data of Sessions 13–16 are prepared later with `case-study/prepare_data.py`.

If Python is new to you after the summer, work through workbooks 02–06.

**Team project until the next session.** Teams of three are formed and shortlist three project topics.

**After.** Repeat questions 4 and 5 for the district you live in or know best and compare it with the city as a whole, and finish the workspace exercises.

**Further reading (optional).**

- [The Python Tutorial](https://docs.python.org/3/tutorial/), sections 3–5 and 9 (Python Software Foundation)
- [Python for Data Analysis, 3rd edition](https://wesmckinney.com/book/), chapters 2–5 (McKinney, open access)
- [Think Python, 3rd edition](https://allendowney.github.io/ThinkPython/), chapters 14–15 (Downey, free online)
- [Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510) (Wilson et al. 2017)

## Setup

The workbooks need `pandas`, `pyarrow` and `jupyterlab`, all in the course environment. Without it, start JupyterLab from the repository root with:

```bash
uv run --with pandas --with pyarrow --with jupyterlab jupyter lab
```

The *Think Python* workbooks (01, 14, 15) download a small helper file (`thinkpython.py`, `diagram.py`) on first run and need an internet connection; the diagrams in them use `matplotlib`. The workspace is its own uv project:

```bash
cd sessions/01-careers-and-python/workspace
uv run pytest -q
```
