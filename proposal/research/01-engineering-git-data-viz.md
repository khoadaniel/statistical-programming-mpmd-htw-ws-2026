# Research notes — Python engineering, Git/GitHub, databases, visualisation, peer courses

Links checked on 2026-09-24 (HTTP 200; final URLs after redirects).

## Findings that change the plan
- **GitHub Classroom was retired on 28 Aug 2026** ([announcement](https://github.com/orgs/community/discussions/205975)). Plan around **Classroom 50** (https://classroom50.org/, free, open source, autograding via Actions) or plain GitHub org + template repos.
- **pandas 3.0 (21 Jan 2026)**: Copy-on-Write is now the only mode, and strings get a dedicated `str` dtype (https://pandas.pydata.org/docs/whatsnew/v3.0.0.html). McKinney 3e is still useful but predates this.
- **Berkeley Data 100 (Fall 2026) now teaches with Polars.** This supports teaching Polars + DuckDB next to pandas.
- **MIT Missing Semester 2026** has lectures on *Agentic Coding* and *Code Quality*. This is the basis for an explicit AI-assisted coding policy.

## 1. Professional Python engineering and AI-assisted coding
| Title | URL | Notes |
|---|---|---|
| uv docs | https://docs.astral.sh/uv/ · projects guide https://docs.astral.sh/uv/guides/projects/ | Standard project setup; GH Actions + pre-commit integration guides |
| Ruff | https://docs.astral.sh/ruff/ | Lint + format. New type checkers: ty (https://docs.astral.sh/ty/), Pyrefly (beta) |
| mypy / Pyright / typing docs | https://mypy.readthedocs.io/en/stable/ · https://typing.python.org/en/latest/ | |
| pytest / pydantic / logging HOWTO | https://docs.pytest.org/en/stable/getting-started.html · https://docs.pydantic.dev/latest/ · https://docs.python.org/3/howto/logging.html | Lab handouts |
| Scientific Python Development Guide | https://learn.scientific-python.org/development/ | Most current opinionated guide (pyproject, ruff, pre-commit, CI) |
| PyPA Packaging tutorial | https://packaging.python.org/en/latest/tutorials/packaging-projects/ | |
| Python Packages (py-pkgs) | https://py-pkgs.org/ | Full lifecycle. Uses Poetry, not uv |
| Carpentries Incubator — Intermediate Research Software Development | https://carpentries-incubator.github.io/python-intermediate-development/ | One running project: envs, linting, pytest, CI, code review, packaging. Closest to our scope |
| Good enough practices in scientific computing | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510 | Week-1 reading |
| Research Software Engineering with Python | https://third-bit.com/py-rse/ | Exercises per chapter. 2021/22, conda-based |
| Better Code, Better Science (Poldrack) | https://bettercode-book.org/ | New; includes a "Coding with AI" chapter |
| Ten Simple Rules for AI-Assisted Coding in Science (2025) | https://arxiv.org/abs/2510.22254 | Basis for the course AI policy |
| MIT Missing Semester 2026 — Agentic Coding / Code Quality | https://missing.csail.mit.edu/2026/agentic-coding/ · https://missing.csail.mit.edu/2026/code-quality/ | 6 exercises (Claude Code, Codex, opencode) |
| CS50 AI policy / duck | https://cs50.harvard.edu/x/notes/ai/ · https://dl.acm.org/doi/10.1145/3626252.3630938 | Model course AI policy |
| marimo / Jupytext | https://docs.marimo.io/ · https://jupytext.readthedocs.io/ | Notebooks git can diff |

## 2. Git & GitHub
| Title | URL | Notes |
|---|---|---|
| Software Carpentry git-novice | https://swcarpentry.github.io/git-novice/ | No branching/PRs |
| Carpentries — branches & PRs | https://carpentries-incubator.github.io/git-novice-branch-pr/ | Fills that gap |
| CodeRefinery Git intro + Collaborative Git | https://coderefinery.github.io/git-intro/ · https://coderefinery.github.io/git-collaborative/ | European, well maintained, exercises |
| MIT Missing Semester — Version Control | https://missing.csail.mit.edu/2026/version-control/ | Data model first |
| Pro Git 2e | https://git-scm.com/book/en/v2 | Ch. 3, 6 |
| GitHub Skills | https://github.com/skills/introduction-to-github · https://github.com/skills/review-pull-requests · https://github.com/skills/resolve-merge-conflicts · https://github.com/skills/test-with-actions | Auto-checked exercises |
| Learn Git Branching | https://learngitbranching.js.org/ | Visual sandbox |
| GitHub Actions — Python | https://docs.github.com/en/actions/tutorials/build-and-test-code/python | CI template |
| Google Eng Practices — Code Review | https://google.github.io/eng-practices/review/ | Reviewer + author guides |
| Codespaces / Education | https://docs.github.com/en/codespaces · https://github.com/education/students | Uniform environment |

## 3. Databases & data manipulation
| Title | URL | Notes |
|---|---|---|
| Python for Data Analysis 3e | https://wesmckinney.com/book/ | Open access. Predates pandas 3.0 |
| Polars User Guide | https://docs.pola.rs/user-guide/ · https://docs.pola.rs/user-guide/migration/pandas/ | |
| Modern Polars | https://kevinheavey.github.io/modern-polars/ | pandas vs Polars |
| DuckDB docs | https://duckdb.org/docs/stable/sql/introduction · https://duckdb.org/docs/stable/clients/python/overview | No-server SQL |
| Ibis | https://ibis-project.org/tutorials/getting_started | One API, many backends |
| SQLAlchemy 2.0 tutorial / SQLModel | https://docs.sqlalchemy.org/en/20/tutorial/ · https://sqlmodel.tiangolo.com/tutorial/ | |
| SQLBolt / Select Star SQL | https://sqlbolt.com/ · https://selectstarsql.com/ | Browser SQL practice |
| PostgreSQL Exercises | https://pgexercises.com/ | Joins, window functions |
| SWC Databases and SQL | https://swcarpentry.github.io/sql-novice-survey/ | |
| dbt Fundamentals / dbt-duckdb | https://learn.getdbt.com/courses/dbt-fundamentals · https://github.com/duckdb/dbt-duckdb | Modelling on a laptop |
| Tidy Data (Wickham 2014) | https://www.jstatsoft.org/article/view/v059i10 | |

## 4. Visualisation
| Title | URL | Notes |
|---|---|---|
| Fundamentals of Data Visualization (Wilke) | https://clauswilke.com/dataviz/ | R examples; principles don't depend on the tool |
| Data Visualization (Healy) | https://socviz.co/ | Concepts |
| UW Visualization Curriculum | https://idl.uw.edu/visualization-curriculum/ | Altair 6 notebooks |
| Vega-Altair | https://altair-viz.github.io/ | |
| plotnine / seaborn.objects / Plotly | https://plotnine.org/guide/overview.html · https://seaborn.pydata.org/tutorial/objects_interface.html · https://plotly.com/python/ | seaborn.objects still experimental |
| Scientific Visualization (Rougier) | https://github.com/rougier/scientific-visualization-book | Advanced matplotlib |
| GeoPandas / Folium / Lonboard / leafmap | https://geopandas.org/en/stable/ · https://developmentseed.org/lonboard/latest/ | |
| Geographic Data Science with Python | https://geographicdata.science/book/ | |
| Streamlit / Shiny for Python / Dash / Panel / Quarto dashboards | https://docs.streamlit.io/get-started/tutorials · https://shiny.posit.co/py/docs/overview.html · https://quarto.org/docs/dashboards/ | |
| Accessibility | https://colorbrewer2.org/ · https://www.fabiocrameri.ch/colourmaps/ · https://chartability.fizz.studio/ | Chartability as a rubric |
| Chart choice | https://ft-interactive.github.io/visual-vocabulary/ · https://www.data-to-viz.com/ | |

## 5. Peer courses
| Course | URL | Notes |
|---|---|---|
| Berkeley Data 100 Fa26 | https://ds100.org/fa26/ · https://learningds.org/ | Polars; public HW/projects |
| UCSD DSC 80 | https://dsc80.com/ | Labs + projects on GitHub |
| MIT Missing Semester 2026 | https://missing.csail.mit.edu/ | Tooling track |
| UBC MDS DSCI 524 | https://ubc-mds.github.io/DSCI_524_collab-sw-dev/ | Team package project; grading model |
| Data Science: A First Introduction (Python) | https://python.datasciencebook.ca/ | Bridge material |
| Python for Data Science (Turrell) | https://aeturrell.github.io/python4DS/ | R4DS port |
| Hands-on Intro to Data Science (F. Huber, HS Düsseldorf) | https://florian-huber.github.io/data_science_course/ | German UAS peer, 2026 |
| Python Data Science Handbook | https://jakevdp.github.io/PythonDataScienceHandbook/ | Free = 1st ed. (dated) |
| TU Delft MUDE | https://mude.citg.tudelft.nl/ | EU MSc, Jupyter Book |

Excluded (dead or unusable): merely-useful.tech, cs109a.org, CodeRefinery good-enough-practices page, UBC DSCI 513/532, COGS108 sites. Harvard CS109 is Canvas-only since ~2021.
