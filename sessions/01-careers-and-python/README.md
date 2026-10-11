# Session 1 · Data careers, the final project challenge and Python for analysis

> **Guiding question.** What do data professionals do, what will we build in this course, and how do we start working with Python?

**Learning outcomes.** After this session you are able to

- describe the main data roles, their tasks and the skills they require
- explain the task of the final project challenge
- work in Jupyter notebooks and use Python's data types and data structures
- write small programs with conditions, loops and functions, and load and summarise a table with pandas

## Why this session matters

Data work combines a question, evidence and a decision, and job titles describe that work only loosely. This session first separates the roles by their tasks and outputs, explains how the course and its assessment are organised, and introduces the challenge that every team works on until the final presentations: predicting the customs heading of a product from its description. It then teaches Python from the start, assuming no earlier programming: notebooks, values and collections, conditions, loops and functions, and a first analysis of the Berlin Airbnb listings with pandas. Session 2 builds on this and turns analysis code into tested, shared software.

> This page is the teaching overview of the session. How to work with it (the ▶ links, the numbered notebooks, the extras) is explained once in the [course README](../../README.md#how-to-use-the-session-pages).

**Notes for this session**

- Session 1 has three blocks. Block 2 starts with a short third-party introduction to Jupyter before the lecture notebook.
- The job advertisements of the Block 1 exercise are handed out by the lecturer; they are not in the repository (see [Setup](#setup)).

## Block 1 · Data careers, the course and the challenge

### Theory (45 min)

**Why.** The same job title can mean different work in different organisations, and one person may cover several functions. Reading the tasks and outputs behind a title tells you which skills a role really needs and what to show in a project. This block also explains how the course moves through that work, how it is assessed, and which problem your team will solve.

**Topics**

- **Common data roles.** Data analyst, data scientist, data engineer, ML engineer and AI engineer differ in their main responsibility and typical outputs. They overlap, and the boundaries depend on the team and the product.
- **Skills shared across roles.** Technical, method, domain and communication skills. A strong result is more than code that runs: it needs a definition of the population, quality checks and a statement of its limits.
- **What Berlin job ads ask for.** A heat map of 238 Berlin advertisements shows the shared base (SQL, Python, communication) and the signature skills of each role. Knowing it helps you choose what to learn and what to show in a project.
- **Reading a job description.** Separating an advertisement into problem, deliverables, methods, tools, collaborators and requirements shows what the work is, not just which tools are listed.
- **Course organisation and assessment.** 18 sessions in blocks of 45 minutes of theory, 30 minutes of exercises and 15 minutes of debrief. The final project, presented in Session 17 or 18, is the only graded component.
- **The final project challenge.** Predicting the four-digit customs heading of a product from its description, using EU binding tariff decisions (EBTI) in 23 languages, with a class leaderboard. A high leaderboard score alone does not determine the grade.

▶ **Lecture:** [01-lecture-data-careers.ipynb](workbooks/01-lecture-data-careers.ipynb)

### Exercises (30 min) and debrief (15 min)

▶ **Exercises:** [session-01-exercises.ipynb](workbooks/session-01-exercises.ipynb), part 1.

- **Five real advertisements per role** (Groups, Paper, 25 min). Each group takes one role and its five Berlin job advertisements (handed out by the lecturer, retrieved 11 October 2026), fills the six-heading table, marks the shared skills, compares them with the heat map and reports what the role does all day.
- **Be the classifier** (Pairs, Paper, 10 min). Choose the heading of four real shoe descriptions from three candidate headings and mark the deciding words.
- **Which inputs are allowed?** (Pairs, Paper, 5 min). Decide which fields of a customs decision a model may use for a new request.

**Debrief.** Each group reports its role in two minutes; compare where five ads differ from the heat map. For the classifier task, compare the answers to the descriptions that contain "leather" or "suede" but describe plastic.

## Block 2 · Notebooks and Python's data types

### Theory (45 min)

**Why.** Every later session runs in notebooks and builds on Python's values and collections. A notebook result only counts if it survives a clean rerun in the course environment, and a value of the wrong type (a price stored as text, a missing value stored as zero) produces wrong results without any error. This block teaches these foundations from the start.

**Topics**

- **The working environment: uv, VS Code and the course environment.** uv installs Python and the packages from a lock file into `.venv`, so the code runs the same on every laptop. The notebook's kernel must be `.venv`.
- **Jupyter notebooks: cells, kernel, Markdown, order of execution.** The kernel keeps every value in memory and runs cells in the order you run them, not the order on the page. Restart and run all before sharing a result.
- **Variables, expressions and operators.** `=` binds a name to a value once; it is not a formula. Arithmetic, whole-number division, remainder and comparisons.
- **Numbers, strings, booleans and None.** Every value has a type. A missing price is `None`, not zero, and an identifier such as a postal code stays text.
- **Lists, tuples, dictionaries and sets.** Ordered sequences, fixed groups, records with named fields and unique values; a list of dictionaries is a first table.
- **Getting help: `help()`, `dir()` and the documentation.** The built-in help and the error message are the first sources; the official documentation is the authority.

▶ **Worked example (first):** [02-jupyter-introduction.ipynb](workbooks/02-jupyter-introduction.ipynb) (*Think Python*): what a notebook is, how to run cells, command and edit mode. For everyone who has not used Jupyter before.

▶ **Lecture:** [03-lecture-notebooks-and-data-types.ipynb](workbooks/03-lecture-notebooks-and-data-types.ipynb)

### Exercises (30 min) and debrief (15 min)

▶ **Exercises:** [session-01-exercises.ipynb](workbooks/session-01-exercises.ipynb), part 2.

- **Swap two values** (Paper, then Code, 5 min). Swap `a` and `b` with a third variable, then without one.
- **Is 0.1 + 0.2 equal to 0.3?** (Predict, 3 min). Predict a comparison of floats, then run it.
- **The alias trap** (Predict, 3 min). Predict what happens to a list when a second name refers to it.
- **Out of order** (Predict, 5 min). Predict what a notebook prints when cells run out of order and after a restart.
- **Pick the container** (Paper, 5 min). Choose a list, tuple, dictionary, set or single value for five pieces of listing data.
- **Find it yourself with `help()` and `dir()`** (Code, 10 min). Four small tasks solved with the built-in help.
- *Stretch:* **Rice on a chessboard** (Code, 5 min). How many grains after 64 doublings?

**Debrief.** Compare the float and alias predictions before running the code, and look for the swap that overwrites `a` before saving it. For the containers, ask why the postal code is not an `int`.

## Block 3 · Control flow, functions and first steps with pandas

### Theory (45 min)

**Why.** Programs become useful when they make decisions, repeat work and give a rule a name so that it can be checked and reused. pandas then does the same work on whole tables: the first real question about the 12,776 Berlin listings takes a few lines. The block ends with how to use AI coding assistants without trusting them blindly.

**Topics**

- **Conditions: `if`, `elif`, `else`.** Only the first true branch runs, so the order of the conditions matters; boundaries such as "exactly on budget" need a deliberate choice of `<` or `<=`.
- **Loops: `for`, `while`, comprehensions.** A `for` loop with an accumulator processes each record; `while` repeats until a condition fails; a comprehension builds a list in one line.
- **Functions: parameters, return values, docstrings.** A function gives a rule a name and makes it testable. `return` hands back a value, `print` only shows it.
- **Importing modules and packages.** The standard library and third-party packages such as pandas, installed by uv into the course environment.
- **pandas: Series and DataFrame; reading, selecting, filtering, sorting and grouping a table.** Define what a row is, select with `.loc` and Boolean masks, find missing values with `.isna()`, summarise with `value_counts()` and `groupby`. For skewed prices the median says more than the mean.
- **Tips for using AI coding assistants.** Generated code is a draft: specify inputs and edge cases, check on a table small enough to compute by hand, and document the use.

▶ **Lecture:** [04-lecture-control-flow-functions-and-pandas.ipynb](workbooks/04-lecture-control-flow-functions-and-pandas.ipynb)

▶ **Worked examples**

- [05-case-study-five-questions.ipynb](workbooks/05-case-study-five-questions.ipynb): five questions about the Berlin Airbnb listings (how many, where, of what type, at what price, with how many reviews), with worked answers and a reference solution. For self-study after the session if there is no time in class.

### Exercises (30 min) and debrief (15 min)

▶ **Exercises:** [session-01-exercises.ipynb](workbooks/session-01-exercises.ipynb), part 3.

- **FizzBuzz** (Code, 10 min). Numbers 1 to 100 with `Fizz`, `Buzz` and `FizzBuzz`.
- **Maximum without `max()`, then the second-largest** (Code, 10 min). Write both functions; decide first what `second_largest([5, 5, 3])` should be.
- **Collatz** (Code, 10 min). Count the steps from 27 to 1 with a `while` loop.
- **Word count with a dictionary** (Code, 10 min). The three most frequent words of a sentence.
- **Average price per room type by hand** (Code, 15 min). Ten listings as a list of dictionaries, loops only; one price is missing.
- **Large flats in the Berlin listings** (Pairs, Code, 15 min). Short-stay entire homes for four or more guests: where they are, and their price per night and per guest compared with small flats.
- *Stretch:* **Caesar cipher** (Code, 20 min) and **Armstrong numbers** (Code, 10 min).

**Debrief.** In FizzBuzz, look for the solution that tests multiples of 3 before multiples of 15. In the word count, check whether "The" and "the", and "quiet," with its comma, were counted as one word. In the price tasks, ask what happened to the listing without a price and which rows were counted.

## Team project until the next session

Teams of three are formed and read the [project brief](../17-18-final-presentations/final-project.md) and the leaderboard rules. Each member creates a GitHub account before Session 2.

## Extras (optional, self-study)

The reference notebooks go deeper on individual topics; each lecture notebook links them with a 📖 **Read more** pointer. They are not a required run-all sequence.

| Notebook | Content | Block |
|---|---|---|
| [extras/01-python-variables.ipynb](workbooks/extras/01-python-variables.ipynb) | Variables, dynamic typing, object references (*Whirlwind Tour of Python*) | 2 |
| [extras/02-python-scalar-types.ipynb](workbooks/extras/02-python-scalar-types.ipynb) | `int`, `float`, `str`, `bool`, `None` | 2 |
| [extras/03-python-data-structures.ipynb](workbooks/extras/03-python-data-structures.ipynb) | Lists, tuples, dictionaries, sets | 2 |
| [extras/04-python-control-flow.ipynb](workbooks/extras/04-python-control-flow.ipynb) | `if`/`elif`/`else`, `for` and `while` loops, `break`/`continue` | 3 |
| [extras/05-python-functions.ipynb](workbooks/extras/05-python-functions.ipynb) | Defining functions, default and keyword arguments, `*args`/`**kwargs`, `lambda` | 3 |
| [extras/06-python-errors-and-exceptions.ipynb](workbooks/extras/06-python-errors-and-exceptions.ipynb) | Errors, `try`/`except`/`finally`, raising exceptions (prepares Session 2) | 3 |
| [extras/07-pandas-introduction.ipynb](workbooks/extras/07-pandas-introduction.ipynb) | Beginner pandas: Series, DataFrame, selection, filtering, grouping, time series. Some examples need updating for current pandas; use the lecture notebook and the case study as the first runnable path | 3 |
| [extras/08-pandas-indexing-and-selection.ipynb](workbooks/extras/08-pandas-indexing-and-selection.ipynb) | `loc`, `iloc`, Boolean masks (PDSH 3.02). **ND**: shared unchanged | 3 |
| [extras/09-pandas-merge-and-join.ipynb](workbooks/extras/09-pandas-merge-and-join.ipynb) | Joining tables with `pd.merge`, US states case study (PDSH 3.07; prepares Session 3). Reads `data/state-*.csv`. **ND** | 3 |
| [extras/10-pandas-groupby.ipynb](workbooks/extras/10-pandas-groupby.ipynb) | Aggregation, split-apply-combine, `groupby` (PDSH 3.08). **ND** | 3 |
| [extras/11-python-modules-and-packages.ipynb](workbooks/extras/11-python-modules-and-packages.ipynb) | `import` forms, standard library, third-party packages | 3 |

Classes and objects are taught in Session 2; the *Think Python* chapters on classes are among its extras.

## Before the session and further reading

**Before.** Set up your laptop as described in the [course README](../../README.md#setup) (VS Code, Git, uv, the course repository and `uv sync`), then generate the case-study data of Sessions 1–12 once from the repository root (downloads the Berlin snapshot of Inside Airbnb, about 100 MB, and the Berlin weather from Open-Meteo; writes about 6 MB of Parquet files to `case-study/data/airbnb/`):

```bash
uv run python case-study/prepare_airbnb.py
```

The EBTI customs data of the final project are prepared before Session 3 with `case-study/prepare_data.py`.

**After.** Work through the [case study](workbooks/05-case-study-five-questions.ipynb), then repeat questions 4 and 5 for the district you live in or know best and compare it with the city as a whole. Finish the exercises you did not complete in class.

**Further reading (optional)**

- [The Python Tutorial](https://docs.python.org/3/tutorial/), sections 3–6 (Python Software Foundation)
- [Python for Data Analysis, 3rd edition](https://wesmckinney.com/book/), chapters 2–5 (McKinney, open access)
- [Think Python, 3rd edition](https://allendowney.github.io/ThinkPython/), chapters 1–11 (Downey, free online)
- [pandas getting-started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
- [Good enough practices in scientific computing](https://doi.org/10.1371/journal.pcbi.1005510) (Wilson et al. 2017)
- [US Bureau of Labor Statistics: data scientists](https://www.bls.gov/ooh/math/data-scientists.htm) and [O*NET: data-warehousing specialists](https://www.onetonline.org/link/details/15-1243.01), task-based descriptions of US occupations

## Setup

All notebooks of this session run in the course environment: open them in VS Code with the `.venv` kernel (setup: [course README](../../README.md#setup)). The lecture notebook of Block 3, the case study and exercise 3.6 read `case-study/data/airbnb/listings.parquet` (see *Before* above).

The *Think Python* notebook (02-jupyter-introduction) downloads a small helper file (`thinkpython.py`, MIT) from the author's repository when its last cells run, which needs an internet connection; it was written for Google Colab, and its references to Colab buttons correspond to the notebook toolbar in VS Code. The merge notebook (extras 09) reads the US state tables in `workbooks/extras/data/state-*.csv`.

To regenerate the figures of the lecture notebooks, run `uv run python sessions/01-careers-and-python/figures/make_figures.py`. To refresh the skills heat map with today's job ads, first run `uv run --with trafilatura python sessions/01-careers-and-python/figures/job_ad_skills.py` (about 250 page requests to the Data Berlin board).

**Job advertisements for the Block 1 exercise.** The 25 PDFs (five per role) are kept outside the public repository because the texts belong to the employers; hand them out through the learning platform. Each PDF states the source, the original URL and the retrieval date; contact persons are removed.

## Sources and licences

Keep the attribution when you reuse or share a file. Course-written material is licensed CC BY 4.0. Third-party notebooks keep their original licence; "ND" means the file must be shared unchanged, "NC" means non-commercial use only and "SA" means adaptations must use the same licence.

**Course material (course team)**

| File | Content | Licence |
|---|---|---|
| [01-lecture-data-careers.ipynb](workbooks/01-lecture-data-careers.ipynb) | Data roles, skills, Berlin job ads, course organisation and assessment, the final project challenge | CC BY 4.0 |
| [03-lecture-notebooks-and-data-types.ipynb](workbooks/03-lecture-notebooks-and-data-types.ipynb) | uv, VS Code, Jupyter, variables and operators, numbers, strings, booleans, None, collections, getting help | CC BY 4.0 |
| [04-lecture-control-flow-functions-and-pandas.ipynb](workbooks/04-lecture-control-flow-functions-and-pandas.ipynb) | Conditions, loops, functions, imports, pandas, AI coding assistants | CC BY 4.0 |
| [05-case-study-five-questions.ipynb](workbooks/05-case-study-five-questions.ipynb) | Worked example: five questions about the Berlin Airbnb listings (how many, where, what type, what price, how many reviews) | CC BY 4.0 |
| [session-01-exercises.ipynb](workbooks/session-01-exercises.ipynb) | Exercises of the three blocks: job advertisements and the challenge; values, types and collections; conditions, loops, functions and pandas | CC BY 4.0 |
| [figures/make_figures.py](figures/make_figures.py) and its PNGs (`role_map.png`, `skills_by_role.png`, `price_by_room_type.png`, `listings_by_district.png`) | Roles along the path from data to product (course team's simplified view); skills named in Berlin job ads by role; short-stay prices by room type; listings and prices by district (Inside Airbnb data) | CC BY 4.0; data CC BY 4.0, Inside Airbnb |
| [figures/job_ad_skills.py](figures/job_ad_skills.py) and [figures/job_ad_skills.csv](figures/job_ad_skills.csv) | Downloads the open Berlin data and AI ads from the Data Berlin board and counts, per role, the share of ads naming each skill group; the CSV holds the aggregated shares of 11 October 2026 (no ad texts) | CC BY 4.0, course team |

The lecture notebooks are informed by the canonical references linked in each of them; their code examples use small fictional records or the prepared course dataset, and the language and library behaviour was checked against those references. Several exercises come from established sources named in their *Source* line.

**Third-party notebooks**

| File | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|
| [02-jupyter-introduction.ipynb](workbooks/02-jupyter-introduction.ipynb) | [AllenDowney/ThinkPython](https://raw.githubusercontent.com/AllenDowney/ThinkPython/v3/chapters/jupyter_intro.ipynb) | [MIT (code) / CC BY-NC-SA 4.0 (text)](https://allendowney.github.io/ThinkPython/) **NC: non-commercial only** | 2026-10-01 | Renamed from `jupyter_intro.ipynb`; content unchanged |
| [extras/01-python-variables.ipynb](workbooks/extras/01-python-variables.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/03-Semantics-Variables.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-09-26 | Renamed from `03-Semantics-Variables.ipynb`; content unchanged |
| [extras/02-python-scalar-types.ipynb](workbooks/extras/02-python-scalar-types.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/05-Built-in-Scalar-Types.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-09-26 | Renamed from `05-Built-in-Scalar-Types.ipynb`; content unchanged |
| [extras/03-python-data-structures.ipynb](workbooks/extras/03-python-data-structures.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/06-Built-in-Data-Structures.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-09-26 | Renamed from `06-Built-in-Data-Structures.ipynb`; content unchanged |
| [extras/04-python-control-flow.ipynb](workbooks/extras/04-python-control-flow.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/07-Control-Flow-Statements.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-09-26 | Renamed from `07-Control-Flow-Statements.ipynb`; content unchanged |
| [extras/05-python-functions.ipynb](workbooks/extras/05-python-functions.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/08-Defining-Functions.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-09-26 | Renamed from `08-Defining-Functions.ipynb`; content unchanged |
| [extras/06-python-errors-and-exceptions.ipynb](workbooks/extras/06-python-errors-and-exceptions.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/09-Errors-and-Exceptions.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-09-26 | Renamed from `09-Errors-and-Exceptions.ipynb`; content unchanged |
| [extras/07-pandas-introduction.ipynb](workbooks/extras/07-pandas-introduction.ipynb) | [microsoft/Data-Science-For-Beginners](https://raw.githubusercontent.com/microsoft/Data-Science-For-Beginners/main/2-Working-With-Data/07-python/notebook.ipynb) | [MIT](https://github.com/microsoft/Data-Science-For-Beginners/blob/main/LICENSE) | 2026-09-26 | Renamed from `notebook.ipynb` (lesson `2-Working-With-Data/07-python`); content unchanged |
| [extras/08-pandas-indexing-and-selection.ipynb](workbooks/extras/08-pandas-indexing-and-selection.ipynb) | [jakevdp/PythonDataScienceHandbook](https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/03.02-Data-Indexing-and-Selection.ipynb) | [MIT (code) / CC-BY-NC-ND 3.0 (text)](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/LICENSE-CODE) **NC, ND** | 2026-09-26 | Renamed from `03.02-Data-Indexing-and-Selection.ipynb`; content unchanged (ND: must stay unchanged) |
| [extras/09-pandas-merge-and-join.ipynb](workbooks/extras/09-pandas-merge-and-join.ipynb) | [jakevdp/PythonDataScienceHandbook](https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/03.07-Merge-and-Join.ipynb) | [MIT (code) / CC-BY-NC-ND 3.0 (text)](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/LICENSE-CODE) **NC, ND** | 2026-09-26 | Renamed from `03.07-Merge-and-Join.ipynb`; content unchanged (ND: must stay unchanged) |
| [extras/10-pandas-groupby.ipynb](workbooks/extras/10-pandas-groupby.ipynb) | [jakevdp/PythonDataScienceHandbook](https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/03.08-Aggregation-and-Grouping.ipynb) | [MIT (code) / CC-BY-NC-ND 3.0 (text)](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/LICENSE-CODE) **NC, ND** | 2026-09-26 | Renamed from `03.08-Aggregation-and-Grouping.ipynb`; content unchanged (ND: must stay unchanged) |
| [extras/11-python-modules-and-packages.ipynb](workbooks/extras/11-python-modules-and-packages.ipynb) | [jakevdp/WhirlwindTourOfPython](https://raw.githubusercontent.com/jakevdp/WhirlwindTourOfPython/master/13-Modules-and-Packages.ipynb) | [CC0-1.0](https://github.com/jakevdp/WhirlwindTourOfPython/blob/master/LICENSE) | 2026-10-01 | Renamed from `13-Modules-and-Packages.ipynb`; content unchanged |

The files `workbooks/extras/data/state-*.csv` belong to extras 09 and come from the same repository ([jakevdp/PythonDataScienceHandbook, notebooks/data](https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data)).

> **Note.** The *Think Python* notebook (02-jupyter-introduction) is licensed CC BY-NC-SA 4.0 for the text: it may be shared and adapted for non-commercial teaching with attribution and under the same licence. The Python Data Science Handbook text (extras 08–10) is CC BY-NC-ND 3.0: shared unchanged only.

**Data**

| Dataset | Used in | Source | Licence |
|---|---|---|---|
| Inside Airbnb, Berlin snapshot of 26 June 2026 (listings), prepared by `case-study/prepare_airbnb.py` | lecture notebook 04, case study 05, exercise 3.6, figures | [insideairbnb.com](https://insideairbnb.com/get-the-data/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); host names, texts and photos removed by the script |
| Open-Meteo historical weather, downloaded by the same script | later sessions | [open-meteo.com](https://open-meteo.com/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| EBTI binding tariff decisions (counts and examples in lecture notebook 01; four descriptions in exercise 1.2), prepared by `case-study/prepare_data.py` | lecture notebook 01, exercises 1.2 and 1.3 | [European Commission, EBTI database](https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en) | Reuse with acknowledgement of the source (Commission Decision 2011/833/EU) |
| Language shares chart (`proposal/assets/final-project/languages.png`) | lecture notebook 01 | course team, computed from the EBTI data | CC BY 4.0 |

The data are not stored in the repository; each student downloads them with the scripts.

<details>
<summary><b>References cited in the lecture notebooks</b></summary>

- Astral (2026). *uv documentation: working on projects*. https://docs.astral.sh/uv/guides/projects/
- Beauchemin, M. (2017). *The Rise of the Data Engineer*. freeCodeCamp. https://www.freecodecamp.org/news/the-rise-of-the-data-engineer-91be18f1e603/
- Bitkom (2026, 3 September). *IT-Fachkräftemangel in den letzten drei Jahren halbiert*. https://www.bitkom.org/Presse/Presseinformation/IT-Fachkraeftemangel-in-den-letzten-drei-Jahren-halbiert
- Bitkom Research (2026). *IT-Fachkräfte 2025: Studienbericht* (survey of 855 companies), p. 14. https://www.bitkom.org/sites/main/files/2026-01/bitkom-studienbericht-it-fachkraefte-2025.pdf
- Data Berlin (2026). *Skills Dictionary* and *Seniority insights*, live views of the Berlin data and AI job board (retrieved 2026-10-11). https://databerlin.net/skills, https://databerlin.net/insights/seniority/intern-student
- Data Berlin (2026). *Jobs*, Berlin data and AI job board; source of the 238 advertisements of the skills heat map and of the 25 handed-out advertisements (retrieved 2026-10-11). https://databerlin.net/jobs
- Downey, A. B. (2024). *Think Python*, 3rd edition. O'Reilly. https://allendowney.github.io/ThinkPython/
- European Commission (2026). *European Binding Tariff Information (EBTI) database*. https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en
- Grewal, E. (2018). *One Data Science Job Doesn't Fit All*. https://www.linkedin.com/pulse/one-data-science-job-doesnt-fit-all-elena-grewal/
- Inside Airbnb (2026). *Berlin, 26 June 2026 snapshot*; *About Inside Airbnb*. CC BY 4.0. https://insideairbnb.com/get-the-data/
- Indeed Hiring Lab (2026, 23 September). *Berufseinstieg: Stellenangebot sinkt*. https://hiringlab.indeed.com/de/blog/2026/09/23/berufseinstieg-stellenangebot-sinkt-ausbildungssystem-bietet-einen-anker/
- McKinney, W. (2022). *Python for Data Analysis*, 3rd edition. O'Reilly. https://wesmckinney.com/book/
- pandas development team (2026). *Getting started tutorials*; *Working with missing data*. https://pandas.pydata.org/docs/
- Project Jupyter (2026). *JupyterLab documentation: notebooks*. https://jupyterlab.readthedocs.io/en/stable/user/notebook.html
- Python Software Foundation (2026). *The Python Tutorial*. https://docs.python.org/3/tutorial/
- VanderPlas, J. (2016). *A Whirlwind Tour of Python*. O'Reilly. https://jakevdp.github.io/WhirlwindTourOfPython/
- VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly. https://jakevdp.github.io/PythonDataScienceHandbook/
- Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L., & Teal, T. K. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510. https://doi.org/10.1371/journal.pcbi.1005510

</details>
