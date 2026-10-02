# Session 2 · Software engineering for data work: object-oriented programming, APIs, testing, Git and continuous integration

> [!NOTE]
> **Guiding question.** How do we write data code that others can use, connect it to other systems, and develop it together without breaking it?

**Learning outcomes.** After this session you are able to

- design and use classes for data tasks and handle errors explicitly
- request data from a web API and write automated tests for the code that processes it
- work with Git branches and pull requests and set up a CI workflow that runs the tests

This is a workspace session: you work in one small project, [workspace/](workspace/README.md), that grows through the three blocks.

## Session plan

**0:00–0:45 · Object-oriented programming, errors and tests** ([theory](theory/01-oop-errors-and-tests.md))

- [Object-oriented programming in practice: classes, methods, composition and inheritance, dataclasses](theory/01-oop-errors-and-tests.md#object-oriented-programming-in-practice)
- [Type hints and docstrings](theory/01-oop-errors-and-tests.md#type-hints-and-docstrings)
- [Exceptions and error handling](theory/01-oop-errors-and-tests.md#exceptions-and-error-handling)
- [Automated tests with pytest](theory/01-oop-errors-and-tests.md#automated-tests-with-pytest)
- *Practice:* write and test a class that validates and cleans one listing (price stored as text, coordinates inside Berlin, valid room type): `ListingRecord`, workspace exercises 1–2

**1:00–1:45 · Web APIs** ([theory](theory/02-web-apis.md))

- [Why analysts need APIs: data that change every day and are not offered as a file, such as weather for a demand forecast](theory/02-web-apis.md#why-analysts-need-apis)
- [What an API is and how web APIs work](theory/02-web-apis.md#what-an-api-is-and-how-web-apis-work)
- [HTTP requests and responses, JSON, authentication, pagination and rate limits](theory/02-web-apis.md#http-requests-and-responses-json-authentication-pagination-and-rate-limits)
- [Requesting data with httpx](theory/02-web-apis.md#requesting-data-with-httpx)
- [A minimal API of one's own with FastAPI](theory/02-web-apis.md#a-minimal-api-of-ones-own-with-fastapi) (developed further in Session 16)
- *Practice:* does the weather explain how busy Berlin's Airbnb market is? Fetch daily Berlin weather from the Open-Meteo API, parse it into a validated class and test the parsing offline; the data are reused in the forecast of Session 12 ([workbook 03](workbooks/03-case-study-weather-api.ipynb), workspace exercises 3–5)

**2:00–2:45 · Git, GitHub and continuous integration** ([theory](theory/03-git-github-and-ci.md))

- [Git and GitHub: commits, branches, merging and merge conflicts](theory/03-git-github-and-ci.md#git-and-github-commits-branches-merging-and-merge-conflicts), with a [step-by-step merge-conflict exercise](theory/03-git-github-and-ci.md#exercise-a-merge-conflict-step-by-step)
- [Pull requests and code review](theory/03-git-github-and-ci.md#pull-requests-and-code-review)
- [Continuous integration with GitHub Actions (ruff and pytest on every pull request)](theory/03-git-github-and-ci.md#continuous-integration-with-github-actions)
- *Practice:* case study: contribute a tested module of simple features of listing titles (`text_features.py`: digits, upper-case share) through a reviewed pull request; teams set up their repository with CI (workspace exercises 6–8)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-oop-errors-and-tests.md](theory/01-oop-errors-and-tests.md) | Composition, inheritance, dataclasses; type hints, docstrings; exceptions; pytest | 1 | core |
| [theory/02-web-apis.md](theory/02-web-apis.md) | Why analysts need APIs; HTTP, JSON, authentication, pagination, rate limits, timeouts, retries; httpx and MockTransport; FastAPI | 2 | core |
| [theory/03-git-github-and-ci.md](theory/03-git-github-and-ci.md) | Commits, branches, merges, conflicts (exercise); pull requests, review; GitHub Actions | 3 | core |
| [workspace/](workspace/README.md) | Project `listingtools`: `ListingRecord`, `WeatherClient` and `DailyWeather` (Open-Meteo), FastAPI app (`/districts/{district}`, `POST /listings/validate`, `/features`), `text_features.py` stub, tests, `ci.yml`, solutions | 1–3 | core (practice) |
| [workbooks/01-classes-and-objects.ipynb](workbooks/01-classes-and-objects.ipynb) | Objects inside objects (composition), equivalence and identity, deep copy, polymorphism (*Think Python*) | 1 | optional |
| [workbooks/02-inheritance.ipynb](workbooks/02-inheritance.ipynb) | Parent and child classes, specialisation (*Think Python*) | 1 | optional |
| [workbooks/03-case-study-weather-api.ipynb](workbooks/03-case-study-weather-api.ipynb) | Berlin weather from Open-Meteo: live request, errors, rate limits, paging by year, `DailyWeather`, weather and monthly reviews, a test with MockTransport; optional VBB departures (own) | 2 | core |

Sources and licences: [source.md](source.md).

## Before and after the session

**Before.** Create a free [GitHub account](https://github.com/signup) (students can apply for the [GitHub Student Developer Pack](https://education.github.com/pack)), install Git, and set your name and e-mail (`git config --global user.name ...`). Run the workspace tests once: `cd sessions/02-software-engineering/workspace && uv run pytest -q`. Optional: work through [Learn Git Branching](https://learngitbranching.js.org/) (first four levels).

**Team project until the next session.** Team repository from the template, with branch protection and CI.

**After.** Finish the workspace exercises you did not complete in class. Each team member opens at least one reviewed pull request in the team repository.

**Further reading (optional).**

- [Software Carpentry: Version Control with Git](https://swcarpentry.github.io/git-novice/) (CC-BY 4.0) and [CodeRefinery: Collaborative distributed version control](https://coderefinery.github.io/git-collaborative/) (CC-BY 4.0)
- [Carpentries Incubator: Intermediate Research Software Development in Python](https://carpentries-incubator.github.io/python-intermediate-development/) (CC-BY 4.0): testing, design, collaboration around one project
- [MIT Missing Semester 2026: Version control; Code quality](https://missing.csail.mit.edu/2026/version-control/) (CC BY-NC-SA 4.0; linked, not copied)
- [Google Engineering Practices: Code review](https://google.github.io/eng-practices/review/)
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/) and [HTTPX documentation](https://www.python-httpx.org/)

## Setup

The workspace is its own uv project with `httpx`, `fastapi` and `uvicorn`, and `pytest`, `ruff` and `httpx2` (used by FastAPI's test client) as development tools:

```bash
cd sessions/02-software-engineering/workspace
uv run pytest -q
uv run uvicorn listingtools.api:app --reload     # optional: local server, stop with Ctrl+C
```

The workbooks need `httpx` and `pandas`; the *Think Python* workbooks download two helper files and use `matplotlib`:

```bash
uv run --with httpx --with pandas --with matplotlib --with jupyterlab jupyter lab
```

Workbook 03 and the client's live requests need internet access; the workspace tests do not. Workbook 03 also reads `case-study/data/airbnb/` (`uv run python case-study/prepare_airbnb.py`, Session 1).
