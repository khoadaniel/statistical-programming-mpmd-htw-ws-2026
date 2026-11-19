# Session 2 · Software engineering for data work

> **Guiding question.** How do we turn analysis code into code that others can use, test and develop together without breaking it?

## 1 · Classes, errors and tests

[Open notebook](workbooks/01-lecture-classes-errors-and-tests.ipynb)

- Classes and objects: attributes, methods, `self`, class attributes, properties, comparing and copying objects.
- Composition, inheritance, dataclasses and polymorphism.
- Type hints, docstrings, doctest and comments.
- Exceptions: the hierarchy, own exception classes, raising, handling, chaining and the call stack.
- Automated tests with pytest: why test, fixtures, parametrize, choosing test cases, test-driven development.
- Worked example: does every Berlin listing pass the rules of `ListingRecord`?

**Practice first:** [Python classes, step by step](workbooks/03-practice-python-classes.ipynb) builds a product, a shopping cart and their subclasses one idea at a time, as a warm-up for the [supermarket challenge](workspace/supermarket_challenge/README.md).

## 2 · From notebook to application, Git and continuous integration

[Open notebook](workbooks/02-lecture-notebook-to-application-git-and-ci.ipynb)

- From notebook to application: hidden state, explicit inputs, when to use which.
- From a script to a command in seven steps: scripts, `__name__` and `"__main__"`, packages, `python -m`, a command-line interface with `argparse`, `[project.scripts]`; then project structure and packaging.
- Git: repositories, commits, `.gitignore`, branches, merging, merge conflicts, undoing, remotes, tags and releases.
- GitHub: pull requests, fork and pull, code review.
- Continuous integration with GitHub Actions: workflows, linting and formatting with ruff, branch protection.
- Worked example: does the listing package pass the checks CI would run?

## Team project until the next session

Team repository from the template, with branch protection and CI (exercise 2.5). Each member opens at least one reviewed pull request. Session 3 follows on the same day.

## Setup

- A free [GitHub account](https://github.com/signup), and your name and e-mail set in Git: `git config --global user.name "..."` and `git config --global user.email "..."`.
- All end-to-end projects of this session are uv projects in [workspace/](workspace/), one folder each. Every one is run from its own folder with `uv run`.
- [listing_tools](workspace/listing_tools/README.md): validated listings and a command-line summary, with tests and CI. `cd sessions/02-software-engineering/workspace/listing_tools`, then `uv run pytest -q`. Its command `listing-summary` reads the case-study listings: get the data with `git lfs pull` (see [case-study/README.md](../../case-study/README.md)).
- [staycost_cli](workspace/staycost_cli/README.md): a tiny project with no data, the command `staycost` built in the seven steps of block 2. `cd sessions/02-software-engineering/workspace/staycost_cli`, then `uv run staycost 80 --nights 3`.
- [gradebook_module_guide](workspace/gradebook_module_guide/README.md): a student gradebook that shows how the files of a package import each other: absolute imports, why the course avoids relative imports, `python -m` against a file path, and why the folder you open the terminal in matters. `cd sessions/02-software-engineering/workspace/gradebook_module_guide`, then `uv run gradebook`.
- [supermarket_challenge](workspace/supermarket_challenge/README.md): HTW Markt, a small shop in plain classes. You teach its till to sell sandwiches cheaper on their last day and to refuse expired ones, in three short tasks on inheritance, overriding a method and your own exception. `cd sessions/02-software-engineering/workspace/supermarket_challenge`, then `uv run htw-checkout lina`.
- [airbnb_price_regression](workspace/airbnb_price_regression/README.md): how to organise a machine learning project. A linear regression (`price = a * guests + b`) in object-oriented code, with training, a saved checkpoint and a FastAPI service that you call with `curl`; `make train`, then `make serve`. `cd sessions/02-software-engineering/workspace/airbnb_price_regression`, then `make install`.
- The [Git workspace](git-workspace/README.md) holds four small classes for the team activity at the end of the Git deck: in pairs, you share one fork, open pull requests that merge cleanly, then resolve a conflict on the same line. Run it with `python3 main.py`; no uv needed.
- The notebooks contain Mermaid diagrams: VS Code shows them with the extension *Markdown Mermaid* (`bierner.markdown-mermaid`); JupyterLab 4.1 and newer renders them without an extension.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
