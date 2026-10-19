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
- The [workspace](workspace/README.md) is its own uv project: `cd sessions/02-software-engineering/workspace`, then `uv run pytest -q`. Its command `listing-summary` reads the case-study listings: download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- The [CLI workspace](cli-workspace/README.md) is a second, tiny uv project with no data: the command `staycost` built in the seven steps of block 2. `cd sessions/02-software-engineering/cli-workspace`, then `uv run staycost 80 --nights 3`.
- The notebooks contain Mermaid diagrams: VS Code shows them with the extension *Markdown Mermaid* (`bierner.markdown-mermaid`); JupyterLab 4.1 and newer renders them without an extension.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
