# Session 2 · Software engineering for data work

> **Guiding question.** How do we turn analysis code into code that others can use, test and develop together without breaking it?

## 1 · Classes, errors and tests

[Open notebook](workbooks/01-lecture-classes-errors-and-tests.ipynb)

- Classes and objects: attributes, methods, `self`, properties, class attributes and copies.
- Composition, inheritance, polymorphism and dataclasses.
- Type hints, docstrings, doctest and comments.
- Exceptions: raising, handling, own exception classes and the call stack.
- Automated tests with pytest: fixtures, parametrize, choosing test cases.
- Worked example: does every Berlin listing pass the rules of `ListingRecord`?

## 2 · From notebook to application, Git and continuous integration

[Open notebook](workbooks/02-lecture-notebook-to-application-git-and-ci.ipynb)

- From notebook to application: hidden state, explicit inputs, when to use which.
- Scripts, modules, packages, command-line programs and project structure.
- Git: commits, branches, merging, merge conflicts, undoing and remotes.
- GitHub: pull requests, fork and pull, code review.
- Continuous integration with GitHub Actions; linting and formatting with ruff.
- Worked example: does the listing package pass the checks CI would run?

## Team project until the next session

Team repository from the template, with branch protection and CI (exercise 2.5). Each member opens at least one reviewed pull request. Session 3 follows on the same day.

## Setup

- A free [GitHub account](https://github.com/signup), and your name and e-mail set in Git: `git config --global user.name "..."` and `git config --global user.email "..."`.
- The [workspace](workspace/README.md) is its own uv project: `cd sessions/02-software-engineering/workspace`, then `uv run pytest -q`. Its command `listing-summary` reads the case-study listings (`uv run python case-study/prepare_airbnb.py`, Session 1).
- The notebooks contain Mermaid diagrams: VS Code shows them with the extension *Markdown Mermaid* (`bierner.markdown-mermaid`); JupyterLab 4.1 and newer renders them without an extension.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
