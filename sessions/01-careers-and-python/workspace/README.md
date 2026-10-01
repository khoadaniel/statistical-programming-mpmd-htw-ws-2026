# Workspace · Session 1: from notebook to module

This small project shows the step from notebook code to an importable, tested module. It belongs to block 3 of Session 1 ([theory page](../theory/03-notebook-to-application.md)). The data are the course's EU Binding Tariff Information (BTI) decisions.

```
workspace/
├── pyproject.toml                project description, dependencies, tool settings
├── notebooks/
│   └── explore_decisions.py      step 0: the analysis as notebook cells (# %%)
├── src/btitools/
│   ├── __init__.py               makes the folder a package; re-exports DecisionTable
│   ├── decisions.py              the class DecisionTable (load, summarise)
│   ├── cli.py                    command-line entry point `decision-summary`
│   └── __main__.py               allows `python -m btitools`
├── tests/
│   ├── conftest.py               six hand-made decisions used by all tests
│   ├── test_decisions.py         tests of the finished parts (pass in the starter state)
│   └── test_exercises.py         tests of the exercises (skipped until you do them)
└── solutions/decisions.py        reference solution, for self-checking only
```

## Run it

You need the case-study data (`uv run python case-study/prepare_data.py`, run once from the repository root).

```bash
# from this folder: run the tests (uv creates .venv and installs the package)
uv run pytest -q                    # 9 passed, 3 skipped

# from the repository root: print the summary of the 50,000-decision sample
uv run --project sessions/01-careers-and-python/workspace decision-summary

# lint and format
uvx ruff check .
uvx ruff format --check .
```

In a notebook or the Python console of this environment the class is imported like any library:

```python
import pandas as pd
from btitools import DecisionTable

table = DecisionTable.from_parquet("case-study/data/train_sample.parquet")
table.language_shares()
table.top_headings(5, nomenclature=pd.read_parquet("case-study/data/nomenclature.parquet"))
```

## Exercises

1. **Compare the two versions.** Open `notebooks/explore_decisions.py` and `src/btitools/decisions.py` side by side. List three differences (state, paths, testability) and note which questions of the case-study notebook each method answers.
2. **Description length by language.** Implement `DecisionTable.median_description_length_by_language` (see the TODO in the code). Delete the `skip` line above `test_median_description_length_by_language` in `tests/test_exercises.py` and run `uv run pytest -q`.
3. **Valid decisions by year.** Implement `valid_share_by_year` and activate its test in the same way. Why are almost only the decisions of 2023 still valid? (A BTI decision is valid for three years.)
4. **Complete the summary.** Add both results to `summary()`, activate the last test and run `decision-summary` on the sample. Compare the numbers with your answers in the case-study notebook.
5. **Stretch.** Add a method `share_with_code()` (share of descriptions that contain the placeholder `<CODE>`), add `keywords` as an optional column, and write a test with a value you computed by hand from `tests/conftest.py`.

`solutions/decisions.py` contains a reference solution for self-checking after you have tried the exercises.
