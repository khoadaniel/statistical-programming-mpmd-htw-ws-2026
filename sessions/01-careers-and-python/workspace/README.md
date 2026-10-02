# Workspace · Session 1: from notebook to module

This small project shows the step from notebook code to an importable, tested module. It belongs to block 3 of Session 1 ([theory page](../theory/03-notebook-to-application.md)). The data are the Berlin listings of Inside Airbnb, the course dataset of Sessions 1–12.

```
workspace/
├── pyproject.toml                project description, dependencies, tool settings
├── notebooks/
│   └── explore_listings.py       step 0: the analysis as notebook cells (# %%)
├── src/listingtools/
│   ├── __init__.py               makes the folder a package; re-exports ListingTable
│   ├── listings.py               the class ListingTable (load, summarise)
│   ├── cli.py                    command-line entry point `listing-summary`
│   └── __main__.py               allows `python -m listingtools`
├── tests/
│   ├── conftest.py               eight hand-made listings used by all tests
│   ├── test_listings.py          tests of the finished parts (pass in the starter state)
│   └── test_exercises.py         tests of the exercises (skipped until you do them)
└── solutions/listings.py         reference solution, for self-checking only
```

## Run it

You need the case-study data (`uv run python case-study/prepare_airbnb.py`, run once from the repository root).

```bash
# from this folder: run the tests (uv creates .venv and installs the package)
uv run pytest -q                    # 9 passed, 3 skipped

# from the repository root: print the summary of the 12,776 Berlin listings
uv run --project sessions/01-careers-and-python/workspace listing-summary

# lint and format
uvx ruff check .
uvx ruff format --check .
```

In a notebook or the Python console of this environment the class is imported like any library:

```python
from listingtools import ListingTable

table = ListingTable.from_parquet("case-study/data/airbnb/listings.parquet")
table.by_district()
table.price_by_room_type()
```

## Exercises

1. **Compare the two versions.** Open `notebooks/explore_listings.py` and `src/listingtools/listings.py` side by side. List three differences (state, paths, testability) and note which of the five questions of the case-study notebook each method answers.
2. **Price by district.** Implement `ListingTable.median_price_by_district` (see the TODO in the code). Delete the `skip` line above `test_median_price_by_district` in `tests/test_exercises.py` and run `uv run pytest -q`.
3. **Reviews.** Implement `review_summary` and activate its test in the same way. Why is the median number of reviews so much lower than the mean? (Look at the listing with the most reviews.)
4. **Complete the summary.** Add both results to `summary()`, activate the last test and run `listing-summary` on the listings. Compare the numbers with your answers in the case-study notebook.
5. **Stretch.** Add a method `price_per_guest()` that divides the short-stay price by `accommodates` and returns the median per room type. Add `accommodates` to the test data in `tests/conftest.py`, and write a test with a value you computed by hand. Should `accommodates` become a required column?

`solutions/listings.py` contains a reference solution for self-checking after you have tried the exercises.
