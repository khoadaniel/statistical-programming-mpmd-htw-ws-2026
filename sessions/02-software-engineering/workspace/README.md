# Workspace · Session 2: software engineering for data work

One small project that grows through the two blocks of Session 2. In block 1 you build a validated record of one Berlin Airbnb listing. In block 2 the listing analysis of the Session 1 case study moves from notebook cells into a package with a class, `ListingTable`. You also add a module of simple features of listing titles, through a reviewed pull request with continuous integration. The data are the Berlin listings of Inside Airbnb, the course dataset of Sessions 1–12.

```
workspace/
├── pyproject.toml             project description, dependencies (pandas, pyarrow), dev tools (pytest, ruff)
├── uv.lock                    exact versions; CI installs from it
├── .python-version            Python version used by uv and CI
├── .github/workflows/ci.yml   GitHub Actions: ruff and pytest on every pull request
├── notebooks/
│   └── explore_listings.py    the analysis as notebook cells (# %%), the starting point      (block 2)
├── src/listingtools/
│   ├── __init__.py            makes the folder a package; re-exports the main names
│   ├── errors.py              exception hierarchy (ListingToolsError, ValidationError)
│   ├── records.py             ListingRecord: a dataclass that validates and cleans itself      (block 1)
│   ├── listings.py            ListingTable: load and summarise a table of listings            (block 2)
│   ├── text_features.py       n_chars, n_words, has_exclamation, mentions_size; stubs        (block 2)
│   ├── cli.py                 command-line entry point `listing-summary`
│   └── __main__.py            allows `python -m listingtools`
├── tests/                     pytest tests; conftest.py holds the shared test data
└── solutions/                 reference solutions, for self-checking only
```

Teaching notebooks: [block 1, classes, errors and tests](../workbooks/01-lecture-classes-errors-and-tests.ipynb) (exercise 1.6 = workspace exercises 1–2) · [block 2, from notebook to application, Git and CI](../workbooks/02-lecture-notebook-to-application-git-and-ci.ipynb) (exercises 2.2–2.6 = workspace exercises 3–9). The exercises below are also listed there, with their hand-outs.

## Run it

The summary of the real listings needs the case-study data (download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` of the repository, see [case-study/README.md](../../../case-study/README.md)); the tests do not.

```bash
cd sessions/02-software-engineering/workspace
uv run pytest -q                      # 70 passed, 22 skipped (the skipped ones are your exercises)
uv run ruff check . && uv run ruff format --check .

# from the repository root: print the summary of the 10,221 Berlin listings
uv run --project sessions/02-software-engineering/workspace listing-summary
```

In a notebook or the Python console of this environment the classes are imported like any library:

```python
from listingtools import ListingRecord, ListingTable

table = ListingTable.from_parquet("case-study/data/airbnb/listings.parquet")
table.by_district()
table.price_by_room_type()
```

## Exercises

**Block 1 · Classes, errors and tests**

1. **Validate `maximum_nights`.** In `src/listingtools/records.py`, complete the TODO in `ListingRecord.__post_init__`: `maximum_nights` is optional (`None` stays `None`); the raw file writes 2,147,483,647 when a host set no limit, which also becomes `None`; otherwise it is converted with `parse_int` and must not be smaller than `minimum_nights`. Delete the two `skip` lines of `test_maximum_nights_rules` and `test_maximum_nights_rejects_invalid_values` in `tests/test_records.py` and run `uv run pytest -q tests/test_records.py`.
2. **Write your own tests.** Add tests to `tests/test_records.py` for two rules that are not tested yet: the room type `"  private ROOM "` becomes `"Private room"`, and `minimum_nights=0` is rejected. Write each test first, run it, and check that it would fail if the rule were wrong (change the expected value once).

**Block 2 · From notebook to application, Git and CI**

3. **Compare the two versions.** Open `notebooks/explore_listings.py` and `src/listingtools/listings.py` side by side. List three differences (state, paths, testability) and note which of the five questions of the Session 1 case study each method answers.
4. **Price by district.** Implement `ListingTable.median_price_by_district` (see the TODO in the code). Delete the `skip` line above `test_median_price_by_district` in `tests/test_listings_exercises.py` and run `uv run pytest -q`.
5. **Reviews.** Implement `review_summary` and activate its test in the same way. Why is the median number of reviews so much lower than the mean? (Look at the listing with the most reviews.)
6. **Complete the summary.** Add both results to `summary()`, activate the last test and run `listing-summary` on the listings. Compare the numbers with your answers in the Session 1 case study.
7. **Pull request, in pairs.** Copy the workspace into a new GitHub repository (or your team repository) and push `main`. Person A creates the branch `feature/n-digits`, person B the branch `feature/upper-share`. Each implements one function in `src/listingtools/text_features.py`, adds its name to `FEATURE_NAMES` at the top of the file, deletes the `skip` line of its test in `tests/test_text_features.py`, adds one test of their own, commits, pushes and opens a pull request. The partner reviews it (at least one comment on a line), the author answers or fixes, and the reviewer approves. Merge when CI is green. Because both pull requests change the `FEATURE_NAMES` line, the second one will have a merge conflict: resolve it as practised, keeping both names.
8. **Team repository.** Set up the team repository from the template with `.github/workflows/ci.yml` and a ruleset for `main`: pull request required, one approval, status check `test` required, no force pushes. Make one deliberately failing pull request (for example an unused import) and check that it cannot be merged.
9. **Stretch.** Add a method `price_per_guest()` to `ListingTable` that divides the short-stay price by `accommodates` and returns the median per room type. Add `accommodates` to the test data in `tests/conftest.py`, and write a test with a value you computed by hand. Should `accommodates` become a required column?

`solutions/` contains reference versions of `records.py`, `listings.py` and `text_features.py` for self-checking after you have tried the exercises.
