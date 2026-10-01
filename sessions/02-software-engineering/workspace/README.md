# Workspace · Session 2: software engineering for data work

One small project that grows through the three blocks of Session 2: a validated review record (block 1), a client for a public open-data API and a minimal web API (block 2), and a module of text features that you contribute through a reviewed pull request with continuous integration (block 3). It uses the same package name, `reviewtools`, as the Session 1 workspace; in a team repository both parts can live in one package.

```
workspace/
├── pyproject.toml             dependencies (httpx, fastapi, uvicorn), dev tools (pytest, ruff)
├── uv.lock                    exact versions; CI installs from it
├── .python-version            Python version used by uv and CI
├── .github/workflows/ci.yml   GitHub Actions: ruff and pytest on every pull request
├── src/reviewtools/
│   ├── errors.py              exception hierarchy (ReviewToolsError, ValidationError, APIError, ...)
│   ├── records.py             ReviewRecord: a dataclass that validates and cleans itself   (block 1)
│   ├── opendata.py            OpenDataClient and DatasetRecord for CKAN portals          (block 2)
│   ├── api.py                 FastAPI app: /health, /labels, /reviews/validate, /features (block 2)
│   └── text_features.py       n_words; stubs for the pull-request exercise               (block 3)
├── tests/                     pytest tests; HTTP is mocked, so they run offline
└── solutions/                 reference solutions, for self-checking only
```

Theory: [01 OOP, errors and tests](../theory/01-oop-errors-and-tests.md) · [02 web APIs](../theory/02-web-apis.md) · [03 Git, GitHub and CI](../theory/03-git-github-and-ci.md).

## Run it

```bash
cd sessions/02-software-engineering/workspace
uv run pytest -q                      # 52 passed, 18 skipped (the skipped ones are your exercises)
uvx ruff check . && uvx ruff format --check .

uv run opendata-search baum           # live request to GovData (Berlin datasets); needs internet
uv run uvicorn reviewtools.api:app --reload   # web API on http://127.0.0.1:8000/docs, stop with Ctrl+C
```

The open-data client uses the CKAN API of [GovData](https://www.govdata.de/), the German national open-data portal, filtered to datasets published by Berlin (organisation `berlin-open-data`). No key is needed. The Berlin portal itself (`daten.berlin.de`) runs the same software, but currently shows a browser check to scripts, so the client does not use it by default.

## Exercises

**Block 1 · Classes, errors and tests**

1. **Validate `helpful_vote`.** In `src/reviewtools/records.py`, complete the TODO in `ReviewRecord.__post_init__`: `helpful_vote` must be a whole number ≥ 0 (accept `3` and `"3"`, reject `-1`, `2.5`, `"many"` and `True`). Delete the `skip` line of `test_helpful_vote_rules` in `tests/test_records.py` and run `uv run pytest -q tests/test_records.py`.
2. **Write your own tests.** Add tests to `tests/test_records.py` for two rules that are not tested yet: a title that consists only of spaces becomes `""`, and `verified_purchase="N"` becomes `False`. Write each test first, run it, and check that it would fail if the rule were wrong (change the expected value once).

**Block 2 · Web APIs**

3. **Look at the raw response.** In a Python console (`uv run python`), request `https://ckan.govdata.de/api/3/action/package_search?q=baum&rows=2&fq=organization:berlin-open-data` with `httpx.get(...)`. Print `status_code`, `headers["content-type"]` and the keys of `response.json()["result"]`. Which field gives the total number of matches? Which fields does `DatasetRecord.from_ckan` use?
4. **Pagination.** Implement `OpenDataClient.iter_datasets` (TODO in `opendata.py`): request page after page with `start = 0, page_size, 2 * page_size, ...` until `max_records` records are yielded or the results end. Activate `test_iter_datasets_follows_pages` in `tests/test_opendata.py`. The test uses `httpx.MockTransport`, so it needs no network. Then try it live: how many Berlin datasets mention *Radverkehr*?
5. **The web API.** Start the app with uvicorn and open `/docs`. Send a valid and an invalid review to `POST /reviews/validate` from the browser. Then add a test to `tests/test_api.py` that posts a review without `text` and checks the status code and the reported field.

**Block 3 · Git, pull requests and CI** (work in pairs; see the [merge-conflict exercise](../theory/03-git-github-and-ci.md#exercise-a-merge-conflict-step-by-step))

6. **Pull request.** Copy the workspace into a new GitHub repository (or your team repository) and push `main`. Person A creates the branch `feature/n-exclamations`, person B the branch `feature/count-negations`. Each implements one function in `src/reviewtools/text_features.py`, adds its name to `FEATURE_NAMES` at the top of the file, deletes the `skip` line of its test in `tests/test_text_features.py`, adds one test of their own, commits, pushes and opens a pull request. The partner reviews it (at least one comment on a line), the author answers or fixes, and the reviewer approves. Merge when CI is green. Because both pull requests change the `FEATURE_NAMES` line, the second one will have a merge conflict: resolve it as practised, keeping both names.
7. **Use the merged features.** On a new branch, add `n_exclamations` and `count_negations` to the `/features` endpoint in `api.py` and to `tests/test_api.py`. Open a pull request.
8. **Team repository.** Set up the team repository from the template with `.github/workflows/ci.yml` and a ruleset for `main`: pull request required, one approval, status check `test` required, no force pushes. Make one deliberately failing pull request (for example an unused import) and check that it cannot be merged.

`solutions/` contains reference versions of `records.py`, `opendata.py` and `text_features.py` for self-checking after you have tried the exercises.
