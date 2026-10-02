# Workspace · Session 2: software engineering for data work

One small project that grows through the three blocks of Session 2: a validated record of one Berlin Airbnb listing (block 1), a client for the Open-Meteo weather API and a minimal web API of your own (block 2), and a module of simple features of listing titles that you contribute through a reviewed pull request with continuous integration (block 3). It uses the same package name, `listingtools`, as the Session 1 workspace; in a team repository both parts can live in one package.

```
workspace/
├── pyproject.toml             dependencies (httpx, fastapi, uvicorn), dev tools (pytest, ruff)
├── uv.lock                    exact versions; CI installs from it
├── .python-version            Python version used by uv and CI
├── .github/workflows/ci.yml   GitHub Actions: ruff and pytest on every pull request
├── src/listingtools/
│   ├── errors.py              exception hierarchy (ListingToolsError, ValidationError, APIError, ...)
│   ├── records.py             ListingRecord: a dataclass that validates and cleans itself   (block 1)
│   ├── weather.py             WeatherClient and DailyWeather for the Open-Meteo API         (block 2)
│   ├── api.py                 FastAPI app: /health, /districts, /districts/{district},     (block 2)
│   │                          POST /listings/validate, /features
│   └── text_features.py       n_chars, n_words, has_exclamation, mentions_size; stubs     (block 3)
├── tests/                     pytest tests; HTTP is mocked, so they run offline
└── solutions/                 reference solutions, for self-checking only
```

Theory: [01 OOP, errors and tests](../theory/01-oop-errors-and-tests.md) · [02 web APIs](../theory/02-web-apis.md) · [03 Git, GitHub and CI](../theory/03-git-github-and-ci.md).

## Run it

```bash
cd sessions/02-software-engineering/workspace
uv run pytest -q                      # 94 passed, 21 skipped (the skipped ones are your exercises)
uvx ruff check . && uvx ruff format --check .

uv run berlin-weather 2024-07-01 2024-07-07   # live request to Open-Meteo; needs internet
uv run uvicorn listingtools.api:app --reload  # web API on http://127.0.0.1:8000/docs, stop with Ctrl+C
```

The weather client uses the [Open-Meteo](https://open-meteo.com/) archive and forecast APIs. No key is needed for non-commercial use; the free API allows fewer than 600 calls per minute, 5,000 per hour and 10,000 per day, and a request for more than two weeks of data counts as several calls. The data are licensed CC BY 4.0: name Open-Meteo when you show them.

## Exercises

**Block 1 · Classes, errors and tests**

1. **Validate `maximum_nights`.** In `src/listingtools/records.py`, complete the TODO in `ListingRecord.__post_init__`: `maximum_nights` is optional (`None` stays `None`); the raw file writes 2,147,483,647 when a host set no limit, which also becomes `None`; otherwise it is converted with `parse_int` and must not be smaller than `minimum_nights`. Delete the two `skip` lines of `test_maximum_nights_rules` and `test_maximum_nights_rejects_invalid_values` in `tests/test_records.py` and run `uv run pytest -q tests/test_records.py`.
2. **Write your own tests.** Add tests to `tests/test_records.py` for two rules that are not tested yet: the room type `"  private ROOM "` becomes `"Private room"`, and `minimum_nights=0` is rejected. Write each test first, run it, and check that it would fail if the rule were wrong (change the expected value once).

**Block 2 · Web APIs**

3. **Look at the raw response.** In a Python console (`uv run python`), request `https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2024-07-01&end_date=2024-07-07&daily=temperature_2m_mean,precipitation_sum,sunshine_duration&timezone=Europe%2FBerlin` with `httpx.get(...)`. Print `status_code`, `headers["content-type"]` and the keys of `response.json()["daily"]`. In which unit is the sunshine duration given? Then change the latitude to 152.52: which status code and which message come back?
4. **Paging by year.** Implement `yearly_chunks` and `WeatherClient.history` (TODOs in `weather.py`): split a long period into calendar years and request one year after the other. Activate `test_yearly_chunks` and `test_history_requests_one_year_at_a_time` in `tests/test_weather.py`. The tests use `httpx.MockTransport`, so they need no network. Then try it live for 2016-01-01 to 2026-06-26: how many requests are sent, how many days come back, and how many API calls does Open-Meteo count for them?
5. **The web API.** Start the app with uvicorn and open `/docs`. Look up `/districts/neukölln`, `/districts/atlantis` and `/districts/10115`: what is the difference between the answers 404 and 422? Send a valid and an invalid listing to `POST /listings/validate` from the browser. Then add a test to `tests/test_api.py` that posts a listing without `price` (allowed: the price is optional) and one with `room_type` `"Castle"` (not allowed), and checks the status codes and the reported field.

**Block 3 · Git, pull requests and CI** (work in pairs; see the [merge-conflict exercise](../theory/03-git-github-and-ci.md#exercise-a-merge-conflict-step-by-step))

6. **Pull request.** Copy the workspace into a new GitHub repository (or your team repository) and push `main`. Person A creates the branch `feature/n-digits`, person B the branch `feature/upper-share`. Each implements one function in `src/listingtools/text_features.py`, adds its name to `FEATURE_NAMES` at the top of the file, deletes the `skip` line of its test in `tests/test_text_features.py`, adds one test of their own, commits, pushes and opens a pull request. The partner reviews it (at least one comment on a line), the author answers or fixes, and the reviewer approves. Merge when CI is green. Because both pull requests change the `FEATURE_NAMES` line, the second one will have a merge conflict: resolve it as practised, keeping both names.
7. **Use the merged features.** On a new branch, add `n_digits` and `upper_share` to the `/features` endpoint in `api.py` and to `tests/test_api.py`. Open a pull request.
8. **Team repository.** Set up the team repository from the template with `.github/workflows/ci.yml` and a ruleset for `main`: pull request required, one approval, status check `test` required, no force pushes. Make one deliberately failing pull request (for example an unused import) and check that it cannot be merged.

`solutions/` contains reference versions of `records.py`, `weather.py` and `text_features.py` for self-checking after you have tried the exercises.
