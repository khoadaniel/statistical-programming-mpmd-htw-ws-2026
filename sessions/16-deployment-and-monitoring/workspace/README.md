# Workspace: the tariff heading service

A complete, small service around the case-study model: training script, FastAPI prediction API with
pydantic schemas that returns the **three highest-scoring HS headings** with their English texts, tests with
`TestClient`, a Dockerfile, a Compose file that runs dashboard and API together (with the prediction log in a
SQLite file), CI and CD workflows for GitHub Actions, drift statistics as tested pure functions, a monitoring
report and a model card template. The API answers the interface of the
[final project](../../17-18-final-presentations/final-project.md): `POST /predict` with `description` (the only input the
evaluation sends; an optional `language` is logged for monitoring) returns `heading` and `top`, `GET /health` answers `{"status": "ok", ...}`, port 8000, started with
`docker compose up`. Session 16 builds version 1, trained on the decisions before 2026, and then version 2, trained on all decisions. Copy the
folder into your team repository: running service and dashboard with Docker Compose and releasing through CI/CD is
part of the team project.

```
src/tariff_service/
  model.py        pipeline, top-3 scores, pruning, evaluation, save/load with metadata and checksum, fixture model
  train.py        CLI: train a version on the case-study data (time-based validation)
  schemas.py      pydantic request/response models
  app.py          FastAPI app: /health, /metadata, /predict, /predict/batch (exercise 2)
  monitoring.py   prediction log (JSON lines or a SQLite table, no description text)
  drift.py        KS test, PSI, category shift, unseen headings, retraining rule, OOV rate (exercise 4)
  monitor.py      CLI: drift report of new decisions against the model's reference statistics
  config.py       settings from environment variables (MODEL_DIR, PREDICTION_LOG, PREDICTION_DB)
tests/            pytest: API contract, validation, behaviour, model files, drift statistics
dashboard/        Streamlit app (front end) and its Dockerfile
Dockerfile        container image of the API; .dockerignore keeps data and tests out of it
compose.yaml      front end and back end in two containers; the prediction log in SQLite on a volume
.github/workflows ci.yml (lint, tests, container smoke test), cd.yml (release image on version tags)
MODEL_CARD.md     template after Mitchell et al. (2019)
uv.lock           pinned versions of all dependencies
solutions/        reference solutions of the exercises, for self-checking only
```

## Setup and first run

```bash
cd sessions/16-deployment-and-monitoring/workspace
uv sync                                   # creates .venv with exactly the versions in uv.lock
uv run pytest -q                          # 35 passed, 1 skipped (skops not installed), 2 xfailed (open exercises)
uvx ruff check .

# version 1 on all 103,656 decisions (a few minutes; accuracy 0.789, top-3 0.860 on the newest 20 % of them)
uv run python -m tariff_service.train --data ../../../case-study/data/ebti/decisions.parquet --version 1.0.0

# start the API (Ctrl+C stops it) and open http://localhost:8000/docs
uv run uvicorn tariff_service.app:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"description": "Stiefel, mit Laufsohlen aus Kautschuk, mit Oberteil aus Leder (Rindleder)"}'

# in a second terminal, from the repository root: the check the lecturer runs after the code freeze,
# here on your own validation decisions (a Parquet file with description and heading)
uv run python case-study/evaluate_api.py --url http://localhost:8000 --data my_validation.parquet
```

The tests never need the case-study data: `tests/conftest.py` trains a model on 16 built-in descriptions.

**What the scores mean.** The model is a linear support vector machine on word TF-IDF features (Session 13).
A score is its margin: above 0 the model votes for the heading, below 0 it is not convinced of any heading.
Scores are not probabilities. For the boot request above, version 1 of the first run (trained on all decisions)
answers 6404 with score 0.46, and the correct heading 6403 is not among the three suggestions. Short descriptions,
unlike the long descriptions of real decisions, give the model little information. A service would show such low-score answers as suggestions only and
route them to an officer (Session 14, abstention).

**Model size.** With about 1,000 headings and 158,000 words, the dense coefficient matrix takes about 1.3 GB.
`train.py` sets coefficients with |w| < 0.05 to zero and stores them sparsely (`--prune`): about 1 % of the
weights remain, validation accuracy is unchanged, and the model file has about 15 MB.

## Workspace exercises

The exercises in class are at the end of the two block notebooks
([Block 1](../workbooks/01-lecture-web-apis-and-a-released-service.ipynb), [Block 2](../workbooks/02-lecture-architecture-monitoring-and-maintenance.ipynb)). The workspace exercises below are longer:
1, 2, 4 and 5 are self-study after Blocks 1 and 2, and 3 is part of the team project until the final presentations.

1. **Version 1 of the service (after Block 1).** Train v1, start the API and send five descriptions through `/docs`,
   at least one long description copied from `decisions.parquet`. Then save the model with skops instead of joblib
   (`uv sync --extra skops`, `--format skops`) and explain in two sentences why a skops file is safer to load than
   a pickle from an unknown source.
2. **Batch endpoint (after Block 1).** Implement `/predict/batch` (TODO in `app.py`). The tests in
   `tests/test_exercises.py` turn from `xfail` into `XPASS`. Add one test of your own: a batch with one
   invalid decision must be rejected with 422.
3. **Container, Compose and CI/CD (team project).** Read the `Dockerfile` line by line and explain each layer. If Docker
   is installed: `docker build -t tariff-service:1.0.0 .` and `docker run --rm -p 8000:8000
   tariff-service:1.0.0`. Then start dashboard and API with `docker compose up --build`: open the dashboard
   at http://localhost:8501, send three descriptions, and find them in the SQLite log (command in
   "Two containers and a SQLite file" below). Put the workspace into a GitHub
   repository, push, and check that `ci.yml` passes. Create the release `v1.0.0` with the model files attached
   and follow `cd.yml`. Finally, sketch the cloud architecture of your team project on AWS (which part runs on
   which service; see the block notebook [02-lecture-architecture-monitoring-and-maintenance.ipynb](../workbooks/02-lecture-architecture-monitoring-and-maintenance.ipynb)).
4. **Monitoring (after Block 2).** Implement `oov_rate` (TODO in `drift.py`). Simulate a model released at the start
   of 2026, as in the block notebook: `uv run python -m tariff_service.train --data
   ../../../case-study/data/ebti/decisions.parquet --before 2026-01-01 --version 1.0.0`. Run the drift report on
   the decisions of 2026, which v1 never saw, first on their inputs only, as before the labels arrive:
   `uv run python -m tariff_service.monitor --current ../../../case-study/data/ebti/decisions.parquet --since 2026-01-01
   --hide-labels`, then without `--hide-labels`, so that their headings are used. Which input drifts? Is the condition of the
   retraining rule met?
5. **Version 2 and the model card (after Block 2).** Retrain on all decisions, 2026 included
   (`--data ../../../case-study/data/ebti/decisions.parquet --version 2.0.0`), compare v1 and v2 on validation decisions that neither of them saw in training, as in
   the block notebook [02-lecture-architecture-monitoring-and-maintenance.ipynb](../workbooks/02-lecture-architecture-monitoring-and-maintenance.ipynb), check the API with
   `case-study/evaluate_api.py`, update the reported score in your README, and fill in `MODEL_CARD.md` for v2.

With the course data, v1 of exercise 4, trained with `--before 2026-01-01` on all decisions before 2026, has a
time-based validation accuracy of 0.8025 on the newest 20 % of those decisions. (The 0.789 of the first run above
belongs to a different model: v1 trained on all decisions and validated on the newest 20 % of all of them, 2026
included.) The report on the 27,744 decisions of 2026 shows: description-length PSI 0.002, language PSI 0.033 and
country PSI 0.035 (France's share rises), predicted-chapter PSI 0.049, chapter label shift PSI 0.046, 0.2 % of the
2026 headings never seen in training, and accuracy 0.7801 against 0.8025 in validation. The written rule (accuracy
more than 0.03 below validation, or an input PSI above 0.25) is not met, but the margin is small: the drop is 0.022. No input
PSI comes near 0.25, so the clearest warning sign comes from the labels, not from the input statistics. The block
notebook shows that retraining with the labels of 2026 raises accuracy on newer decisions.

## Containers, CI/CD and hosting

- **Docker.** The image is built in layers: dependencies from `uv.lock` first (cached), then the code,
  then the model. It runs as a non-root user and has a health check. Docker Desktop is free for education
  and small companies; [Podman](https://podman.io/) is an open-source alternative with the same commands.
- **CI** (`ci.yml`): on every push and pull request, ruff and pytest run on a clean machine; then the image
  is built with the fixture model and smoke-tested with `curl` (a smoke test is a quick check that the service
  starts and answers one request).
- **CD** (`cd.yml`): pushing a tag such as `v1.0.0` downloads the model files attached to the GitHub
  release of that tag, checks that the version in `metadata.json` matches the tag, and pushes the image to
  the GitHub container registry (`ghcr.io`). A hosting platform then runs that image.
- GitHub reads workflows only from `.github/workflows/` at the **repository root**. Inside the course
  repository they are examples; in your team repository, put them at the root.

### Publishing a dashboard or the API

| Option | What it runs | Free tier (check current terms) | Notes |
|---|---|---|---|
| [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) | Streamlit apps from a public GitHub repository | yes | simplest for a dashboard; model files must be in the repository or downloaded at start |
| [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview) | Gradio, Streamlit, static or Docker apps | limited | free hardware for some app types only; good for model demos |
| [Render](https://render.com/docs/free) | Docker images and web services | yes, sleeps when idle | can deploy the API container from a registry |
| [Google Cloud Run](https://cloud.google.com/run/pricing) | any container listening on a port | monthly free quota | needs a billing account; scales to zero |
| University or administration server | the container with `docker run` or Podman | n/a | often the right choice for confidential requests |

The dashboard in `dashboard/streamlit_app.py` shows the top-3 headings and the metadata. On its own it loads
the model itself: `uv sync --extra dashboard && MODEL_DIR=models uv run streamlit run dashboard/streamlit_app.py`.
With `API_URL` set, it is only a front end and sends each description to the API.

### Two containers and a SQLite file: front end, back end, database

`compose.yaml` runs the dashboard (port 8501) and the API (port 8000) on one private network. The database is a
SQLite file, `/app/data/predictions.sqlite`, on the volume `prediction-data` that only the API container mounts:
the API writes every prediction to its table `predictions`, because `PREDICTION_DB` is set, and the file is not
reachable from outside. The dashboard calls `http://api:8000`, because `API_URL` is set. Train or download the
model into `models/` first, then:

```bash
docker compose up --build
docker compose exec api python -c "import sqlite3; print(sqlite3.connect('/app/data/predictions.sqlite').execute('SELECT heading, language, model_version FROM predictions').fetchall())"
docker compose down -v            # -v also deletes the volume with the log
```

The test `test_predictions_are_logged_to_sqlite_when_a_database_is_configured` checks the SQLite log without
Docker. One SQLite file serves one API container well; when several copies of the API on different machines must
write to one database, a database server is the better choice ([SQLite: Appropriate uses](https://www.sqlite.org/whentouse.html)).

## Design notes

- **One preprocessing function** (`decision_text`) is used by training, the API, the dashboard and the
  monitor. Serving therefore prepares the text in the same way as training; a difference between the two is
  called training–serving skew.
- **Only the description is an input.** The justification, keywords and CN code are decided together with
  the heading and are not available for a new request; the schema forbids extra fields such as `heading`.
- **Metadata travel with the model**: version, library versions, training data checksum, validation
  scores, the reference statistics that monitoring compares with, and the English heading texts. `load_model`
  refuses a model file whose checksum does not match.
- **The log stores no description**, only its length, language, the top heading and its score: a pending BTI
  request is confidential business information.
- **Configuration by environment variables** (`MODEL_DIR`, `PREDICTION_LOG`, `PREDICTION_DB`), as in the Twelve-Factor App.

## Further reading

- FastAPI documentation, [Tutorial](https://fastapi.tiangolo.com/tutorial/) and [FastAPI in containers](https://fastapi.tiangolo.com/deployment/docker/) (MIT).
- Goku Mohandas, [Made With ML](https://madewithml.com/) (MIT): one project from design to CI/CD and serving.
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAT\* 2019*. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
