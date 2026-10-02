# Workspace: the tariff heading service

A complete, small service around the case-study model: training script, FastAPI prediction API with
pydantic schemas that returns the **three highest-scoring HS headings** with their English texts, tests with
`TestClient`, a Dockerfile, CI and CD workflows for GitHub Actions, drift statistics as tested pure functions,
a monitoring report and a model card template. Session 16 builds and releases version 1 and then, after the
2024 feedback labels are released, version 2. Copy the folder into your team repository.

```
src/tariff_service/
  model.py        pipeline, top-3 scores, pruning, evaluation, save/load with metadata and checksum, fixture model
  train.py        CLI: train a version on the case-study data (time-based validation)
  schemas.py      pydantic request/response models
  app.py          FastAPI app: /health, /metadata, /predict, /predict/batch (exercise 2)
  monitoring.py   prediction log (JSON lines, no description text)
  drift.py        KS test, PSI, category shift, unseen headings, retraining rule, OOV rate (exercise 4)
  monitor.py      CLI: drift report of new decisions against the model's reference statistics
  config.py       settings from environment variables (MODEL_DIR, PREDICTION_LOG)
tests/            pytest: API contract, validation, behaviour, model files, drift statistics
dashboard/        Streamlit demo app
Dockerfile        container image; .dockerignore keeps data and tests out of it
.github/workflows ci.yml (lint, tests, container smoke test), cd.yml (release image on version tags)
MODEL_CARD.md     template after Mitchell et al. (2019)
uv.lock           pinned versions of all dependencies
solutions/        reference solutions of the exercises, for self-checking only
```

## Setup and first run

```bash
cd sessions/16-deployment-and-monitoring/workspace
uv sync                                   # creates .venv with exactly the versions in uv.lock
uv run pytest -q                          # 34 passed, 2 xfailed (open exercises); with --extra skops: 35 passed
uvx ruff check .

# version 1 on the 50,000-decision sample (about 20 s; validation accuracy about 0.78, top-3 0.85)
uv run python -m tariff_service.train --data ../../../case-study/data/train_sample.parquet --version 1.0.0

# start the API (Ctrl+C stops it) and open http://localhost:8000/docs
uv run uvicorn tariff_service.app:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"description": "Stiefel, mit Laufsohlen aus Kautschuk, mit Oberteil aus Leder (Rindleder)", "language": "de"}'
```

The tests never need the case-study data: `tests/conftest.py` trains a model on 16 built-in descriptions.

**What the scores mean.** The model is a linear support vector machine on word TF-IDF features (Session 13).
A score is its margin: above 0 the model votes for the heading, below 0 it is not convinced of any heading.
Scores are not probabilities. For the boot request above, version 1 answers 6404 with score −0.28 and has
the correct heading 6403 only in third place (−0.68): short descriptions, unlike the long descriptions of real
decisions, give the model little to go on. A service would show such low-score answers as suggestions only and
route them to an officer (Session 8, abstention).

**Model size.** With about 900 headings and 80,000 words, the dense coefficient matrix takes about 600 MB.
`train.py` sets coefficients with |w| < 0.05 to zero and stores them sparsely (`--prune`): about 1 % of the
weights remain, validation accuracy is unchanged, and the model file has about 12 MB.

## Exercises

1. **Version 1 of the service (block 1).** Train v1, start the API and send five descriptions through `/docs`,
   at least one long description copied from `test.parquet`. Then save the model with skops instead of joblib
   (`uv sync --extra skops`, `--format skops`) and explain in two sentences why a skops file is safer to load than
   a pickle from an unknown source.
2. **Batch endpoint (block 1).** Implement `/predict/batch` (TODO in `app.py`). The tests in
   `tests/test_exercises.py` turn from `xfail` into `XPASS`. Add one test of your own: a batch with one
   invalid decision must be rejected with 422.
3. **Container and CI/CD (block 2).** Read the `Dockerfile` line by line and explain each layer. If Docker
   is installed: `docker build -t tariff-service:1.0.0 .` and `docker run --rm -p 8000:8000
   tariff-service:1.0.0`. Put the workspace into a GitHub repository, push, and check that `ci.yml`
   passes. Create the release `v1.0.0` with the model files attached and follow `cd.yml`.
4. **Monitoring (block 3).** Implement `oov_rate` (TODO in `drift.py`). Run the drift report on the 2024–2026
   test decisions: `uv run python -m tariff_service.monitor --current ../../../case-study/data/test.parquet`,
   then with the feedback labels (`--labels ../../../case-study/data/feedback_2024.csv`). Which input drifts?
   Does the retraining rule fire?
5. **Version 2 and the model card (block 3).** Retrain with the feedback
   (`--feedback ../../../case-study/data/feedback_2024.csv --version 2.0.0`), compare v1 and v2 as in
   the workbook `01-case-study-drift-and-retraining.ipynb`, and fill in `MODEL_CARD.md` for v2.

With the course data, the report for v1 on the 2024 decisions shows: country PSI 0.24 (GB issued about 4 % of
the training decisions and none since 2021), language PSI 0.09, description-length PSI 0.04, predicted-chapter
PSI 0.06, chapter label shift PSI 0.06 (chapter 85 falls from 14.7 % to 11.7 %), 0.4 % of the 2024 headings
never seen in training, and accuracy 0.80 on 2024 against 0.78 in validation. The written rule does **not**
fire, yet retraining with the 2024 labels still raises accuracy on 2025–2026 (see the workbook): a rule that
waits for a drop can miss an opportunity to improve.

## Containers, CI/CD and hosting

- **Docker.** The image is built in layers: dependencies from `uv.lock` first (cached), then the code,
  then the model. It runs as a non-root user and has a health check. Docker Desktop is free for education
  and small companies; [Podman](https://podman.io/) is an open-source alternative with the same commands.
- **CI** (`ci.yml`): on every push and pull request, ruff and pytest run on a clean machine; then the image
  is built with the fixture model and smoke-tested with `curl`.
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

The dashboard in `dashboard/streamlit_app.py` loads the model with `load_model` and shows the top-3 headings
and the metadata: `uv sync --extra dashboard && MODEL_DIR=models uv run streamlit run dashboard/streamlit_app.py`.

## Design notes

- **One preprocessing function** (`decision_text`) is used by training, the API, the dashboard and the
  monitor, so that serving cannot drift from training (training–serving skew).
- **Only the description is an input.** The justification, keywords and CN code are decided together with
  the heading and are not available for a new request; the schema forbids extra fields such as `heading`.
- **Metadata travel with the model**: version, library versions, training data checksum, validation
  scores, the reference statistics that monitoring compares with, and the English heading texts. `load_model`
  refuses a model file whose checksum does not match.
- **The log stores no description**, only its length, language, the top heading and its score: a pending BTI
  request is confidential business information.
- **Configuration by environment variables** (`MODEL_DIR`, `PREDICTION_LOG`), as in the Twelve-Factor App.

## Further reading

- FastAPI documentation, [Tutorial](https://fastapi.tiangolo.com/tutorial/) and [FastAPI in containers](https://fastapi.tiangolo.com/deployment/docker/) (MIT).
- Goku Mohandas, [Made With ML](https://madewithml.com/) (MIT): one project from design to CI/CD and serving.
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAT\* 2019*. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
