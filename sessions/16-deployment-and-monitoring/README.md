# Session 16 · Deployment, monitoring and maintenance

> [!NOTE]
> **Guiding question.** How does a model or dashboard reach its users, and how does it stay useful?

**Learning outcomes.** Students are able to

- serve a model as a tested web API in a container, released with CI/CD
- monitor a deployed model for data drift and label shift
- decide on retraining and document a model in a model card

## Session plan

**0:00–0:45 · From notebook to service** ([theory](theory/01-from-notebook-to-service.md))

- From notebook to service: [saving pipelines](theory/01-from-notebook-to-service.md#from-notebook-to-artefact-saving-pipelines), [versioning](theory/01-from-notebook-to-service.md#versioning), [pinned dependencies](theory/01-from-notebook-to-service.md#pinned-dependencies)
- [The prediction API with FastAPI and pydantic](theory/01-from-notebook-to-service.md#the-prediction-api-with-fastapi-and-pydantic), building on Session 2
- [Tests for the service](theory/01-from-notebook-to-service.md#tests-for-the-service)

*Practice:* Build and test version 1 of the tariff heading service (top-3 headings with scores and English texts) → [`workspace/`](workspace/README.md), exercises 1–2

**1:00–1:45 · Containers and continuous delivery** ([theory](theory/02-containers-and-continuous-delivery.md))

- [Containers with Docker](theory/02-containers-and-continuous-delivery.md#containers-with-docker)
- [Continuous delivery with GitHub Actions](theory/02-containers-and-continuous-delivery.md#continuous-delivery-with-github-actions)
- [Publishing a dashboard](theory/02-containers-and-continuous-delivery.md#publishing-a-dashboard)

*Practice:* Release the service in a container through the CI/CD workflow → [`workspace/`](workspace/README.md), exercise 3

**2:00–2:45 · Monitoring and maintenance** ([theory](theory/03-monitoring-and-maintenance.md))

- Monitoring: [data drift (KS test, population stability index)](theory/03-monitoring-and-maintenance.md#data-drift-the-ks-test-and-the-population-stability-index), [label shift](theory/03-monitoring-and-maintenance.md#prediction-drift-and-label-shift)
- [Retraining triggers and versioning](theory/03-monitoring-and-maintenance.md#retraining-triggers-and-versioning)
- [Documenting a model in a model card](theory/03-monitoring-and-maintenance.md#documenting-a-model-in-a-model-card)

*Practice:* Case study: the 2024 labels are released as feedback data; detect the shifts (no GB decisions after Brexit, chapter 85 from 14 % to 12 %, a few headings never seen), compare retraining candidates, retrain and make the final leaderboard submission (L4), ranked on the 2025–2026 decisions → [`01-case-study-drift-and-retraining.ipynb`](workbooks/01-case-study-drift-and-retraining.ipynb), then workspace exercises 4–5

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-from-notebook-to-service.md](theory/01-from-notebook-to-service.md) | Saving pipelines (joblib, skops), versioning, pinned dependencies, FastAPI and pydantic, tests | 1 | core |
| [theory/02-containers-and-continuous-delivery.md](theory/02-containers-and-continuous-delivery.md) | Docker, GitHub Actions CI/CD, publishing a dashboard | 2 | core |
| [theory/03-monitoring-and-maintenance.md](theory/03-monitoring-and-maintenance.md) | KS test, PSI, prediction drift, label shift, nomenclature change, retraining, model card | 3 | core |
| [workspace/](workspace/README.md) | Complete tariff heading service (package `tariff_service`): train script, FastAPI app, tests, Dockerfile, `ci.yml`, `cd.yml`, drift module, model card template, dashboard | 1–3 | core |
| [workspace/MODEL_CARD.md](workspace/MODEL_CARD.md) | Model card template (Mitchell et al. 2019) | 3 | core |
| [workbooks/01-case-study-drift-and-retraining.ipynb](workbooks/01-case-study-drift-and-retraining.ipynb) | Own: input and prediction drift, label shift with the 2024 feedback, accuracy by language, retraining candidates, v2, leaderboard round L4 | 3 | core |
| [workbooks/02-evidently-data-drift-report.ipynb](workbooks/02-evidently-data-drift-report.ipynb) | Evidently: data drift and data summary reports on a tabular dataset | 3 | optional |
| [workbooks/make_feedback_2024.py](workbooks/make_feedback_2024.py) | Lecturer only: writes `case-study/data/feedback_2024.csv` (labels of the 2024 decisions) from the hidden solution | 3 | lecturer |

Sources and licences: [source.md](source.md).

## Before and after the session

**Preparation.** Re-read the FastAPI and CI parts of Session 2. Run `cd workspace && uv sync && uv run pytest -q` once, so that all packages are installed before class. Docker Desktop (or Podman) is useful but not required: the tests and the notebook do not need it.

**Lecturer.** Before block 3, run `uv run python sessions/16-deployment-and-monitoring/workbooks/make_feedback_2024.py` and share `case-study/data/feedback_2024.csv` (labels of the 40,369 decisions of 2024 only, columns `id,heading`) with the students. `--out PATH` writes the file elsewhere.

**Team project until the next session.** Release: dashboard or deployed model, repository and documentation complete.

**Further reading (optional, free)**

- FastAPI: [Tutorial – User Guide](https://fastapi.tiangolo.com/tutorial/) and [FastAPI in Containers](https://fastapi.tiangolo.com/deployment/docker/)
- Goku Mohandas: [Made With ML](https://madewithml.com/) (testing, CI/CD and serving of one project)
- Christian Kästner: [Machine Learning in Production](https://mlip-cmu.github.io/book/) (open textbook)
- Mitchell et al. (2019): [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- Evidently: [ML observability course](https://www.evidentlyai.com/ml-observability-course)

## Setup

The workspace has its own locked environment: `cd workspace && uv sync` (FastAPI, uvicorn, pydantic, scikit-learn, joblib, scipy, pandas; dev: pytest, ruff, httpx2). Optional extras: `--extra skops`, `--extra dashboard` (Streamlit).

The notebook runs from the repository root:

```bash
uv run --with jupyterlab --with pandas --with pyarrow --with scikit-learn --with scipy --with matplotlib jupyter lab
```

It writes `submission_L4.csv` next to itself (do not commit it) and reads the feedback file from `case-study/data/feedback_2024.csv` or from the path in `FEEDBACK_PATH`. Run time about two minutes. The Evidently notebook needs `--with evidently` and downloads the Adult dataset from OpenML.
