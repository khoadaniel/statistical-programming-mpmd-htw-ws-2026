# Session 16 · Web APIs, deployment, monitoring and maintenance

> **Guiding question.** How does a model or dashboard reach its users, and how does it stay useful?

## 1 · Web APIs and a released service

[Open notebook](workbooks/01-lecture-web-apis-and-a-released-service.ipynb)

- How web APIs work: requests, responses, status codes, JSON, authentication, pagination, rate limits, retries.
- Requesting Berlin weather from Open-Meteo with httpx, failing gracefully, and testing with a mock server.
- From notebook to service: the fitted pipeline saved with joblib and skops, versions, checksums, `uv.lock`.
- A prediction API with FastAPI and pydantic: validation, generated documentation, tests with `TestClient`.
- Containers with Docker: the workspace Dockerfile, layers and the build cache.
- Continuous delivery with GitHub Actions: CI on every push, a release on a version tag.

## 2 · Architecture in the cloud, monitoring and maintenance

[Open notebook](workbooks/02-lecture-architecture-monitoring-and-maintenance.ipynb)

- System design: front end, back end and database; Docker Compose; a prediction log in SQLite.
- Cloud services on AWS: ECR, ECS on Fargate, load balancer, RDS, S3, CloudWatch (introduction).
- Data drift: the KS test, the population stability index, drift reports and text drift.
- Prediction drift and label shift with the released 2024 labels.
- Retraining triggers, fair comparison of candidates, and versioning.
- Model cards: intended use, results per language, a card generated with skops.
- Version 2 and the final leaderboard submission (L3).

## Team project until the final presentations

Service and dashboard run together with Docker Compose and are released through the CI/CD workflow. Release: tested service, monitoring plan and model card; retraining with the 2024 labels and final leaderboard round L3; repository tagged.

## Setup

- The tariff service is a separate uv project in [workspace/](workspace/README.md): `cd workspace && uv sync && uv run pytest -q`. Docker Desktop or Podman is needed only for the Compose part of the team project.
- Block 1 sends live requests to Open-Meteo; without internet it falls back to saved answers.
- Before Block 2 the lecturer releases the 2024 labels: `uv run python sessions/16-deployment-and-monitoring/workbooks/make_feedback_2024.py` writes `case-study/data/feedback_2024.csv` (or `--out PATH`; the notebook also reads `FEEDBACK_PATH`).
- The Evidently extra needs its own environment: `uv run --no-project --python 3.14 --with jupyterlab --with "evidently>=0.7" --with scikit-learn jupyter lab sessions/16-deployment-and-monitoring/workbooks/extras`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
