# Session 16 · Web APIs, deployment, monitoring and maintenance

> **Guiding question.** How does a model or dashboard reach its users, and how does it stay useful?

## 1 · Web APIs and a released service

[Open notebook](workbooks/01-lecture-web-apis-and-a-released-service.ipynb)

- How web APIs work: requests, responses, status codes, JSON, authentication, pagination, rate limits, retries.
- Requesting Berlin weather from Open-Meteo with httpx: failing gracefully, a client class with validated records, ten years in yearly requests, nested JSON, and testing with a mock server.
- From notebook to service: the fitted pipeline saved with joblib and skops, versions, checksums, `uv.lock`, configuration from environment variables.
- A prediction API with FastAPI and pydantic: validation, start-up loading, generated documentation, tests with `TestClient`, time per request, and checking the final-project interface with `case-study/evaluate_api.py`.
- Containers with Docker: the workspace Dockerfile, layers and the build cache, building and running the image.
- Continuous delivery with GitHub Actions: workflows, CI on every push, a release on a version tag with a version check.
- Worked example: the service answers newer validation decisions in five languages.

## 2 · Architecture in the cloud, monitoring and maintenance

[Open notebook](workbooks/02-lecture-architecture-monitoring-and-maintenance.ipynb)

- Setup: the validation design, version 1 of the model trained before a simulated release date, the training decisions of 2026 as production requests whose labels arrive later.
- System design: front end, back end and database; Docker Compose; a prediction log in SQLite; publishing a dashboard.
- Cloud services on AWS: ECR, ECS on Fargate, load balancer, RDS, S3, CloudWatch (introduction).
- Data drift: the KS test, the population stability index and its binning, the reference period, drift reports, text drift and unseen words.
- Prediction drift, label shift and concept drift once the labels of 2026 arrive.
- Retraining triggers, fair comparison of candidates, and versioning.
- Model cards: intended use, results per language, a card generated with skops.
- Worked example: version 2, the score a team reports and why only unseen decisions can estimate the independent score.

## Team project until the final presentations

API release: the service answers the interface of the [final project](../17-18-final-presentations/final-project.md) (checked with `case-study/evaluate_api.py`), runs with the dashboard under Docker Compose and is released through the CI/CD workflow, with a monitoring plan and a model card. Retrain on all your decisions, including the newest, and update the reported score. Release a version tag (for example `v2.0.0`) through the CD workflow. At the code freeze, the day before Session 17, set the tag `submission` on `main` ([how to submit](../17-18-final-presentations/final-project.md#how-to-submit-the-submission-tag)).

## Setup

- The tariff service is a separate uv project in [workspace/](workspace/README.md): `cd workspace && uv sync && uv run pytest -q`. Docker Desktop or Podman is needed only for the Compose part of the team project.
- Block 1 sends live requests to Open-Meteo; without internet it uses saved answers.
- The Evidently extra needs its own environment: `uv run --no-project --python 3.14 --with jupyterlab --with "evidently>=0.7" --with scikit-learn jupyter lab sessions/16-deployment-and-monitoring/workbooks/extras`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
