# Session 16 · Web APIs, deployment, monitoring and maintenance

> **Guiding question.** How does a model or dashboard reach its users, and how does it stay useful?

## 1 · Web APIs and a released service

[Open notebook](workbooks/01-lecture-web-apis-and-a-released-service.ipynb)

- How web APIs work: requests, responses, status codes, JSON, authentication, pagination, rate limits, retries.
- Requesting Berlin weather from Open-Meteo with httpx: failing gracefully, a client class with validated records, ten years in yearly requests, nested JSON, and testing with a mock server.
- From notebook to service: the fitted pipeline saved with joblib and skops, versions, checksums, `uv.lock`, configuration from environment variables.
- A prediction API with FastAPI and pydantic: validation, start-up loading, generated documentation, tests with `TestClient`, time per request.
- Containers with Docker: the workspace Dockerfile, layers and the build cache, building and running the image.
- Continuous delivery with GitHub Actions: workflows, CI on every push, a release on a version tag with a version check.
- Worked example: the service answers real 2026 requests in five languages.

## 2 · Architecture in the cloud, monitoring and maintenance

[Open notebook](workbooks/02-lecture-architecture-monitoring-and-maintenance.ipynb)

- Setup: version 1 of the model trained before a simulated release date, the requests of 2026 as production data.
- System design: front end, back end and database; Docker Compose; a prediction log in SQLite; publishing a dashboard.
- Cloud services on AWS: ECR, ECS on Fargate, load balancer, RDS, S3, CloudWatch (introduction).
- Data drift: the KS test, the population stability index and its binning, the reference period, drift reports, text drift and unseen words.
- Prediction drift, label shift and concept drift with the released labels of the Public half of the test set.
- Retraining triggers, fair comparison of candidates, and versioning.
- Model cards: intended use, results per language, a card generated with skops.
- Worked example: version 2 and the final leaderboard submission (L3).

## Team project until the final presentations

Service and dashboard run together with Docker Compose and are released through the CI/CD workflow. Release: tested service, monitoring plan and model card; retraining with the released labels and final leaderboard round L3; repository tagged.

## Setup

- The tariff service is a separate uv project in [workspace/](workspace/README.md): `cd workspace && uv sync && uv run pytest -q`. Docker Desktop or Podman is needed only for the Compose part of the team project.
- Block 1 sends live requests to Open-Meteo; without internet it falls back to saved answers.
- In Block 2 the lecturer releases the labels of the Public half of the test set as `public_labels.csv` on the course drive. Put it into `case-study/data/ebti/`; until then the notebook uses half of `val.parquet` as a stand-in ([case-study/README.md](../../case-study/README.md)).
- The Evidently extra needs its own environment: `uv run --no-project --python 3.14 --with jupyterlab --with "evidently>=0.7" --with scikit-learn jupyter lab sessions/16-deployment-and-monitoring/workbooks/extras`.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
