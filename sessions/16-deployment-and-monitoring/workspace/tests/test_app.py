"""Contract, validation and behavioural tests of the API."""

import json

import pytest

BOOTS = {"description": "Damenstiefel mit Oberteil aus Rindleder und Laufsohle aus Gummi", "language": "de"}


def test_health_reports_the_model_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "0.0.0-test"}


def test_predict_contract(client):
    r = client.post("/predict", json=BOOTS)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"heading", "top", "model_version"}
    assert len(body["top"]) == 3 and body["heading"] == body["top"][0]["heading"]
    scores = [t["score"] for t in body["top"]]
    assert scores == sorted(scores, reverse=True)          # margins of the linear SVM, best first
    assert all(len(t["heading"]) == 4 and t["heading"].isdigit() for t in body["top"])
    assert body["top"][0]["heading_description"].startswith(("Footwear", "Textiles", "Tricycles"))


@pytest.mark.parametrize("text, heading", [
    ("Damenstiefel mit Oberteil aus Rindleder und Laufsohle aus Gummi", "6403"),
    ("Voiture jouet en matière plastique pour enfants", "9503"),
    ("Disposable face mask made of nonwoven fabric", "6307"),
])
def test_obvious_descriptions_get_the_obvious_heading(client, text, heading):  # behavioural test
    assert client.post("/predict", json={"description": text}).json()["heading"] == heading


def test_the_right_heading_is_at_least_in_the_top_three(client):
    top = client.post("/predict", json={"description": "Hausschuhe aus Textil"}).json()["top"]
    assert "6404" in [t["heading"] for t in top]


@pytest.mark.parametrize("payload", [
    {"description": ""},                               # empty description
    {"description": "   "},                            # only spaces
    {"text": "Schuhe"},                                # wrong field name
    {"description": 42},                               # wrong type
    {"description": "Schuhe", "heading": "6403"},      # unknown field (the label is never an input)
    {"description": "Schuhe", "language": "deu"},      # language must have two letters
    {"description": "x" * 20_001},                     # too long
])
def test_invalid_input_is_rejected_with_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_metadata_endpoint(client):
    meta = client.get("/metadata").json()
    assert meta["model_version"] == "0.0.0-test" and meta["n_classes"] == 4
    assert {"accuracy", "top3_accuracy"} <= set(meta["validation"])


def test_openapi_documents_the_schemas(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert {"DecisionIn", "PredictionOut", "HeadingScore"} <= set(schema["components"]["schemas"])


def test_predictions_are_logged_without_the_text(client, tmp_path):
    client.post("/predict", json=BOOTS)
    record = json.loads((tmp_path / "predictions.jsonl").read_text().splitlines()[-1])
    assert record["model_version"] == "0.0.0-test" and record["heading"] == "6403"
    assert record["text_length"] == len(BOOTS["description"]) and record["language"] == "de"
    assert "description" not in record and "text" not in record


def test_predictions_are_logged_to_postgres_when_a_database_is_configured(model_dir, monkeypatch):
    """Runs only with a PostgreSQL server: TEST_DATABASE_URL=postgresql://... uv run --extra postgres pytest"""
    import os

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL to run this test against PostgreSQL")
    psycopg = pytest.importorskip("psycopg")
    from fastapi.testclient import TestClient

    from tariff_service.app import create_app

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("PREDICTION_LOG", raising=False)
    with psycopg.connect(url) as conn:
        conn.execute("DROP TABLE IF EXISTS predictions")
    with TestClient(create_app(model_dir=model_dir)) as c:
        heading = c.post("/predict", json=BOOTS).json()["heading"]
    with psycopg.connect(url) as conn:
        rows = conn.execute("SELECT heading, language, model_version FROM predictions").fetchall()
    assert rows == [(heading, "de", "0.0.0-test")]
