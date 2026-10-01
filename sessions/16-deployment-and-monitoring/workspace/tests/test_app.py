"""Contract, validation and behavioural tests of the API."""

import json

import pytest

NEGATIVE = {"text": "Broke after two days. Waste of money."}


def test_health_reports_the_model_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "0.0.0-test"}


def test_predict_contract(client):
    r = client.post("/predict", json=NEGATIVE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "probabilities", "model_version"}
    assert body["label"] in {"neg", "neu", "pos"}
    assert set(body["probabilities"]) == {"neg", "neu", "pos"}
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01
    assert body["label"] == max(body["probabilities"], key=body["probabilities"].get)


@pytest.mark.parametrize("text, label", [
    ("Broke after two days. Waste of money.", "neg"),
    ("Excellent product, works great and I love it.", "pos"),
])
def test_obvious_reviews_get_the_obvious_label(client, text, label):  # behavioural test
    assert client.post("/predict", json={"text": text}).json()["label"] == label


@pytest.mark.parametrize("payload", [
    {"text": ""},                         # empty text
    {"text": "   "},                      # only spaces
    {"review": "Great product"},          # wrong field name
    {"text": 42},                         # wrong type
    {"text": "ok", "rating": 5},          # unknown field
    {"text": "x" * 10_001},               # too long
])
def test_invalid_input_is_rejected_with_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_metadata_endpoint(client):
    meta = client.get("/metadata").json()
    assert meta["model_version"] == "0.0.0-test"
    assert meta["classes"] == ["neg", "neu", "pos"]
    assert "macro_f1" in meta["validation"]


def test_openapi_documents_the_schemas(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert {"ReviewIn", "PredictionOut"} <= set(schema["components"]["schemas"])


def test_predictions_are_logged_without_the_text(client, tmp_path):
    client.post("/predict", json=NEGATIVE)
    lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    record = json.loads(lines[-1])
    assert record["model_version"] == "0.0.0-test" and record["label"] in {"neg", "neu", "pos"}
    assert record["text_length"] == len(NEGATIVE["text"])
    assert "text" not in record
