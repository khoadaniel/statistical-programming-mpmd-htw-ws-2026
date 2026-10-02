"""Tests of the FastAPI app with TestClient: requests without starting a server."""

import pytest
from fastapi.testclient import TestClient

from btitools.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_headings_lists_the_extract():
    assert "9503" in client.get("/headings").json()["headings"]


@pytest.mark.parametrize(("heading", "chapter"), [("0901", "09"), ("6404", "64"), ("9503", "95")])
def test_heading_details(heading, chapter):
    body = client.get(f"/headings/{heading}").json()
    assert body["heading"] == heading
    assert body["chapter"] == chapter
    assert body["description"]


def test_unknown_heading_is_404():
    assert client.get("/headings/9999").status_code == 404


@pytest.mark.parametrize("heading", ["950", "95031", "toys"])
def test_malformed_heading_is_422(heading):
    assert client.get(f"/headings/{heading}").status_code == 422


def test_validate_decision_returns_cleaned_record(raw_decision):
    response = client.post("/decisions/validate", json=raw_decision)
    assert response.status_code == 200
    body = response.json()
    assert body["bti_reference"] == "DE0001/23-1"
    assert body["start_date"] == "2023-05-10"
    assert body["chapter"] == "95"


def test_validate_decision_reports_the_field(raw_decision):
    raw_decision["issuing_country"] = "Germany"
    response = client.post("/decisions/validate", json=raw_decision)
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "issuing_country"


def test_features():
    response = client.get("/features", params={"text": "Plush toy\nsee <CODE>"})
    assert response.json() == {"n_chars": 20, "n_words": 4, "n_lines": 2, "has_code": 1}
