"""Tests of the FastAPI app with TestClient: requests without starting a server."""

import pytest
from fastapi.testclient import TestClient

from reviewtools.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(("rating", "label"), [(1, "neg"), (3, "neu"), (5, "pos")])
def test_label_for_rating(rating, label):
    assert client.get(f"/labels/{rating}").json() == {"rating": rating, "label": label}


def test_label_rejects_out_of_range_rating():
    assert client.get("/labels/6").status_code == 422


def test_validate_review_returns_cleaned_record(raw_review):
    response = client.post("/reviews/validate", json=raw_review)
    assert response.status_code == 200
    body = response.json()
    assert body["review_id"] == "r000042"
    assert body["label"] == "neg"


def test_validate_review_reports_the_field(raw_review):
    raw_review["rating"] = 7
    response = client.post("/reviews/validate", json=raw_review)
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "rating"


def test_features():
    assert client.get("/features", params={"text": "Works as described"}).json() == {"n_words": 3}
