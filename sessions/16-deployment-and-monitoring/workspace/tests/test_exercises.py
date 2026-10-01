"""Tests for the exercises in README.md. They are expected to fail (xfail) until you solve them."""

import pytest

from sentiment_service.drift import oov_rate

exercise = pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="exercise not solved yet")


@exercise
def test_exercise2_batch_prediction(client):
    reviews = [{"text": "Broke after two days. Waste of money."}, {"text": "Love it! Works perfectly."}]
    r = client.post("/predict/batch", json={"reviews": reviews})
    assert r.status_code == 200
    labels = [p["label"] for p in r.json()["predictions"]]
    assert labels == ["neg", "pos"]


def test_exercise2_batch_validation(client):  # validation works already: pydantic runs before the TODO
    assert client.post("/predict/batch", json={"reviews": []}).status_code == 422


@exercise
def test_exercise4_oov_rate():
    vocab = {"great", "product", "works"}
    assert oov_rate(["Great product, works!"], vocab) == 0.0
    assert oov_rate(["Great gadget", "a b"], vocab) == 0.5      # single letters are not tokens
    assert oov_rate([""], vocab) == 0.0
