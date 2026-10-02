"""Tests for the exercises in README.md. They are expected to fail (xfail) until you solve them."""

import pytest

from tariff_service.drift import oov_rate

exercise = pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="exercise not solved yet")


@exercise
def test_exercise2_batch_prediction(client):
    decisions = [{"description": "Damenstiefel mit Oberteil aus Rindleder"},
                 {"description": "Voiture jouet en matière plastique"}]
    r = client.post("/predict/batch", json={"decisions": decisions})
    assert r.status_code == 200
    assert [p["heading"] for p in r.json()["predictions"]] == ["6403", "9503"]


def test_exercise2_batch_validation(client):  # validation works already: pydantic runs before the TODO
    assert client.post("/predict/batch", json={"decisions": []}).status_code == 422


@exercise
def test_exercise4_oov_rate():
    vocab = {"schuhe", "aus", "leder"}
    assert oov_rate(["Schuhe aus Leder!"], vocab) == 0.0
    assert oov_rate(["Schuhe aus Textil", "a b"], vocab) == pytest.approx(1 / 3)   # single letters are not tokens
    assert oov_rate([""], vocab) == 0.0
