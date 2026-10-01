"""Fixtures: a tiny model trained on twelve built-in sentences. No case-study data are needed."""

import pytest
from fastapi.testclient import TestClient

from sentiment_service.app import create_app
from sentiment_service.model import train_fixture_model


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train_fixture_model(out, version="0.0.0-test")
    return out


@pytest.fixture
def client(model_dir, tmp_path):
    # `with` runs the lifespan: the model is loaded as in a real server start
    with TestClient(create_app(model_dir=model_dir, log_path=tmp_path / "predictions.jsonl")) as c:
        yield c
