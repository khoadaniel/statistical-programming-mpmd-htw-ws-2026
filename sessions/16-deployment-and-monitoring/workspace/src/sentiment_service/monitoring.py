"""Prediction logging: one JSON line per request, the raw data of every later drift check."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .schemas import PredictionOut, ReviewIn


class PredictionLogger:
    """Append-only log. It stores the text length, not the text: reviews can contain personal data."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, review: ReviewIn, prediction: PredictionOut) -> None:
        record = {"time": datetime.now(UTC).isoformat(timespec="seconds"),
                  "model_version": prediction.model_version,
                  "text_length": len(review.text) + len(review.title),
                  "label": prediction.label,
                  "max_probability": max(prediction.probabilities.values())}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
