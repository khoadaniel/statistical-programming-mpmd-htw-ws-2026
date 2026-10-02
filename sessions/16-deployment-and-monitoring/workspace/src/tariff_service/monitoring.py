"""Prediction logging: one JSON line per request, the raw data of every later drift check."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .schemas import DecisionIn, PredictionOut


class PredictionLogger:
    """Append-only log. It stores the length and language of a description, not the text: a pending BTI
    request contains confidential business information."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, decision: DecisionIn, prediction: PredictionOut) -> None:
        record = {"time": datetime.now(UTC).isoformat(timespec="seconds"),
                  "model_version": prediction.model_version,
                  "text_length": len(decision.description),
                  "language": decision.language,
                  "heading": prediction.heading,
                  "top_score": prediction.top[0].score}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
