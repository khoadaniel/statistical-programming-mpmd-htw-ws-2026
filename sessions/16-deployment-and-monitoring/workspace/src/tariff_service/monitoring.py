"""Prediction logging, the raw data of every later drift check.

PredictionLogger writes one JSON line per request to a file (laptop, tests). PostgresPredictionLogger writes the
same record to a table, as in the three-container setup of compose.yaml, where the database is its own service.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .schemas import DecisionIn, PredictionOut


def _record(decision: DecisionIn, prediction: PredictionOut) -> dict:
    return {"model_version": prediction.model_version, "text_length": len(decision.description),
            "language": decision.language, "heading": prediction.heading, "top_score": prediction.top[0].score}


class PredictionLogger:
    """Append-only log. It stores the length and language of a description, not the text: a pending BTI
    request contains confidential business information."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, decision: DecisionIn, prediction: PredictionOut) -> None:
        record = {"time": datetime.now(UTC).isoformat(timespec="seconds"), **_record(decision, prediction)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class PostgresPredictionLogger:
    """The same record in a PostgreSQL table `predictions` (needs the extra `postgres`: psycopg).

    A new connection per request keeps the example short; a real service would use a connection pool."""

    CREATE = """CREATE TABLE IF NOT EXISTS predictions (
        time timestamptz NOT NULL DEFAULT now(), model_version text NOT NULL, text_length integer NOT NULL,
        language text, heading char(4) NOT NULL, top_score double precision NOT NULL)"""
    INSERT = """INSERT INTO predictions (model_version, text_length, language, heading, top_score)
        VALUES (%(model_version)s, %(text_length)s, %(language)s, %(heading)s, %(top_score)s)"""

    def __init__(self, url: str) -> None:
        import psycopg  # imported here: only this logger needs the database driver

        self._connect = lambda: psycopg.connect(url.replace("postgresql+psycopg://", "postgresql://"))
        with self._connect() as conn:
            conn.execute(self.CREATE)

    def log(self, decision: DecisionIn, prediction: PredictionOut) -> None:
        with self._connect() as conn:  # commits when the block ends
            conn.execute(self.INSERT, _record(decision, prediction))


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
