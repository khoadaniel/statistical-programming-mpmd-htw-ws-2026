"""Prediction logging, the raw data of every later drift check.

PredictionLogger writes one JSON line per request to a file (laptop, tests). SQLitePredictionLogger writes the
same record to a table in a SQLite database file, as in compose.yaml, where the file lies on a volume of the API
container so that the log survives a restart.
"""

from __future__ import annotations

import json
import sqlite3
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


class SQLitePredictionLogger:
    """The same record in the table `predictions` of a SQLite database file (sqlite3 is part of Python).

    A new connection per request keeps the example short and is cheap for SQLite: it only opens a file."""

    CREATE = """CREATE TABLE IF NOT EXISTS predictions (
        time TEXT NOT NULL, model_version TEXT NOT NULL, text_length INTEGER NOT NULL,
        language TEXT, heading TEXT NOT NULL, top_score REAL NOT NULL)"""
    INSERT = """INSERT INTO predictions (time, model_version, text_length, language, heading, top_score)
        VALUES (:time, :model_version, :text_length, :language, :heading, :top_score)"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(self.CREATE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def log(self, decision: DecisionIn, prediction: PredictionOut) -> None:
        record = {"time": datetime.now(UTC).isoformat(timespec="seconds"), **_record(decision, prediction)}
        conn = self._connect()
        try:
            with conn:  # commits when the block ends
                conn.execute(self.INSERT, record)
        finally:
            conn.close()


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
