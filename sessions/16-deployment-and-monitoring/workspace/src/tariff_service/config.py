"""Settings that differ between a laptop, CI and a server come from environment variables, not from code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    log_path: Path | None
    database_url: str | None = None  # set in compose.yaml: predictions are then logged to PostgreSQL

    @classmethod
    def from_env(cls) -> Settings:
        log = os.environ.get("PREDICTION_LOG")
        return cls(model_dir=Path(os.environ.get("MODEL_DIR", "models")), log_path=Path(log) if log else None,
                   database_url=os.environ.get("DATABASE_URL") or None)
