"""The prediction API.

    MODEL_DIR=models uv run uvicorn tariff_service.app:app --reload     # docs at http://localhost:8000/docs

The model is loaded once at start-up (lifespan), not per request. /predict returns the three most likely
HS headings with their scores and English texts: a suggestion for a customs officer, not a decision.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from . import __version__
from .config import Settings
from .model import decision_text, load_model, top_k
from .monitoring import PredictionLogger
from .schemas import BatchIn, BatchOut, DecisionIn, HeadingScore, Health, PredictionOut


def create_app(model_dir: Path | None = None, log_path: Path | None = None) -> FastAPI:
    settings = Settings.from_env()
    model_dir = model_dir or settings.model_dir
    log_path = log_path or settings.log_path

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.meta = load_model(model_dir)
        app.state.logger = PredictionLogger(log_path) if log_path else None
        yield

    app = FastAPI(title="Tariff heading API", version=__version__, lifespan=lifespan)

    def predict_one(request: Request, decision: DecisionIn) -> PredictionOut:
        model, meta = request.app.state.model, request.app.state.meta
        texts = meta.get("headings", {})
        ranked = top_k(model, [decision_text(decision.description)], k=3)[0]
        top = [HeadingScore(heading=h, score=s, heading_description=texts.get(h, "")) for h, s in ranked]
        out = PredictionOut(heading=top[0].heading, top=top, model_version=meta["model_version"])
        if request.app.state.logger is not None:
            request.app.state.logger.log(decision, out)
        return out

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=request.app.state.meta["model_version"])

    @app.get("/metadata")
    def metadata(request: Request) -> dict:
        meta = request.app.state.meta
        out = {k: meta[k] for k in ("model_version", "created_at", "scikit_learn", "validation") if k in meta}
        out["n_classes"] = len(meta.get("classes", []))
        return out

    @app.post("/predict", response_model=PredictionOut)
    def predict(decision: DecisionIn, request: Request) -> PredictionOut:
        return predict_one(request, decision)

    @app.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn, request: Request) -> BatchOut:
        # TODO (exercise 2): return one prediction per decision, in the same order, by calling
        # predict_one for each decision in batch.decisions. The tests in tests/test_exercises.py describe it.
        raise NotImplementedError("exercise 2: batch prediction")

    return app


app = create_app()  # what uvicorn imports; the model is loaded when the server starts
