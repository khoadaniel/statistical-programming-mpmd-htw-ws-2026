"""The prediction API.

    MODEL_DIR=models uv run uvicorn sentiment_service.app:app --reload     # docs at http://localhost:8000/docs

The model is loaded once at start-up (lifespan), not per request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from . import __version__
from .config import Settings
from .model import load_model, review_text
from .monitoring import PredictionLogger
from .schemas import BatchIn, BatchOut, Health, PredictionOut, ReviewIn


def create_app(model_dir: Path | None = None, log_path: Path | None = None) -> FastAPI:
    settings = Settings.from_env()
    model_dir = model_dir or settings.model_dir
    log_path = log_path or settings.log_path

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.meta = load_model(model_dir)
        app.state.logger = PredictionLogger(log_path) if log_path else None
        yield

    app = FastAPI(title="Review sentiment API", version=__version__, lifespan=lifespan)

    def predict_one(request: Request, review: ReviewIn) -> PredictionOut:
        model, meta = request.app.state.model, request.app.state.meta
        proba = model.predict_proba([review_text(review.title, review.text)])[0]
        probs = {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba, strict=True)}
        out = PredictionOut(label=max(probs, key=probs.get), probabilities=probs,
                            model_version=meta["model_version"])
        if request.app.state.logger is not None:
            request.app.state.logger.log(review, out)
        return out

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=request.app.state.meta["model_version"])

    @app.get("/metadata")
    def metadata(request: Request) -> dict:
        meta = request.app.state.meta
        return {k: meta[k] for k in ("model_version", "created_at", "scikit_learn", "classes", "validation")
                if k in meta}

    @app.post("/predict", response_model=PredictionOut)
    def predict(review: ReviewIn, request: Request) -> PredictionOut:
        return predict_one(request, review)

    @app.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn, request: Request) -> BatchOut:
        # TODO (exercise 2): return one prediction per review, in the same order, by calling
        # predict_one for each review in batch.reviews. The tests in tests/test_exercises.py describe it.
        raise NotImplementedError("exercise 2: batch prediction")

    return app


app = create_app()  # what uvicorn imports; the model is loaded when the server starts
