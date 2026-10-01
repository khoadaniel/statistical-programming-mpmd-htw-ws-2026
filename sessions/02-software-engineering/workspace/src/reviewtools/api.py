"""A minimal web API for the review tools, built with FastAPI.

Start a development server (stop it with Ctrl+C):

    uv run uvicorn reviewtools.api:app --reload

then open http://127.0.0.1:8000/docs for the interactive documentation that FastAPI
generates. The tests in tests/test_api.py call the same app without a server, through
FastAPI's TestClient. Session 16 turns this app into a model service.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Path, Query

from reviewtools.errors import ValidationError
from reviewtools.records import LABELS, ReviewRecord, to_label
from reviewtools.text_features import n_words

#: Example request body shown in the interactive documentation.
EXAMPLE_REVIEW = {"review_id": " r1 ", "rating": "2", "text": "Broke  after a week"}

app = FastAPI(
    title="Review tools API",
    version="0.2.0",
    description="Validate review records and compute simple text features.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by monitoring: answers if the service is running."""
    return {"status": "ok"}


@app.get("/labels")
def labels() -> dict[str, list[str]]:
    """The sentiment classes of the course leaderboard."""
    return {"labels": list(LABELS)}


@app.get("/labels/{rating}")
def label_for_rating(
    rating: Annotated[int, Path(ge=1, le=5, description="Star rating 1-5")],
) -> dict[str, Any]:
    """The sentiment label of a star rating."""
    return {"rating": rating, "label": to_label(rating)}


@app.post("/reviews/validate")
def validate_review(
    raw: Annotated[dict[str, Any], Body(examples=[EXAMPLE_REVIEW])],
) -> dict[str, Any]:
    """Validate and clean one review record; answer 422 with the offending field if invalid."""
    try:
        record = ReviewRecord.from_dict(raw)
    except ValidationError as err:
        detail = {"field": err.field, "message": str(err)}
        raise HTTPException(status_code=422, detail=detail) from err
    return record.to_dict()


@app.get("/features")
def features(
    text: Annotated[str, Query(min_length=1, description="Review text")],
) -> dict[str, int]:
    """Simple features of a text.

    TODO (exercise 7): after your pull request for text_features is merged, add
    ``n_exclamations`` and ``count_negations`` to the response and to tests/test_api.py.
    """
    return {"n_words": n_words(text)}
