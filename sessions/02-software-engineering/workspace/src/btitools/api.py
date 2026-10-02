"""A minimal web API for the BTI tools, built with FastAPI.

Start a development server (stop it with Ctrl+C):

    uv run uvicorn btitools.api:app --reload

then open http://127.0.0.1:8000/docs for the interactive documentation that FastAPI
generates. The tests in tests/test_api.py call the same app without a server, through
FastAPI's TestClient. Session 16 turns this app into a model service that proposes
headings for a description of goods.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Path, Query

from btitools.errors import ValidationError
from btitools.records import DecisionRecord
from btitools.text_features import has_code, n_chars, n_lines, n_words

#: A small extract of the HS nomenclature (English, HS 2022; long texts shortened with ...).
#: The full table with 1,229 headings is case-study/data/nomenclature.parquet; a fixture
#: keeps the app self-contained.
NOMENCLATURE: dict[str, dict[str, str]] = {
    "0901": {
        "chapter": "09",
        "description": "Coffee, whether or not roasted or decaffeinated; ...",
    },
    "2106": {
        "chapter": "21",
        "description": "Food preparations not elsewhere specified or included",
    },
    "3926": {"chapter": "39", "description": "Articles of plastics ..., n.e.c. in chapter 39"},
    "4202": {"chapter": "42", "description": "Trunks; suit, camera, jewellery, cutlery cases; ..."},
    "6307": {"chapter": "63", "description": "Textiles; made up articles n.e.c. in chapter 63"},
    "6403": {"chapter": "64", "description": "Footwear; ... and uppers of leather"},
    "6404": {"chapter": "64", "description": "Footwear; ... and uppers of textile materials"},
    "8517": {"chapter": "85", "description": "Telephone sets, including smartphones ..."},
    "9503": {"chapter": "95", "description": "Tricycles, scooters, ...; dolls; other toys; ..."},
}

#: Example request body shown in the interactive documentation.
EXAMPLE_DECISION = {
    "bti_reference": " DE0001/23-1 ",
    "issuing_country": "de",
    "language": "DE",
    "start_date": "10/05/2023",
    "description": "Plüschtier  in Form eines Bären, Höhe 30 cm",
    "heading": "9503",
}

app = FastAPI(
    title="BTI tools API",
    version="0.2.0",
    description="Look up tariff headings, validate BTI decisions and compute text features.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by monitoring: answers if the service is running."""
    return {"status": "ok"}


@app.get("/headings")
def headings() -> dict[str, list[str]]:
    """The headings this small service knows."""
    return {"headings": sorted(NOMENCLATURE)}


@app.get("/headings/{heading}")
def heading_details(
    heading: Annotated[str, Path(pattern=r"^\d{4}$", description="Four-digit HS heading")],
) -> dict[str, Any]:
    """The chapter and English description of a heading; 404 if it is not in the extract."""
    if heading not in NOMENCLATURE:
        raise HTTPException(status_code=404, detail=f"heading {heading} not in the extract")
    return {"heading": heading, **NOMENCLATURE[heading]}


@app.post("/decisions/validate")
def validate_decision(
    raw: Annotated[dict[str, Any], Body(examples=[EXAMPLE_DECISION])],
) -> dict[str, Any]:
    """Validate and clean one decision; answer 422 with the offending field if invalid."""
    try:
        record = DecisionRecord.from_dict(raw)
    except ValidationError as err:
        detail = {"field": err.field, "message": str(err)}
        raise HTTPException(status_code=422, detail=detail) from err
    return record.to_dict()


@app.get("/features")
def features(
    text: Annotated[str, Query(min_length=1, description="Description of goods")],
) -> dict[str, int]:
    """Simple features of a description.

    TODO (exercise 7): after your pull request for text_features is merged, add
    ``n_digits`` and ``upper_share`` to the response and to tests/test_api.py
    (``upper_share`` is a float: change the return type to ``dict[str, float]``).
    """
    return {
        "n_chars": n_chars(text),
        "n_words": n_words(text),
        "n_lines": n_lines(text),
        "has_code": has_code(text),
    }
