"""A minimal web API for the listing tools, built with FastAPI.

Start a development server (stop it with Ctrl+C):

    uv run uvicorn listingtools.api:app --reload

then open http://127.0.0.1:8000/docs for the interactive documentation that FastAPI
generates. The tests in tests/test_api.py call the same app without a server, through
FastAPI's TestClient. Session 16 turns a service like this into a model service.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Path, Query

from listingtools.errors import ValidationError
from listingtools.records import ListingRecord
from listingtools.text_features import has_exclamation, mentions_size, n_chars, n_words

#: District statistics of the Inside Airbnb snapshot of 26 June 2026 (CC BY 4.0): all listings,
#: short-stay listings (price present, minimum stay below 28 nights) and their median price per
#: night in EUR. A small fixture keeps the app self-contained; Session 3 computes the same
#: numbers with SQL.
DISTRICTS: dict[str, dict[str, Any]] = {
    "Mitte": {"n_listings": 2826, "n_short_stays": 1537, "median_price": 187.0},
    "Friedrichshain-Kreuzberg": {"n_listings": 2652, "n_short_stays": 1337, "median_price": 160.0},
    "Pankow": {"n_listings": 1950, "n_short_stays": 1033, "median_price": 174.0},
    "Charlottenburg-Wilm.": {"n_listings": 1432, "n_short_stays": 760, "median_price": 149.3},
    "Neukölln": {"n_listings": 1308, "n_short_stays": 556, "median_price": 130.0},
    "Tempelhof - Schöneberg": {"n_listings": 854, "n_short_stays": 478, "median_price": 137.25},
    "Treptow - Köpenick": {"n_listings": 575, "n_short_stays": 333, "median_price": 125.0},
    "Lichtenberg": {"n_listings": 395, "n_short_stays": 226, "median_price": 132.0},
    "Steglitz - Zehlendorf": {"n_listings": 356, "n_short_stays": 174, "median_price": 130.62},
    "Reinickendorf": {"n_listings": 176, "n_short_stays": 92, "median_price": 99.53},
    "Spandau": {"n_listings": 133, "n_short_stays": 105, "median_price": 126.5},
    "Marzahn - Hellersdorf": {"n_listings": 119, "n_short_stays": 70, "median_price": 126.0},
}


def slug(name: str) -> str:
    """URL-friendly form of a district name: "Tempelhof - Schöneberg" -> "tempelhof-schöneberg"."""
    return "-".join(name.lower().replace(".", "").replace("-", " ").split())


SLUGS = {slug(name): name for name in DISTRICTS}

#: Example request body shown in the interactive documentation (a row of the raw file).
EXAMPLE_LISTING = {
    "id": "3176",
    "name": "Fabulous  Flat in great Location ",
    "neighbourhood_group_cleansed": "Pankow",
    "latitude": 52.53574,
    "longitude": 13.41734,
    "room_type": "Entire home/apt",
    "accommodates": 2,
    "price": "$160.71",
    "minimum_nights": 2,
    "maximum_nights": 730,
}

app = FastAPI(
    title="Listing tools API",
    version="0.2.0",
    description="Berlin districts, validation of Airbnb listings and simple title features.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by monitoring: answers if the service is running."""
    return {"status": "ok"}


@app.get("/districts")
def districts() -> dict[str, list[dict[str, Any]]]:
    """The twelve districts with their slug and number of listings, largest first."""
    rows = [
        {"slug": slug(n), "district": n, "n_listings": s["n_listings"]}
        for n, s in DISTRICTS.items()
    ]
    return {"districts": sorted(rows, key=lambda row: -row["n_listings"])}


@app.get("/districts/{district}")
def district_details(
    district: Annotated[str, Path(pattern=r"^[^0-9]+$", description="District name or slug")],
) -> dict[str, Any]:
    """Listings and median short-stay price of a district; 404 if the district is unknown."""
    name = SLUGS.get(slug(district))
    if name is None:
        raise HTTPException(status_code=404, detail=f"unknown district {district!r}")
    return {"district": name, "slug": slug(name), **DISTRICTS[name]}


@app.post("/listings/validate")
def validate_listing(
    raw: Annotated[dict[str, Any], Body(examples=[EXAMPLE_LISTING])],
) -> dict[str, Any]:
    """Validate and clean one listing; answer 422 with the offending field if invalid."""
    try:
        record = ListingRecord.from_dict(raw)
    except ValidationError as err:
        detail = {"field": err.field, "message": str(err)}
        raise HTTPException(status_code=422, detail=detail) from err
    return record.to_dict()


@app.get("/features")
def features(
    title: Annotated[str, Query(min_length=1, description="Title of a listing")],
) -> dict[str, int]:
    """Simple features of a listing title.

    TODO (exercise 7): after your pull request for text_features is merged, add
    ``n_digits`` and ``upper_share`` to the response and to tests/test_api.py
    (``upper_share`` is a float: change the return type to ``dict[str, float]``).
    """
    return {
        "n_chars": n_chars(title),
        "n_words": n_words(title),
        "has_exclamation": has_exclamation(title),
        "mentions_size": mentions_size(title),
    }
