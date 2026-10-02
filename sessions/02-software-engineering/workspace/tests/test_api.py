"""Tests of the FastAPI app with TestClient: requests without starting a server."""

import pytest
from fastapi.testclient import TestClient

from listingtools.api import DISTRICTS, app, slug

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_districts_lists_all_twelve_largest_first():
    rows = client.get("/districts").json()["districts"]
    assert len(rows) == 12
    assert rows[0]["district"] == "Mitte"
    assert sum(row["n_listings"] for row in rows) == 12776


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Mitte", "mitte"),
        ("Tempelhof - Schöneberg", "tempelhof-schöneberg"),
        ("Charlottenburg-Wilm.", "charlottenburg-wilm"),
    ],
)
def test_slug(name, expected):
    assert slug(name) == expected


@pytest.mark.parametrize("district", ["neukölln", "Neukölln", "tempelhof-schöneberg"])
def test_district_details(district):
    body = client.get(f"/districts/{district}").json()
    assert body["district"] in DISTRICTS
    assert body["n_short_stays"] <= body["n_listings"]
    assert body["median_price"] > 0


def test_unknown_district_is_404():
    assert client.get("/districts/atlantis").status_code == 404


def test_district_with_digits_is_422():
    assert client.get("/districts/10115").status_code == 422


def test_validate_listing_returns_cleaned_record(raw_listing):
    response = client.post("/listings/validate", json=raw_listing)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 3176
    assert body["price"] == pytest.approx(1160.5)
    assert body["room_type"] == "Entire home/apt"


def test_validate_listing_reports_the_field(raw_listing):
    raw_listing["latitude"] = 48.137  # Munich
    response = client.post("/listings/validate", json=raw_listing)
    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "latitude"


def test_features():
    response = client.get("/features", params={"title": "Sunny loft, 75 m²!"})
    assert response.json() == {
        "n_chars": 18,
        "n_words": 4,
        "has_exclamation": 1,
        "mentions_size": 1,
    }
