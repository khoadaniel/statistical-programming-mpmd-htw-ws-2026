"""Validation rules for the Berlin Airbnb listings of the case study, written as pytest tests.

Run from the repository root:
    uv run --with pytest pytest sessions/04-data-quality/workbooks/quality -v

Tests on the prepared tables encode hard rules (keys, domains, ranges, consistency within and across tables,
and the privacy rule for the registration field). Known problems of the snapshot are marked `xfail` with the
reason: they are documented, and pytest reports them as "expected failures". Tests on the cleaned table run
after workbook 15 has written `listings_clean.parquet`; otherwise they are skipped.
Author: course team, licence CC-BY-4.0.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "case-study").is_dir())
DATA = ROOT / "case-study" / "data" / "airbnb"
CLEAN = Path(os.environ.get("AIRBNB_CLEAN_OUT", DATA / "listings_clean.parquet"))

ROOM_TYPES = {"Entire home/apt", "Private room", "Hotel room", "Shared room"}
LICENSE_STATUS = {"registration number", "missing", "legal entity name", "private host name", "other"}
LAT, LON = (52.33, 52.68), (13.08, 13.77)            # bounding box of neighbourhoods.geojson
REGISTRATION = r"\d{2}/[Z3]/[A-Z]{2}/?\d+"           # the pattern prepare_airbnb.py keeps (any case)

if not (DATA / "listings.parquet").exists():
    pytest.skip("Airbnb data missing: run case-study/prepare_airbnb.py", allow_module_level=True)


@pytest.fixture(scope="module")
def listings():
    return pd.read_parquet(DATA / "listings.parquet")


@pytest.fixture(scope="module")
def reviews_monthly():
    return pd.read_parquet(DATA / "reviews_monthly.parquet")


@pytest.fixture(scope="module")
def calendar_ids():
    return pd.read_parquet(DATA / "calendar.parquet", columns=["listing_id"])["listing_id"].drop_duplicates()


@pytest.fixture(scope="module")
def clean():
    if not CLEAN.exists():
        pytest.skip("cleaned table not found: run workbook 15 first")
    return pd.read_parquet(CLEAN)


# ---------- listings: hard rules ----------

def test_id_is_primary_key(listings):
    assert listings["id"].notna().all()
    assert listings["id"].is_unique


def test_room_type_values(listings):
    assert listings["room_type"].isin(ROOM_TYPES).all()


def test_coordinates_inside_berlin(listings):
    assert listings["latitude"].between(*LAT).all()
    assert listings["longitude"].between(*LON).all()


def test_accommodates_at_least_one(listings):
    assert listings["accommodates"].ge(1).all()


def test_price_is_numeric_and_positive(listings):
    assert pd.api.types.is_float_dtype(listings["price"])
    assert listings["price"].dropna().gt(0).all()


def test_minimum_not_above_maximum(listings):
    both = listings["minimum_nights"].notna() & listings["maximum_nights"].notna()
    assert (listings.loc[both, "minimum_nights"] <= listings.loc[both, "maximum_nights"]).all()


@pytest.mark.parametrize("column, days", [("availability_30", 30), ("availability_90", 90), ("availability_365", 365)])
def test_availability_in_range(listings, column, days):
    assert listings[column].between(0, days).all()


def test_review_dates_ordered(listings):
    both = listings["first_review"].notna() & listings["last_review"].notna()
    assert (listings.loc[both, "first_review"] <= listings.loc[both, "last_review"]).all()


def test_rating_missing_exactly_without_reviews(listings):
    assert (listings["review_scores_rating"].isna() == listings["number_of_reviews"].eq(0)).all()


def test_review_count_matches_monthly_table(listings, reviews_monthly):
    monthly = reviews_monthly.groupby("listing_id")["n_reviews"].sum()
    total = listings.set_index("id")["number_of_reviews"]
    assert total.sub(monthly.reindex(total.index, fill_value=0)).eq(0).all()


def test_no_review_month_before_first_review(listings, reviews_monthly):
    first = listings.set_index("id")["first_review"].dt.to_period("M").dt.to_timestamp()
    merged = reviews_monthly.join(first.rename("first_month"), on="listing_id", how="inner")
    assert (merged["month"] >= merged["first_month"]).all()


def test_license_status_values(listings):
    assert listings["license_status"].isin(LICENSE_STATUS).all()


def test_registration_field_holds_no_names(listings):
    """Privacy rule: the field keeps registration numbers or a category, never free text such as a name."""
    value = listings["license"].dropna()
    number = value.str.contains(REGISTRATION, case=False, regex=True)
    category = value.isin(LICENSE_STATUS)
    assert (number | category).all()


# ---------- known problems of the snapshot (documented, expected to fail) ----------

@pytest.mark.xfail(strict=True, reason="known: 2 listings carry maximum_nights = 2,147,483,647, a software default")
def test_raw_no_sentinel_maximum_nights(listings):
    assert not listings["maximum_nights"].eq(2**31 - 1).any()


@pytest.mark.xfail(strict=True, reason="known: 10 listings require a minimum stay above 365 nights (up to 1,125)")
def test_raw_minimum_stay_at_most_one_year(listings):
    assert listings["minimum_nights"].dropna().le(365).all()


@pytest.mark.xfail(strict=True, reason="known: 47 listings have more bedrooms than guests")
def test_raw_bedrooms_not_above_guests(listings):
    assert not listings["bedrooms"].gt(listings["accommodates"]).any()


@pytest.mark.xfail(strict=True, reason="known: the calendar covers 79 listings that are not in the listings table")
def test_raw_calendar_listing_exists(listings, calendar_ids):
    assert calendar_ids.isin(listings["id"]).all()


@pytest.mark.xfail(strict=True, reason="known: reviews_monthly covers 57 listings that are not in the listings table")
def test_raw_reviews_listing_exists(listings, reviews_monthly):
    assert reviews_monthly["listing_id"].isin(listings["id"]).all()


# ---------- cleaned table (after workbook 15) ----------

def test_clean_id_unique(clean):
    assert clean["id"].is_unique


def test_clean_no_sentinel(clean):
    assert not clean["maximum_nights"].eq(2**31 - 1).any()


def test_clean_raw_registration_field_dropped(clean):
    assert "license" not in clean.columns
    assert clean["license_status"].isin(LICENSE_STATUS).all()


def test_clean_has_flags_and_indicators(clean):
    for col in ["price_missing", "bookable", "medium_term", "minimum_over_one_year", "no_reviews",
                "bedrooms_imputed", "price_extreme", "mahalanobis_flag"]:
        assert col in clean.columns
        assert clean[col].dtype == bool


def test_clean_indicators_consistent(clean):
    assert (clean["price_missing"] == clean["price"].isna()).all()
    assert (clean["no_reviews"] == clean["number_of_reviews"].eq(0)).all()
    assert (clean["medium_term"] == clean["minimum_nights"].ge(28)).all()


def test_clean_bedrooms_complete(clean):
    assert clean["bedrooms"].notna().all()
