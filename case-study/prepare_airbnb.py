"""Prepare the second course dataset: Inside Airbnb, Berlin.

Inside Airbnb publishes quarterly snapshots of the Airbnb listings of many cities. The course uses
the Berlin snapshot for the numeric topics: prices, statistics, regression, tree-based models,
clustering, anomalies and demand over time.

    uv run python case-study/prepare_airbnb.py
    uv run python case-study/prepare_airbnb.py --postgres postgresql+psycopg://postgres:course@localhost/postgres

The script downloads the latest Berlin snapshot (about 100 MB, once), keeps the columns needed in
the course, removes personal data (host names, profile texts, photos, review texts, reviewer
names, and names entered in the registration field) and writes case-study/data/airbnb/:

    listings.parquet         one row per listing: location, property, price, availability, reviews, host
    calendar.parquet         availability of each listing for the next 365 days
    reviews_monthly.parquet  number of reviews per listing and month (a proxy for demand), since 2009
    weather_daily.parquet    daily Berlin weather since 2016 (Open-Meteo archive, CC BY 4.0)
    neighbourhoods.geojson   district boundaries

Source: Inside Airbnb, https://insideairbnb.com/get-the-data/ (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

GET_THE_DATA = "https://insideairbnb.com/get-the-data/"
BASE = "https://data.insideairbnb.com/germany/be/berlin/{date}/"
FILES = {"listings.csv.gz": "data/listings.csv.gz", "calendar.csv.gz": "data/calendar.csv.gz",
         "reviews.csv.gz": "data/reviews.csv.gz", "neighbourhoods.geojson": "visualisations/neighbourhoods.geojson"}
HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "data" / "raw" / "airbnb", HERE / "data" / "airbnb"

LISTING_COLUMNS = [
    "id", "name", "host_id", "host_since", "hosts_time_as_host_years", "host_is_superhost", "host_listings_count", "calculated_host_listings_count",
    "host_response_rate", "host_acceptance_rate", "host_identity_verified",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms", "bedrooms", "beds", "amenities",
    "price", "minimum_nights", "maximum_nights", "instant_bookable", "license",
    "availability_30", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "first_review", "last_review", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
    "estimated_occupancy_l365d", "estimated_revenue_l365d", "last_scraped",
]


def latest_snapshot() -> str:
    html = urllib.request.urlopen(GET_THE_DATA).read().decode("utf-8", "ignore")
    dates = sorted(set(re.findall(r"germany/be/berlin/(\d{4}-\d{2}-\d{2})/", html)))
    if not dates:
        raise RuntimeError("no Berlin snapshot found on " + GET_THE_DATA)
    return dates[-1]


def download() -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    marker = RAW / "snapshot.txt"
    if marker.exists() and all((RAW / f).exists() for f in FILES):
        return marker.read_text().strip()
    date = latest_snapshot()
    for name, path in FILES.items():
        print(f"downloading {name} (snapshot {date})")
        urllib.request.urlretrieve(BASE.format(date=date) + path, RAW / name)
    marker.write_text(date)
    return date


def to_share(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.rstrip("%"), errors="coerce") / 100


REGISTRATION = re.compile(r"\d{2}/[Z3]/[A-Z]{2}/?\d+", re.IGNORECASE)


def license_status(value: object) -> str:
    """Type of entry in the registration field, without the names it may contain."""
    if not isinstance(value, str) or not value.strip():
        return "missing"
    if value.startswith("First name"):
        return "private host name"
    if value.startswith("Legal entity"):
        return "legal entity name"
    return "registration number" if REGISTRATION.search(value) else "other"


def build() -> dict[str, pd.DataFrame]:
    listings = pd.read_csv(RAW / "listings.csv.gz", usecols=lambda c: c in LISTING_COLUMNS)
    listings["price"] = pd.to_numeric(listings["price"].str.replace(r"[$€,]", "", regex=True), errors="coerce")
    for col in ("host_response_rate", "host_acceptance_rate"):
        listings[col] = to_share(listings[col])
    for col in ("host_is_superhost", "host_identity_verified", "instant_bookable"):
        listings[col] = listings[col].map({"t": True, "f": False})
    for col in ("host_since", "first_review", "last_review", "last_scraped"):
        listings[col] = pd.to_datetime(listings[col], errors="coerce")
    listings = listings.rename(columns={"neighbourhood_cleansed": "neighbourhood",
                                        "neighbourhood_group_cleansed": "district"})
    # the registration field sometimes holds a host's full name: keep only numbers and the type of entry
    listings["license_status"] = listings["license"].map(license_status)
    listings["license"] = listings["license"].where(listings["license_status"] == "registration number",
                                                    listings["license_status"].where(listings["license"].notna()))
    # Inside Airbnb leaves some fields empty in some snapshots; drop columns without any value
    listings = listings.dropna(axis="columns", how="all")

    calendar = pd.read_csv(RAW / "calendar.csv.gz", usecols=["listing_id", "date", "available", "minimum_nights"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["available"] = calendar["available"].map({"t": True, "f": False})

    # only listing and date are kept: review texts and reviewer names are personal data
    reviews = pd.read_csv(RAW / "reviews.csv.gz", usecols=["listing_id", "date"])
    reviews["month"] = pd.to_datetime(reviews["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = reviews.groupby(["listing_id", "month"]).size().rename("n_reviews").reset_index()
    return {"listings": listings, "calendar": calendar, "reviews_monthly": monthly}


WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41"
               "&start_date=2016-01-01&end_date={end}&timezone=Europe%2FBerlin"
               "&daily=temperature_2m_mean,precipitation_sum,sunshine_duration")


def weather(end: str) -> pd.DataFrame:
    """Daily Berlin weather from the Open-Meteo archive (CC BY 4.0), the data Session 2 fetches by hand."""
    with urllib.request.urlopen(WEATHER_URL.format(end=end)) as response:
        daily = json.load(response)["daily"]
    out = pd.DataFrame(daily).rename(columns={"time": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["sunshine_hours"] = out.pop("sunshine_duration") / 3600
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres", help="SQLAlchemy URL; also load listings, calendar and reviews_monthly into PostgreSQL")
    args = parser.parse_args()
    date = download()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = build()
    tables["weather_daily"] = weather(date)
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:16s} {len(df):>10,} rows")
    (OUT / "neighbourhoods.geojson").write_bytes((RAW / "neighbourhoods.geojson").read_bytes())
    (OUT / "snapshot.txt").write_text(date)
    print(f"Inside Airbnb Berlin, snapshot {date}")
    if args.postgres:
        from sqlalchemy import create_engine, text
        engine = create_engine(args.postgres)
        names = ("listings", "calendar", "reviews_monthly", "weather_daily")
        with engine.begin() as conn:  # CASCADE: constraints added in Session 3 must not block a reload
            for name in names:
                conn.execute(text(f"DROP TABLE IF EXISTS {name} CASCADE"))
        for name in names:
            tables[name].to_sql(name, engine, if_exists="append", index=False, chunksize=50_000)
        print("loaded listings, calendar, reviews_monthly and weather_daily into PostgreSQL")


if __name__ == "__main__":
    main()
