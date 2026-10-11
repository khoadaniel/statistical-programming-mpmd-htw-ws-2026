"""Build the course's SQLite database of the Berlin Airbnb case study.

Session 3 queries the Inside Airbnb Berlin snapshot with SQL. This script reads the Parquet files that
case-study/prepare_airbnb.py writes to case-study/data/airbnb/ and stores three tables in one SQLite file:

    listings   one row per listing (id is the primary key)
    calendar   one row per listing and night of the coming year (listing_id, date, available 0/1)
    reviews    number of reviews per listing and month (listing_id, month, n_reviews)

Dates are stored as text 'YYYY-MM-DD', because SQLite has no date type; text in this format sorts in date
order and works with SQLite's date functions (date(), strftime(), julianday()).

Run from the repository root (a few seconds; the file has about 130 MB):

    uv run python case-study/prepare_airbnb.py                      # once, if the Parquet files are missing
    uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py   # writes case-study/data/airbnb.sqlite
    uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py --force   # rebuild an existing file

The output lies in case-study/data/, which is not committed to Git.
Author: course team, licence CC BY 4.0. Data: Inside Airbnb, Berlin, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "case-study" / "data" / "airbnb"
DB = ROOT / "case-study" / "data" / "airbnb.sqlite"

# The listings table is created with its schema first, so that id is a real primary key.
LISTINGS_SCHEMA = """
CREATE TABLE listings (
    id                   INTEGER PRIMARY KEY,
    name                 TEXT,
    host_id              INTEGER NOT NULL,
    district             TEXT NOT NULL,
    neighbourhood        TEXT,
    latitude             REAL,
    longitude            REAL,
    room_type            TEXT NOT NULL,
    accommodates         INTEGER,
    bedrooms             INTEGER,
    price                REAL,
    minimum_nights       INTEGER,
    maximum_nights       INTEGER,
    availability_365     INTEGER,
    number_of_reviews    INTEGER,
    first_review         TEXT,
    last_review          TEXT,
    review_scores_rating REAL,
    license_status       TEXT
)
"""
LISTING_COLUMNS = ["id", "name", "host_id", "district", "neighbourhood", "latitude", "longitude", "room_type",
                   "accommodates", "bedrooms", "price", "minimum_nights", "maximum_nights", "availability_365",
                   "number_of_reviews", "first_review", "last_review", "review_scores_rating", "license_status"]


def as_text_date(column: pd.Series) -> pd.Series:
    """Datetime column -> 'YYYY-MM-DD' text (missing dates stay missing)."""
    return column.dt.strftime("%Y-%m-%d")


def build(db: Path = DB, parquet: Path = PARQUET) -> None:
    """Write listings, calendar and reviews from the Parquet files into a new SQLite file."""
    if not (parquet / "listings.parquet").exists():
        raise SystemExit(f"{parquet / 'listings.parquet'} not found: run case-study/prepare_airbnb.py first")
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_suffix(".sqlite.tmp")          # build next to the target, then rename: no half-built file
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        listings = pd.read_parquet(parquet / "listings.parquet", columns=LISTING_COLUMNS)
        for col in ["first_review", "last_review"]:
            listings[col] = as_text_date(listings[col])
        con.execute(LISTINGS_SCHEMA)
        listings.to_sql("listings", con, if_exists="append", index=False)

        calendar = pd.read_parquet(parquet / "calendar.parquet", columns=["listing_id", "date", "available"])
        calendar["date"] = as_text_date(calendar["date"])
        calendar["available"] = calendar["available"].astype(int)    # SQLite has no boolean type: 1 or 0
        calendar.to_sql("calendar", con, index=False, chunksize=200_000)

        reviews = pd.read_parquet(parquet / "reviews_monthly.parquet")
        reviews["month"] = as_text_date(reviews["month"])             # first day of the month
        reviews.to_sql("reviews", con, index=False)
        con.commit()

        for table in ["listings", "calendar", "reviews"]:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:10s} {n:>10,} rows")
    finally:
        con.close()
    tmp.replace(db)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DB, help=f"SQLite file to write (default: {DB})")
    parser.add_argument("--force", action="store_true", help="rebuild the file if it exists")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        print(f"{args.output} exists; use --force to rebuild it")
        return
    start = time.perf_counter()
    build(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
