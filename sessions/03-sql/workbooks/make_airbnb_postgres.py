"""Build the course's PostgreSQL database of the Berlin Airbnb case study.

Session 3 queries the Inside Airbnb Berlin snapshot with SQL. This script starts (or reuses) a local PostgreSQL server whose
data live in case-study/data/airbnb/pgdata/, creates the database `airbnb` and stores three tables in it:

    listings   one row per listing (id is the primary key)
    calendar   one row per listing and night of the coming year (listing_id, date, available)
    reviews    number of reviews per listing and month (listing_id, month, n_reviews)

Unlike SQLite, PostgreSQL has real DATE, BOOLEAN and NUMERIC types, so dates stay dates and `available`
stays true/false. The rows are loaded with COPY, PostgreSQL's command for bulk loading.

The server comes from the package pixeltable-pgserver, which ships the PostgreSQL programs as a Python wheel:

    uv run python sessions/03-sql/workbooks/make_airbnb_postgres.py
    ... make_airbnb_postgres.py --force      # drop and rebuild the three tables

The output lies in case-study/data/, which is not committed to Git.
Author: course team, licence CC BY 4.0. Data: Inside Airbnb, Berlin, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import pixeltable_pgserver as pgserver
import psycopg

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "case-study" / "data" / "airbnb"
PGDATA = PARQUET / "pgdata"
DATABASE = "airbnb"

SCHEMA = """
CREATE TABLE listings (
    id                   BIGINT PRIMARY KEY,
    name                 TEXT,
    host_id              BIGINT NOT NULL,
    district             TEXT NOT NULL,
    neighbourhood        TEXT,
    latitude             DOUBLE PRECISION,
    longitude            DOUBLE PRECISION,
    room_type            TEXT NOT NULL,
    accommodates         INTEGER,
    bedrooms             INTEGER,
    price                NUMERIC(10, 2),
    minimum_nights       INTEGER,
    maximum_nights       INTEGER,
    availability_365     INTEGER,
    number_of_reviews    INTEGER,
    first_review         DATE,
    last_review          DATE,
    review_scores_rating NUMERIC(3, 2),
    license_status       TEXT
);
CREATE TABLE calendar (
    listing_id BIGINT  NOT NULL,
    date       DATE    NOT NULL,
    available  BOOLEAN NOT NULL,
    PRIMARY KEY (listing_id, date)
);
CREATE TABLE reviews (
    listing_id BIGINT  NOT NULL,
    month      DATE    NOT NULL,
    n_reviews  INTEGER NOT NULL,
    PRIMARY KEY (listing_id, month)
);
"""
# No foreign keys: some calendar and review rows belong to listings that are missing from listings.parquet
# (the notebook shows them, and shows the error PostgreSQL gives when a foreign key is added).
LISTING_COLUMNS = ["id", "name", "host_id", "district", "neighbourhood", "latitude", "longitude", "room_type",
                   "accommodates", "bedrooms", "price", "minimum_nights", "maximum_nights", "availability_365",
                   "number_of_reviews", "first_review", "last_review", "review_scores_rating", "license_status"]
WHOLE_NUMBERS = ["bedrooms", "minimum_nights", "maximum_nights"]     # float in Parquet because of missing values


def start_server(pgdata: Path = PGDATA) -> pgserver.PostgresServer:
    """Start the local PostgreSQL server, or attach to it if it already runs, and make sure `airbnb` exists."""
    pgdata.parent.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(pgdata)
    with psycopg.connect(server.get_uri(), autocommit=True) as con:
        if not con.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,)).fetchone():
            con.execute(f"CREATE DATABASE {DATABASE}")
    return server


def copy_frame(con: psycopg.Connection, table: str, frame: pd.DataFrame) -> None:
    """Load a DataFrame into an existing table with COPY ... FROM STDIN, as CSV (an empty field is NULL)."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, date_format="%Y-%m-%d")
    with con.cursor().copy(f"COPY {table} ({', '.join(frame.columns)}) FROM STDIN (FORMAT csv)") as copy:
        copy.write(buffer.getvalue())


def build(uri: str, parquet: Path = PARQUET, force: bool = False) -> None:
    """Create listings, calendar and reviews from the Parquet files, in one transaction."""
    if not (parquet / "listings.parquet").exists():
        raise SystemExit(f"{parquet / 'listings.parquet'} not found: download airbnb-v1.zip from the course drive "
                         "and unzip it into case-study/data/ (see case-study/README.md)")
    with psycopg.connect(uri) as con:                  # one transaction: commit at the end, or nothing at all
        exists = con.execute("SELECT to_regclass('listings') IS NOT NULL").fetchone()[0]
        if exists and not force:
            print("the tables exist; use --force to rebuild them")
            return
        con.execute("DROP TABLE IF EXISTS listings, calendar, reviews")
        con.execute(SCHEMA)

        listings = pd.read_parquet(parquet / "listings.parquet", columns=LISTING_COLUMNS)
        listings[WHOLE_NUMBERS] = listings[WHOLE_NUMBERS].astype("Int64")
        copy_frame(con, "listings", listings)
        calendar = pd.read_parquet(parquet / "calendar.parquet", columns=["listing_id", "date", "available"])
        copy_frame(con, "calendar", calendar)
        copy_frame(con, "reviews", pd.read_parquet(parquet / "reviews_monthly.parquet"))

        for table in ["listings", "calendar", "reviews"]:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:10s} {n:>10,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="drop and rebuild the three tables")
    args = parser.parse_args()
    start = time.perf_counter()
    server = start_server()
    build(server.get_uri(DATABASE), force=args.force)
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
