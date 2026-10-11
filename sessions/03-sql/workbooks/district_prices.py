"""Answer one question from the course database: what does a short stay cost in a district?

For the district given on the command line, the script prints the number of short-stay listings with a price
(minimum stay below 28 nights) and their average price per room type.

    uv run python sessions/03-sql/workbooks/make_airbnb_sqlite.py      # once: build the database
    uv run python sessions/03-sql/workbooks/district_prices.py Neukölln
    uv run python sessions/03-sql/workbooks/district_prices.py "Treptow - Köpenick" --max-nights 7

Author: course team, licence CC BY 4.0. Data: Inside Airbnb, Berlin, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parents[3] / "case-study" / "data" / "airbnb.sqlite"

QUERY = """
SELECT room_type, COUNT(*) AS n_listings, ROUND(AVG(price), 1) AS avg_price
FROM listings
WHERE district = ? AND minimum_nights < ? AND price IS NOT NULL
GROUP BY room_type
ORDER BY n_listings DESC
"""


def district_prices(con: sqlite3.Connection, district: str, max_nights: int = 28) -> pd.DataFrame:
    """Short-stay listings and average price per room type in one district."""
    return pd.read_sql_query(QUERY, con, params=(district, max_nights))


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-stay prices per room type in one Berlin district.")
    parser.add_argument("district", help="district name as in the data, for example Neukölln")
    parser.add_argument("--max-nights", type=int, default=28, help="minimum stay must be below this (default 28)")
    args = parser.parse_args()
    if not DB.exists():
        raise SystemExit(f"{DB} not found: run sessions/03-sql/workbooks/make_airbnb_sqlite.py first")

    con = sqlite3.connect(DB)
    try:
        known = [row[0] for row in con.execute("SELECT DISTINCT district FROM listings ORDER BY district")]
        if args.district not in known:
            raise SystemExit(f"unknown district {args.district!r}; choose one of: {', '.join(known)}")
        result = district_prices(con, args.district, args.max_nights)
    finally:
        con.close()
    print(f"Short stays (minimum stay below {args.max_nights} nights) with a price in {args.district}:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
