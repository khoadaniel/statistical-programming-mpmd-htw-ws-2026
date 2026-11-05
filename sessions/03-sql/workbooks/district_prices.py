"""Answer one question from the course database: what does a short stay cost in a district?

For the district given on the command line, the script prints the number of short-stay listings with a price
(minimum stay below 28 nights) and their average price per room type. It connects to the local PostgreSQL server
of the course (and starts it if no notebook runs it).

    uv run python sessions/03-sql/workbooks/make_airbnb_postgres.py    # once: build the database
    uv run python sessions/03-sql/workbooks/district_prices.py Neukölln
    uv run python sessions/03-sql/workbooks/district_prices.py "Treptow - Köpenick" --max-nights 7

Author: course team, licence CC BY 4.0. Data: Inside Airbnb, Berlin, CC BY 4.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pixeltable_pgserver as pgserver
import psycopg
from psycopg.types.numeric import FloatLoader

PGDATA = Path(__file__).resolve().parents[3] / "case-study" / "data" / "airbnb" / "pgdata"

QUERY = """
SELECT room_type, COUNT(*) AS n_listings, ROUND(AVG(price), 1) AS avg_price
FROM listings
WHERE district = %s AND minimum_nights < %s AND price IS NOT NULL
GROUP BY room_type
ORDER BY n_listings DESC
"""


def district_prices(con: psycopg.Connection, district: str, max_nights: int = 28) -> pd.DataFrame:
    """Short-stay listings and average price per room type in one district."""
    with con.cursor() as cur:
        cur.execute(QUERY, (district, max_nights))
        return pd.DataFrame(cur.fetchall(), columns=[column.name for column in cur.description])


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-stay prices per room type in one Berlin district.")
    parser.add_argument("district", help="district name as in the data, for example Neukölln")
    parser.add_argument("--max-nights", type=int, default=28, help="minimum stay must be below this (default 28)")
    args = parser.parse_args()
    if not PGDATA.exists():
        raise SystemExit(f"{PGDATA} not found: run sessions/03-sql/workbooks/make_airbnb_postgres.py first")

    server = pgserver.get_server(PGDATA)                  # start the local server, or reuse the running one
    with psycopg.connect(server.get_uri("airbnb"), autocommit=True) as con:     # closed at the end of the block
        con.adapters.register_loader("numeric", FloatLoader)
        known = [row[0] for row in con.execute("SELECT DISTINCT district FROM listings ORDER BY district")]
        if args.district not in known:
            raise SystemExit(f"unknown district {args.district!r}; choose one of: {', '.join(known)}")
        result = district_prices(con, args.district, args.max_nights)
    print(f"Short stays (minimum stay below {args.max_nights} nights) with a price in {args.district}:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
