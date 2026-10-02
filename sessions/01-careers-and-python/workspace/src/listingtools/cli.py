"""Command-line entry point: print the summary of a file of Airbnb listings.

Run from the repository root (the default path is relative to it):

    uv run --project sessions/01-careers-and-python/workspace listing-summary
    uv run --project sessions/01-careers-and-python/workspace listing-summary other.parquet
"""

from __future__ import annotations

import argparse
from pprint import pprint

from listingtools.listings import ListingTable

DEFAULT_PATH = "case-study/data/airbnb/listings.parquet"


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, load the listings and print their summary."""
    parser = argparse.ArgumentParser(description="Summarise a Parquet file of Airbnb listings.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="Parquet file of listings")
    args = parser.parse_args(argv)

    table = ListingTable.from_parquet(args.path)
    print(table)
    pprint(table.summary(), sort_dicts=False)
    print("\nShort-stay prices by room type (EUR per night):")
    print(table.price_by_room_type().to_string())


if __name__ == "__main__":
    main()
