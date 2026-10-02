"""Load and summarise the course data of Sessions 1-12: Inside Airbnb listings in Berlin.

Reference solution, for self-checking only: copy this file over
src/listingtools/listings.py after you have tried the exercises yourself.

This module is the notebook analysis of Session 1 turned into reusable code: one class,
``ListingTable``, holds the listings and answers the five questions of the case-study
exercise (how many, where, what type, what price, how many reviews) through methods.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Columns every listing table must have; the class refuses tables without them.
REQUIRED_COLUMNS = (
    "id",
    "host_id",
    "district",
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
)

#: Listings with a minimum stay of this many nights or more are rented by the month; their
#: price field is not comparable with that of short stays, so price questions leave them out.
SHORT_STAY_MAX_NIGHTS = 28


class ListingTable:
    """A table of Airbnb listings with methods that summarise it.

    Attributes:
        data: the listings as a pandas DataFrame, one row per listing.

    Example:
        >>> table = ListingTable.from_parquet("case-study/data/airbnb/listings.parquet")
        >>> len(table)
        12776
    """

    def __init__(self, data: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"the listing table is missing the columns {missing}")
        self.data = data

    @classmethod
    def from_parquet(cls, path: str | Path) -> ListingTable:
        """Read a Parquet file of listings and return a ``ListingTable``."""
        return cls(pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"ListingTable({len(self)} listings)"

    # Question 1: how many? ----------------------------------------------------
    def n_hosts(self) -> int:
        """Number of different hosts; many hosts offer more than one listing."""
        return int(self.data["host_id"].nunique())

    # Question 2: where? -------------------------------------------------------
    def by_district(self) -> pd.DataFrame:
        """Number and share of listings per district, largest first."""
        counts = self.data["district"].value_counts()
        return pd.DataFrame(
            {"n_listings": counts, "share": (counts / len(self)).round(3)}
        ).rename_axis("district")

    # Question 3: what type? ---------------------------------------------------
    def room_type_shares(self) -> pd.Series:
        """Share of listings per room type, largest first."""
        return self.data["room_type"].value_counts(normalize=True)

    # Question 4: what price? --------------------------------------------------
    def short_stays(self) -> ListingTable:
        """The listings with a price and a minimum stay of fewer than 28 nights.

        Returns a new ``ListingTable``, so every method also works on the short stays.
        """
        keep = self.data["price"].notna() & (self.data["minimum_nights"] < SHORT_STAY_MAX_NIGHTS)
        return ListingTable(self.data[keep])

    def price_by_room_type(self) -> pd.DataFrame:
        """Number of short-stay listings and their median price per night, per room type."""
        short = self.short_stays().data
        return (
            short.groupby("room_type")["price"]
            .agg(n_listings="size", median_price="median")
            .sort_values("median_price", ascending=False)
        )

    def median_price_by_district(self) -> pd.Series:
        """Median price per night of the short-stay listings in each district, cheapest first."""
        short = self.short_stays().data
        return short.groupby("district")["price"].median().sort_values()

    # Question 5: how many reviews? --------------------------------------------
    def review_summary(self) -> dict[str, float]:
        """Median number of reviews per listing and the share of listings without any review."""
        reviews = self.data["number_of_reviews"]
        return {
            "median_reviews": float(reviews.median()),
            "share_without_reviews": float((reviews == 0).mean()),
        }

    # Summary ------------------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """The answers to the case-study questions as a dictionary."""
        short = self.short_stays()
        return {
            "n_listings": len(self),
            "n_hosts": self.n_hosts(),
            "listings_per_district": self.by_district()["n_listings"].head(5).to_dict(),
            "room_type_shares": self.room_type_shares().round(3).to_dict(),
            "n_short_stays": len(short),
            "median_short_stay_price": float(short.data["price"].median()),
            "median_price_by_room_type": self.price_by_room_type()["median_price"].to_dict(),
            "median_price_by_district": self.median_price_by_district().to_dict(),
            "reviews": self.review_summary(),
        }
