"""Shared test data: one messy raw listing, and eight listings small enough to check by hand."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def raw_listing() -> dict[str, Any]:
    """A messy but valid listing, shaped like a row of the raw Inside Airbnb file."""
    return {
        "id": "3176",
        "name": "  Fabulous Flat   in great Location ",
        "neighbourhood_group_cleansed": "Pankow",
        "latitude": "52.53574",
        "longitude": 13.41734,
        "room_type": "entire home/apt",
        "accommodates": 2,
        "price": "$1,160.50",
        "minimum_nights": "2",
        "maximum_nights": 730,
        "host_name": "ignored: not part of the record",
    }


@pytest.fixture
def tiny_listings() -> pd.DataFrame:
    """Eight listings in the shape of the prepared ``listings.parquet``."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "host_id": [10, 10, 11, 12, 13, 13, 13, 14],
            "district": [
                "Mitte",
                "Mitte",
                "Pankow",
                "Neukölln",
                "Mitte",
                "Pankow",
                "Neukölln",
                "Mitte",
            ],
            "room_type": [
                "Entire home/apt",
                "Private room",
                "Entire home/apt",
                "Private room",
                "Entire home/apt",
                "Entire home/apt",
                "Shared room",
                "Entire home/apt",
            ],
            # listing 4 has no price; listings 5 and 8 are rented by the month (28+ nights)
            "price": [120.0, 60.0, 90.0, None, 200.0, 150.0, 30.0, 80.0],
            "minimum_nights": [2, 1, 3, 2, 30, 2, 1, 90],
            "number_of_reviews": [150, 0, 12, 3, 0, 40, 1, 0],
        }
    )
