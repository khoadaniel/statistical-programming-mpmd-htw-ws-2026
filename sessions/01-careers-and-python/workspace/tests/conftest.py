"""Shared test data: eight listings small enough to check every answer by hand."""

import pandas as pd
import pytest


@pytest.fixture
def tiny_listings() -> pd.DataFrame:
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
