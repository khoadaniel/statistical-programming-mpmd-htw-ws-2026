"""The training step: from the listings file to a saved checkpoint."""

from pathlib import Path

import pandas as pd

from price_model.config import (
    FEATURE_COLUMN,
    MAX_PRICE_EUR,
    SHORT_STAY_MAX_NIGHTS,
    TARGET_COLUMN,
)
from price_model.models import BasePriceModel
from price_model.schemas import ModelCoefficients


class PriceModelTrainer:
    """Trains a price model on the Berlin listings and saves it as a checkpoint."""

    def __init__(self, model: BasePriceModel, data_path: Path, checkpoint_path: Path) -> None:
        """Store what the trainer needs.

        The trainer receives the model from outside instead of creating it, so it works with any
        subclass of ``BasePriceModel`` (it only uses the abstract contract).

        Args:
            model: An unfitted price model.
            data_path: The parquet file with the Inside Airbnb listings.
            checkpoint_path: Where to write the fitted model.
        """
        self._model = model
        self._data_path = data_path
        self._checkpoint_path = checkpoint_path

    def train(self) -> int:
        """Load the data, fit the model, save the checkpoint.

        Returns:
            The number of listings the model was trained on.
        """
        listings = self._load_listings()
        self._model.fit(
            accommodates=listings[FEATURE_COLUMN].to_numpy(),  # pandas column -> NumPy array
            prices=listings[TARGET_COLUMN].to_numpy(),
        )
        self._model.save(self._checkpoint_path)
        return len(listings)

    def _load_listings(self) -> pd.DataFrame:
        """Read the file and keep the listings with a nightly price: the rows one night can use."""
        listings = pd.read_parquet(self._data_path, columns=self._columns_to_read())
        has_price = listings[TARGET_COLUMN].notna()  # a Boolean mask: True where a price exists
        is_short_stay = listings["minimum_nights"] < SHORT_STAY_MAX_NIGHTS
        is_plausible_price = listings[TARGET_COLUMN] <= MAX_PRICE_EUR
        return listings[has_price & is_short_stay & is_plausible_price]

    @staticmethod
    def _columns_to_read() -> list[str]:
        """The columns we need from the file (reading fewer columns is faster)."""
        return [FEATURE_COLUMN, TARGET_COLUMN, "minimum_nights"]


def describe(coefficients: ModelCoefficients, n_listings: int) -> str:
    """Return a one-line summary of the trained line, for the terminal."""
    return (
        f"Trained on {n_listings} listings: "
        f"price = {coefficients.slope:.2f} * accommodates + {coefficients.intercept:.2f}"
    )
