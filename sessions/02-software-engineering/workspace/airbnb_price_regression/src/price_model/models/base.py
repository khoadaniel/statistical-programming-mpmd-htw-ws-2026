"""The abstract base class of every price model: the contract plus the shared save and load."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import joblib  # joblib writes Python objects to a file; scikit-learn installs it for us
import numpy as np


class BasePriceModel(ABC):
    """What every price model must be able to do.

    ABC makes this class abstract (abstraction): you cannot create ``BasePriceModel()`` directly,
    only a subclass that writes every method marked ``@abstractmethod``. The methods without that
    mark (``save``, ``load``) are written once here and inherited by every subclass (inheritance).
    """

    @abstractmethod
    def fit(self, accommodates: np.ndarray, prices: np.ndarray) -> None:
        """Learn the relation between the number of guests and the price.

        Args:
            accommodates: Number of guests of each training listing, one value per listing.
            prices: Price per night in EUR of each training listing, same length as accommodates.
        """

    @abstractmethod
    def predict_price(self, accommodates: int) -> float:
        """Return the predicted price per night in EUR for a listing that sleeps this many guests."""

    def save(self, checkpoint_path: Path) -> None:
        """Write the fitted model to ``checkpoint_path`` so that the API can load it later."""
        # parents=True creates missing folders; exist_ok=True does not fail if it already exists.
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, checkpoint_path)  # stores this whole object, fitted numbers included

    @classmethod
    def load(cls, checkpoint_path: Path) -> Self:
        """Read a model from ``checkpoint_path``.

        A classmethod receives the class (``cls``) instead of an object (``self``), so it can be
        called without an object: ``LinearPriceModel.load(path)``. ``Self`` is the type hint for
        "an object of the class this was called on".

        Only load files that you trained yourself: a joblib file can run code when it is opened.

        Raises:
            FileNotFoundError: If there is no file at ``checkpoint_path``.
            TypeError: If the file holds an object of a different class.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Train a model first.")
        model = joblib.load(checkpoint_path)
        if not isinstance(model, cls):
            raise TypeError(
                f"{checkpoint_path} holds a {type(model).__name__}, not a {cls.__name__}."
            )
        return model
