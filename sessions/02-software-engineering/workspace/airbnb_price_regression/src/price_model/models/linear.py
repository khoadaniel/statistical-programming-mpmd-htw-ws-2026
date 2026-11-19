"""The simplest price model: one straight line, price = slope * accommodates + intercept."""

import numpy as np
from sklearn.linear_model import LinearRegression

from price_model.models.base import BasePriceModel
from price_model.schemas import ModelCoefficients


class LinearPriceModel(BasePriceModel):
    """A linear regression with one feature (the number of guests) and no tuning.

    It inherits from ``BasePriceModel`` (inheritance), so it gets ``save`` and ``load`` for free
    and has to write ``fit`` and ``predict_price`` itself.
    """

    def __init__(self) -> None:
        """Create a model that is not fitted yet."""
        # A leading underscore means "internal": other code should not touch it (encapsulation).
        # Other code uses fit(), predict_price() and the properties below instead.
        self._estimator = LinearRegression()  # scikit-learn's ordinary least squares, defaults only
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """True once ``fit`` has run. A property is read like an attribute: ``model.is_fitted``."""
        return self._is_fitted

    @property
    def coefficients(self) -> ModelCoefficients:
        """The slope and intercept of the fitted line.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._require_fitted()
        return ModelCoefficients(
            # coef_ is an array with one number per feature; we have one feature, so take [0].
            slope=float(self._estimator.coef_[0]),
            intercept=float(self._estimator.intercept_),
        )

    def fit(self, accommodates: np.ndarray, prices: np.ndarray) -> None:
        """Fit the line to the training listings (see ``BasePriceModel.fit``)."""
        # scikit-learn wants the features as a table (2 dimensions): one row per listing,
        # one column per feature. reshape(-1, 1) turns [2, 4, 6] into [[2], [4], [6]].
        feature_table = accommodates.reshape(-1, 1)
        self._estimator.fit(feature_table, prices)
        self._is_fitted = True

    def predict_price(self, accommodates: int) -> float:
        """Predict the price per night for one listing (see ``BasePriceModel.predict_price``).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._require_fitted()
        predictions = self._estimator.predict([[accommodates]])  # a table with one row, one column
        return float(predictions[0])

    def _require_fitted(self) -> None:
        """Stop with a clear message if the model was never fitted (a private helper method)."""
        if not self._is_fitted:
            raise RuntimeError("The model is not fitted yet. Call fit() or load a checkpoint.")
