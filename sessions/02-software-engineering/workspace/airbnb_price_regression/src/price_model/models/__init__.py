"""The models: an abstract base class and the linear regression that implements it."""

from price_model.models.base import BasePriceModel
from price_model.models.linear import LinearPriceModel

# __all__ lists the names that `from price_model.models import *` and the docs treat as public.
__all__ = ["BasePriceModel", "LinearPriceModel"]
