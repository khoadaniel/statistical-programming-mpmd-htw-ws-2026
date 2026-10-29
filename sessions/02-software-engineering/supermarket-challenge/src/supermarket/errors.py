"""The exceptions of the package.

All of them inherit from SupermarketError. A caller who wants to handle every problem of the shop
writes `except SupermarketError:`; a caller who wants to handle one problem names its class, for
example `except OutOfStockError:`.
"""


class SupermarketError(Exception):
    """Base class of every error that this package raises on purpose."""


class UnknownProductError(SupermarketError):
    """Raised when the shop does not sell a product with the name that was asked for."""


class OutOfStockError(SupermarketError):
    """Raised when fewer items are in stock than a customer wants to take."""


# Task 8: add the class PaymentError here.
