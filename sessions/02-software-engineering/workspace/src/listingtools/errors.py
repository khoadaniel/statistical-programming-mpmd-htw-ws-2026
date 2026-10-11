"""Exceptions of the package, organised as a small class hierarchy.

ListingToolsError           base class: catch this to handle any error of the package
└── ValidationError         a record breaks a rule (also a ValueError)
"""

from __future__ import annotations


class ListingToolsError(Exception):
    """Base class of all errors raised by listingtools."""


class ValidationError(ListingToolsError, ValueError):
    """A record does not satisfy the rules of its class.

    Attributes:
        field: name of the offending field, e.g. ``"price"``.
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")
