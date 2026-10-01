"""Exceptions of the package, organised as a small class hierarchy.

ReviewToolsError            base class: catch this to handle any error of the package
├── ValidationError         a record breaks a rule (also a ValueError)
└── APIError                a web API answered with an error
    └── RateLimitError      the API said "too many requests" (HTTP 429)
"""

from __future__ import annotations


class ReviewToolsError(Exception):
    """Base class of all errors raised by reviewtools."""


class ValidationError(ReviewToolsError, ValueError):
    """A record does not satisfy the rules of its class.

    Attributes:
        field: name of the offending field, e.g. ``"rating"``.
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")


class APIError(ReviewToolsError):
    """A web API returned an error status or an unexpected payload."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(APIError):
    """The API refused the request because too many requests were sent (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message, status_code=429)
