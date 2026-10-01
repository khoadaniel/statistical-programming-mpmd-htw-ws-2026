"""A validated, cleaned review record.

``ReviewRecord`` is a dataclass: Python writes ``__init__``, ``__repr__`` and ``__eq__``
from the annotated fields. ``__post_init__`` runs after ``__init__`` and is where the
record checks and cleans its own values, so that every ``ReviewRecord`` that exists is
valid. ``from_dict`` builds a record from raw input (a JSON object, a CSV row).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from reviewtools.errors import ValidationError

#: The three sentiment classes of the course leaderboard.
LABELS = ("neg", "neu", "pos")

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0", ""}


def to_label(rating: int) -> str:
    """Map a star rating (1-5) to a sentiment label: 1-2 neg, 3 neu, 4-5 pos."""
    if rating <= 2:
        return "neg"
    if rating == 3:
        return "neu"
    return "pos"


def clean_text(value: Any, field: str, *, required: bool) -> str:
    """Return ``value`` as a string with surrounding and repeated whitespace removed.

    Raises:
        ValidationError: if ``value`` is not a string, or is empty and ``required``.
    """
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValidationError(field, f"expected text, got {type(value).__name__}")
    cleaned = " ".join(value.split())
    if required and not cleaned:
        raise ValidationError(field, "must not be empty")
    return cleaned


def parse_rating(value: Any) -> int:
    """Convert ``value`` to a star rating between 1 and 5.

    Accepts ``4``, ``4.0`` and ``"4"``; rejects ``True``, ``4.5``, ``"four"`` and ``6``.
    """
    if isinstance(value, bool):
        raise ValidationError("rating", "expected a number, got bool")
    if isinstance(value, str):
        value = value.strip()
        if not value.lstrip("-").isdigit():
            raise ValidationError("rating", f"expected a whole number, got {value!r}")
        value = int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValidationError("rating", f"expected a whole number, got {value}")
        value = int(value)
    if not isinstance(value, int):
        raise ValidationError("rating", f"expected a number, got {type(value).__name__}")
    if not 1 <= value <= 5:
        raise ValidationError("rating", f"must lie between 1 and 5, got {value}")
    return value


def parse_bool(value: Any, field: str) -> bool:
    """Convert ``True``/``False``, ``1``/``0`` or text such as ``"yes"`` to a bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
        return value.strip().lower() in _TRUE
    raise ValidationError(field, f"expected true or false, got {value!r}")


@dataclass
class ReviewRecord:
    """One product review whose fields have been checked and cleaned.

    Example:
        >>> r = ReviewRecord(review_id=" r1 ", rating="2", text="Broke  after a week ")
        >>> r.review_id, r.rating, r.text, r.label
        ('r1', 2, 'Broke after a week', 'neg')
    """

    review_id: str
    rating: int
    text: str
    title: str = ""
    helpful_vote: int = 0
    verified_purchase: bool = False

    def __post_init__(self) -> None:
        self.review_id = clean_text(self.review_id, "review_id", required=True)
        self.rating = parse_rating(self.rating)
        self.text = clean_text(self.text, "text", required=True)
        self.title = clean_text(self.title, "title", required=False)
        self.verified_purchase = parse_bool(self.verified_purchase, "verified_purchase")
        # TODO (exercise 1): validate helpful_vote. It must be a whole number >= 0;
        # accept 3 and "3", reject -1, 2.5, "many" and True by raising
        # ValidationError("helpful_vote", "..."). Then delete the skip marker of
        # test_helpful_vote_rules in tests/test_records.py.

    @property
    def label(self) -> str:
        """Sentiment label derived from the rating (neg, neu or pos)."""
        return to_label(self.rating)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReviewRecord:
        """Build a record from a mapping such as a JSON object or a CSV row.

        Unknown keys are ignored; missing optional keys get their defaults.

        Raises:
            ValidationError: if a required key is missing or a value breaks a rule.
        """
        for key in ("review_id", "rating", "text"):
            if key not in raw:
                raise ValidationError(key, "is missing")
        return cls(
            review_id=raw["review_id"],
            rating=raw["rating"],
            text=raw["text"],
            title=raw.get("title", ""),
            helpful_vote=raw.get("helpful_vote", 0),
            verified_purchase=raw.get("verified_purchase", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """The cleaned record as a dictionary, including the derived label."""
        return {**asdict(self), "label": self.label}
