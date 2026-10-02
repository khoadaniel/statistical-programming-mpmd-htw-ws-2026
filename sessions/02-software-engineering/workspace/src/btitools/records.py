"""A validated, cleaned record of one Binding Tariff Information (BTI) decision.

``DecisionRecord`` is a dataclass: Python writes ``__init__``, ``__repr__`` and ``__eq__``
from the annotated fields. ``__post_init__`` runs after ``__init__`` and is where the
record checks and cleans its own values, so that every ``DecisionRecord`` that exists is
valid. ``from_dict`` builds a record from raw input (a JSON object, a CSV row of the EBTI
export).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from btitools.errors import ValidationError

#: Language codes of the 24 official EU languages (ISO 639-1).
LANGUAGES = frozenset(
    "bg cs da de el en es et fi fr ga hr hu it lt lv mt nl pl pt ro sk sl sv".split()
)

#: Plausible years of a start or end date; the raw export contains typing errors such as 2200.
MIN_YEAR, MAX_YEAR = 1990, 2035

_HEADING = re.compile(r"\d{4}")
_COUNTRY = re.compile(r"[A-Z]{2}")


def chapter_of(heading: str) -> str:
    """The two-digit HS chapter of a four-digit heading: ``"0901"`` -> ``"09"``."""
    return parse_heading(heading)[:2]


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


def parse_heading(value: Any) -> str:
    """Return a four-digit HS heading as text.

    Accepts ``"9503"`` and ``" 95.03 "``; rejects integers (``901`` has lost the leading
    zero of ``"0901"``), ``"950"``, ``"95031"`` and ``"toys"``.
    """
    if not isinstance(value, str):
        raise ValidationError("heading", f"expected text, got {type(value).__name__}")
    cleaned = value.strip().replace(".", "").replace(" ", "")
    if not _HEADING.fullmatch(cleaned):
        raise ValidationError("heading", f"expected four digits, got {value!r}")
    return cleaned


def parse_country(value: Any) -> str:
    """Return a two-letter country code in upper case (``" de "`` -> ``"DE"``)."""
    cleaned = clean_text(value, "issuing_country", required=True).upper()
    if not _COUNTRY.fullmatch(cleaned):
        raise ValidationError("issuing_country", f"expected two letters, got {value!r}")
    return cleaned


def parse_language(value: Any) -> str:
    """Return an EU language code in lower case (``"DE"`` -> ``"de"``)."""
    cleaned = clean_text(value, "language", required=True).lower()
    if cleaned not in LANGUAGES:
        raise ValidationError("language", f"not an EU language code: {value!r}")
    return cleaned


def parse_date(value: Any, field: str) -> date:
    """Convert ``value`` to a date.

    Accepts a ``date``/``datetime``, ISO text (``"2023-05-10"``) and the format of the
    EBTI export (``"10/05/2023"``, day first). Rejects impossible dates such as
    ``"31/02/2023"`` and implausible years such as 2200.
    """
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValidationError(field, f"expected YYYY-MM-DD or DD/MM/YYYY, got {value!r}")
    else:
        raise ValidationError(field, f"expected a date, got {type(value).__name__}")
    if not MIN_YEAR <= parsed.year <= MAX_YEAR:
        raise ValidationError(field, f"implausible year {parsed.year}")
    return parsed


@dataclass
class DecisionRecord:
    """One BTI decision whose fields have been checked and cleaned.

    Example:
        >>> r = DecisionRecord(" DE123 ", "DE", "de", "10/05/2023", "Plush  toy ", "9503")
        >>> r.bti_reference, r.issuing_country, r.start_date, r.description, r.chapter
        ('DE123', 'DE', datetime.date(2023, 5, 10), 'Plush toy', '95')
    """

    bti_reference: str
    issuing_country: str
    language: str
    start_date: date
    description: str
    heading: str
    end_date: date | None = None
    keywords: str = ""

    def __post_init__(self) -> None:
        self.bti_reference = clean_text(self.bti_reference, "bti_reference", required=True)
        self.issuing_country = parse_country(self.issuing_country)
        self.language = parse_language(self.language)
        self.start_date = parse_date(self.start_date, "start_date")
        self.description = clean_text(self.description, "description", required=True)
        self.heading = parse_heading(self.heading)
        self.keywords = clean_text(self.keywords, "keywords", required=False)
        # TODO (exercise 1): validate end_date. It is optional: None stays None.
        # Otherwise convert it with parse_date(self.end_date, "end_date") and raise
        # ValidationError("end_date", "...") if it lies before start_date (the EBTI
        # export gives annulled decisions the placeholder 1900-01-01). Then delete the
        # skip marker of test_end_date_rules in tests/test_records.py.

    @property
    def chapter(self) -> str:
        """The two-digit HS chapter, derived from the heading."""
        return self.heading[:2]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DecisionRecord:
        """Build a record from a mapping such as a JSON object or a CSV row.

        Unknown keys are ignored; missing optional keys get their defaults.

        Raises:
            ValidationError: if a required key is missing or a value breaks a rule.
        """
        required = ("bti_reference", "issuing_country", "language", "start_date")
        for key in (*required, "description", "heading"):
            if key not in raw:
                raise ValidationError(key, "is missing")
        return cls(
            bti_reference=raw["bti_reference"],
            issuing_country=raw["issuing_country"],
            language=raw["language"],
            start_date=raw["start_date"],
            description=raw["description"],
            heading=raw["heading"],
            end_date=raw.get("end_date"),
            keywords=raw.get("keywords", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """The cleaned record as a JSON-ready dictionary, including the derived chapter."""
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        end = self.end_date
        data["end_date"] = end.isoformat() if isinstance(end, date) else end
        return {**data, "chapter": self.chapter}
