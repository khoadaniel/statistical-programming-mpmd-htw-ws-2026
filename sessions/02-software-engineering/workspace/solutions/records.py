"""A validated, cleaned record of one Airbnb listing in Berlin.

Reference solution, for self-checking only: compare with your own version after
you have tried the exercise. Copy it over src/listingtools/records.py to check it.

``ListingRecord`` is a dataclass: Python writes ``__init__``, ``__repr__`` and ``__eq__`` from
the annotated fields. ``__post_init__`` runs after ``__init__`` and is where the record checks
and cleans its own values, so that every ``ListingRecord`` that exists is valid. ``from_dict``
builds a record from raw input: a row of the raw Inside Airbnb file ``listings.csv.gz``, in
which the price is text such as ``"$1,083.00"``, or a JSON object sent to the web API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from listingtools.errors import ValidationError

#: The four room types used by Airbnb.
ROOM_TYPES = frozenset({"Entire home/apt", "Private room", "Shared room", "Hotel room"})

#: Bounding box of the city of Berlin (latitude, longitude), rounded outwards.
BERLIN_LATITUDE = (52.33, 52.68)
BERLIN_LONGITUDE = (13.08, 13.77)

#: Listings with a minimum stay of this many nights or more are rented by the month.
SHORT_STAY_MAX_NIGHTS = 28

#: The raw file writes this number (the largest 32-bit integer) when a host set no maximum stay.
NO_MAXIMUM = 2_147_483_647


def clean_text(value: Any, field: str) -> str:
    """Return ``value`` as a non-empty string with surrounding and repeated whitespace removed.

    Raises:
        ValidationError: if ``value`` is not a string or is empty.
    """
    if not isinstance(value, str):
        raise ValidationError(field, f"expected text, got {type(value).__name__}")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValidationError(field, "must not be empty")
    return cleaned


def parse_int(value: Any, field: str, *, minimum: int) -> int:
    """Convert ``value`` (an int or digits as text, ``"4"``) to an int of at least ``minimum``.

    Floats with a fractional part (``2.5``) and booleans are rejected; ``3.0`` becomes ``3``,
    because the raw file stores some whole numbers with a decimal point.
    """
    if isinstance(value, bool):
        raise ValidationError(field, "expected a whole number, got bool")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValidationError(field, f"expected a whole number, got {value!r}")
    if value < minimum:
        raise ValidationError(field, f"must be at least {minimum}, got {value}")
    return value


def parse_float(value: Any, field: str) -> float:
    """Convert a number or a number written as text (``"52.53574"``) to a finite float."""
    if isinstance(value, bool):
        raise ValidationError(field, "expected a number, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as err:
        raise ValidationError(field, f"expected a number, got {value!r}") from err
    if not math.isfinite(number):
        raise ValidationError(field, f"expected a finite number, got {value!r}")
    return number


def parse_price(value: Any) -> float | None:
    """Convert a price per night to a float; ``None`` and ``""`` mean "no price".

    Accepts numbers and the text of the raw file: ``"$1,083.00"`` -> ``1083.0`` (Inside Airbnb
    writes a dollar sign for every city; Berlin prices are in euros). Rejects text that is not
    a price (``"85 EUR"``, ``"free"``) and prices of zero or below.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        value = value.strip().lstrip("$€").replace(",", "")
    price = parse_float(value, "price")
    if price <= 0:
        raise ValidationError("price", f"must be above 0, got {price}")
    return price


def parse_coordinate(value: Any, field: str, bounds: tuple[float, float]) -> float:
    """A latitude or longitude that lies inside ``bounds`` (Berlin's bounding box)."""
    number = parse_float(value, field)
    low, high = bounds
    if not low <= number <= high:
        raise ValidationError(field, f"{number} lies outside Berlin ({low} to {high})")
    return number


def parse_room_type(value: Any) -> str:
    """One of the four room types; the comparison ignores case and extra spaces."""
    cleaned = clean_text(value, "room_type")
    for room_type in ROOM_TYPES:
        if cleaned.lower() == room_type.lower():
            return room_type
    raise ValidationError("room_type", f"expected one of {sorted(ROOM_TYPES)}, got {value!r}")


@dataclass
class ListingRecord:
    """One Airbnb listing whose fields have been checked and cleaned.

    Example:
        >>> r = ListingRecord(3176, " Fabulous  Flat ", "Pankow", 52.536, 13.417,
        ...                   "Entire home/apt", 2, "$160.71", minimum_nights=2)
        >>> r.name, r.price, r.price_per_guest, r.is_short_stay
        ('Fabulous Flat', 160.71, 80.36, True)
    """

    id: int
    name: str
    district: str
    latitude: float
    longitude: float
    room_type: str
    accommodates: int
    price: float | None = None
    minimum_nights: int = 1
    maximum_nights: int | None = None

    def __post_init__(self) -> None:
        self.id = parse_int(self.id, "id", minimum=1)
        self.name = clean_text(self.name, "name")
        self.district = clean_text(self.district, "district")
        self.latitude = parse_coordinate(self.latitude, "latitude", BERLIN_LATITUDE)
        self.longitude = parse_coordinate(self.longitude, "longitude", BERLIN_LONGITUDE)
        self.room_type = parse_room_type(self.room_type)
        self.accommodates = parse_int(self.accommodates, "accommodates", minimum=1)
        self.price = parse_price(self.price)
        self.minimum_nights = parse_int(self.minimum_nights, "minimum_nights", minimum=1)
        if self.maximum_nights == NO_MAXIMUM:
            self.maximum_nights = None
        if self.maximum_nights is not None:
            self.maximum_nights = parse_int(self.maximum_nights, "maximum_nights", minimum=1)
            if self.maximum_nights < self.minimum_nights:
                raise ValidationError(
                    "maximum_nights",
                    f"{self.maximum_nights} is below minimum_nights {self.minimum_nights}",
                )

    @property
    def is_short_stay(self) -> bool:
        """True if the listing can be booked for fewer than 28 nights."""
        return self.minimum_nights < SHORT_STAY_MAX_NIGHTS

    @property
    def price_per_guest(self) -> float | None:
        """Price per night divided by the number of guests, rounded to cents."""
        return None if self.price is None else round(self.price / self.accommodates, 2)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ListingRecord:
        """Build a record from a mapping such as a JSON object or a row of the raw file.

        The district may be given as ``district`` or, as in the raw file, as
        ``neighbourhood_group_cleansed``. Unknown keys are ignored; missing optional keys
        get their defaults.

        Raises:
            ValidationError: if a required key is missing or a value breaks a rule.
        """
        data = dict(raw)
        if "district" not in data and "neighbourhood_group_cleansed" in data:
            data["district"] = data["neighbourhood_group_cleansed"]
        required = ("id", "name", "district", "latitude", "longitude", "room_type", "accommodates")
        for key in required:
            if key not in data:
                raise ValidationError(key, "is missing")
        return cls(
            **{key: data[key] for key in required},
            price=data.get("price"),
            minimum_nights=data.get("minimum_nights", 1),
            maximum_nights=data.get("maximum_nights"),
        )

    def to_dict(self) -> dict[str, Any]:
        """The cleaned record as a JSON-ready dictionary, including the derived fields."""
        return {
            **asdict(self),
            "is_short_stay": self.is_short_stay,
            "price_per_guest": self.price_per_guest,
        }
