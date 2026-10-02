"""Tests of ListingRecord: valid input is cleaned, invalid input raises ValidationError."""

import pytest

from listingtools import ListingRecord, ListingToolsError, ValidationError, parse_price
from listingtools.records import NO_MAXIMUM


def make(**changes):
    """A valid record with some fields replaced."""
    fields = {
        "id": 1,
        "name": "Room with balcony",
        "district": "Neukölln",
        "latitude": 52.48,
        "longitude": 13.43,
        "room_type": "Private room",
        "accommodates": 2,
        "price": 80.0,
        "minimum_nights": 2,
    }
    return ListingRecord(**{**fields, **changes})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$160.71", 160.71), ("$1,083.00", 1083.0), (" $85 ", 85.0), ("€49.50", 49.5), (97, 97.0)],
)
def test_parse_price(raw, expected):
    assert parse_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_missing_price_becomes_none(raw):
    assert parse_price(raw) is None


@pytest.mark.parametrize("raw", ["85 EUR", "free", "$0.00", -10, "nan", True])
def test_parse_price_rejects_invalid_values(raw):
    with pytest.raises(ValidationError, match="price"):
        parse_price(raw)


def test_from_dict_cleans_values(raw_listing):
    record = ListingRecord.from_dict(raw_listing)
    assert record.id == 3176
    assert record.name == "Fabulous Flat in great Location"
    assert record.district == "Pankow"
    assert record.latitude == pytest.approx(52.53574)
    assert record.room_type == "Entire home/apt"
    assert record.price == pytest.approx(1160.5)
    assert record.minimum_nights == 2


def test_derived_fields():
    record = make(price=90.0, accommodates=3, minimum_nights=30)
    assert record.price_per_guest == pytest.approx(30.0)
    assert record.is_short_stay is False
    assert make(price=None).price_per_guest is None


def test_to_dict_is_json_ready(raw_listing):
    data = ListingRecord.from_dict(raw_listing).to_dict()
    assert data["price"] == pytest.approx(1160.5)
    assert data["is_short_stay"] is True
    assert "host_name" not in data


@pytest.mark.parametrize(
    ("field", "value"),
    [("latitude", 52.30), ("latitude", 48.14), ("longitude", 13.80), ("longitude", "east")],
)
def test_coordinates_must_lie_in_berlin(field, value):
    with pytest.raises(ValidationError) as excinfo:
        make(**{field: value})
    assert excinfo.value.field == field


@pytest.mark.parametrize("room_type", ["Entire home", "Apartment", "", None, 3])
def test_room_type_must_be_known(room_type):
    with pytest.raises(ValidationError, match="room_type"):
        make(room_type=room_type)


@pytest.mark.parametrize("accommodates", [1, 2, "4", 16, 3.0])
def test_accommodates_accepts_whole_numbers(accommodates):
    assert make(accommodates=accommodates).accommodates >= 1


@pytest.mark.parametrize("accommodates", [0, -1, 2.5, "two", True, None])
def test_accommodates_rejects_invalid_values(accommodates):
    with pytest.raises(ValidationError, match="accommodates"):
        make(accommodates=accommodates)


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_empty_name_is_rejected(name):
    with pytest.raises(ValidationError, match="name: must not be empty"):
        make(name=name)


def test_missing_required_key_is_reported(raw_listing):
    del raw_listing["room_type"]
    with pytest.raises(ValidationError, match="room_type: is missing"):
        ListingRecord.from_dict(raw_listing)


def test_validation_error_is_part_of_the_hierarchy():
    # a caller can catch the package's base class or the built-in ValueError
    with pytest.raises(ListingToolsError):
        make(accommodates=0)
    with pytest.raises(ValueError):
        make(accommodates=0)


@pytest.mark.skip(reason="exercise 1: delete this line when maximum_nights is validated")
@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (None, None),
        (365, 365),
        ("1125", 1125),
        (2, 2),  # same as the minimum stay: allowed
        (NO_MAXIMUM, None),  # the sentinel of the raw file means "no limit"
    ],
)
def test_maximum_nights_rules(maximum, expected):
    assert make(minimum_nights=2, maximum_nights=maximum).maximum_nights == expected


@pytest.mark.skip(reason="exercise 1: delete this line when maximum_nights is validated")
@pytest.mark.parametrize("maximum", [1, 0, "a week", 2.5])
def test_maximum_nights_rejects_invalid_values(maximum):
    with pytest.raises(ValidationError, match="maximum_nights"):
        make(minimum_nights=2, maximum_nights=maximum)
