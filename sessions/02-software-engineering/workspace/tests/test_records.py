"""Tests of DecisionRecord: valid input is cleaned, invalid input raises ValidationError."""

from datetime import date

import pytest

from btitools import BtiToolsError, DecisionRecord, ValidationError, chapter_of


def make(**changes):
    """A valid record with some fields replaced."""
    fields = {
        "bti_reference": "FR1",
        "issuing_country": "FR",
        "language": "fr",
        "start_date": "2023-01-02",
        "description": "Sac à main",
        "heading": "4202",
    }
    return DecisionRecord(**{**fields, **changes})


@pytest.mark.parametrize(
    ("heading", "expected"),
    [("0101", "01"), ("0901", "09"), ("9503", "95"), ("9706", "97")],
)
def test_chapter_of(heading, expected):
    assert chapter_of(heading) == expected


def test_from_dict_cleans_values(raw_decision):
    record = DecisionRecord.from_dict(raw_decision)
    assert record.bti_reference == "DE0001/23-1"
    assert record.issuing_country == "DE"
    assert record.language == "de"
    assert record.start_date == date(2023, 5, 10)
    assert record.description == "Plüschtier in Form eines Bären, Höhe 30 cm"
    assert record.keywords == "TOYS, PLUSH"
    assert record.chapter == "95"


def test_to_dict_is_json_ready(raw_decision):
    data = DecisionRecord.from_dict(raw_decision).to_dict()
    assert data["start_date"] == "2023-05-10"
    assert data["chapter"] == "95"
    assert "classification_justification" not in data


@pytest.mark.parametrize("heading", ["9503", " 9503 ", "95.03", "0901"])
def test_heading_accepts_four_digits(heading):
    assert make(heading=heading).heading in {"9503", "0901"}


@pytest.mark.parametrize("heading", [901, 9503, "950", "95031", "toys", "", None])
def test_heading_rejects_invalid_values(heading):
    with pytest.raises(ValidationError) as excinfo:
        make(heading=heading)
    assert excinfo.value.field == "heading"


@pytest.mark.parametrize("country", ["DEU", "D", "1A", ""])
def test_country_must_be_two_letters(country):
    with pytest.raises(ValidationError, match="issuing_country"):
        make(issuing_country=country)


@pytest.mark.parametrize("language", ["xx", "german", "", "en-GB"])
def test_language_must_be_an_eu_language(language):
    with pytest.raises(ValidationError, match="language"):
        make(language=language)


@pytest.mark.parametrize("start", ["2023-01-02", "02/01/2023", date(2023, 1, 2), " 02/01/2023 "])
def test_start_date_formats(start):
    assert make(start_date=start).start_date == date(2023, 1, 2)


@pytest.mark.parametrize("start", ["31/02/2023", "2023/01/02", "01/01/2200", "", 20230102])
def test_start_date_rejects_impossible_values(start):
    with pytest.raises(ValidationError, match="start_date"):
        make(start_date=start)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_description_is_rejected(text):
    with pytest.raises(ValidationError, match="description: must not be empty"):
        make(description=text)


def test_missing_required_key_is_reported(raw_decision):
    del raw_decision["heading"]
    with pytest.raises(ValidationError, match="heading: is missing"):
        DecisionRecord.from_dict(raw_decision)


def test_validation_error_is_part_of_the_hierarchy():
    # a caller can catch the package's base class or the built-in ValueError
    with pytest.raises(BtiToolsError):
        make(heading="95")
    with pytest.raises(ValueError):
        make(heading="95")


@pytest.mark.skip(reason="exercise 1: delete this line when end_date is validated")
@pytest.mark.parametrize(
    ("end", "valid"),
    [
        (None, True),
        ("2026-01-01", True),
        ("01/01/2026", True),
        ("2023-01-02", True),  # same day as the start: allowed
        ("2022-12-31", False),  # before the start: not a valid period
        ("31/13/2025", False),
        ("2300-01-01", False),
    ],
)
def test_end_date_rules(end, valid):
    if valid:
        record = make(end_date=end)
        assert record.end_date is None or record.end_date >= record.start_date
    else:
        with pytest.raises(ValidationError, match="end_date"):
            make(end_date=end)
