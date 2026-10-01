"""Tests of ReviewRecord: valid input is cleaned, invalid input raises ValidationError."""

import pytest

from reviewtools import ReviewRecord, ReviewToolsError, ValidationError, to_label


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1, "neg"), (2, "neg"), (3, "neu"), (4, "pos"), (5, "pos")],
)
def test_to_label(rating, expected):
    assert to_label(rating) == expected


def test_from_dict_cleans_values(raw_review):
    record = ReviewRecord.from_dict(raw_review)
    assert record.review_id == "r000042"
    assert record.rating == 2
    assert record.title == "Disappointed"
    assert record.text == "Stopped working after two weeks."
    assert record.verified_purchase is True
    assert record.label == "neg"


def test_to_dict_contains_label(raw_review):
    data = ReviewRecord.from_dict(raw_review).to_dict()
    assert data["label"] == "neg"
    assert "unused_column" not in data


@pytest.mark.parametrize("rating", [4, 4.0, "4", " 4 "])
def test_rating_accepts_whole_numbers(rating):
    assert ReviewRecord("r1", rating, "ok").rating == 4


@pytest.mark.parametrize("rating", [0, 6, 4.5, "four", "", True, None])
def test_rating_rejects_invalid_values(rating):
    with pytest.raises(ValidationError) as excinfo:
        ReviewRecord("r1", rating, "ok")
    assert excinfo.value.field == "rating"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(text):
    with pytest.raises(ValidationError, match="text: must not be empty"):
        ReviewRecord("r1", 5, text)


def test_missing_required_key_is_reported(raw_review):
    del raw_review["rating"]
    with pytest.raises(ValidationError, match="rating: is missing"):
        ReviewRecord.from_dict(raw_review)


def test_validation_error_is_part_of_the_hierarchy():
    # a caller can catch the package's base class or the built-in ValueError
    with pytest.raises(ReviewToolsError):
        ReviewRecord("r1", 9, "ok")
    with pytest.raises(ValueError):
        ReviewRecord("r1", 9, "ok")


@pytest.mark.parametrize("value", ["maybe", 2, None])
def test_verified_purchase_rejects_unclear_values(value):
    with pytest.raises(ValidationError, match="verified_purchase"):
        ReviewRecord("r1", 5, "ok", verified_purchase=value)


@pytest.mark.skip(reason="exercise 1: delete this line when helpful_vote is validated")
@pytest.mark.parametrize(
    ("value", "valid"),
    [(0, True), (3, True), ("3", True), (-1, False), (2.5, False), ("many", False), (True, False)],
)
def test_helpful_vote_rules(value, valid):
    if valid:
        assert ReviewRecord("r1", 5, "ok", helpful_vote=value).helpful_vote == int(value)
    else:
        with pytest.raises(ValidationError, match="helpful_vote"):
            ReviewRecord("r1", 5, "ok", helpful_vote=value)
