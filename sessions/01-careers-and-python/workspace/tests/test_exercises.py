"""Tests for the exercises. Delete a ``skip`` line once you have done the exercise."""

import pytest

from listingtools import ListingTable


@pytest.mark.skip(reason="exercise 2: delete this line when the method is done")
def test_median_price_by_district(tiny_listings):
    medians = ListingTable(tiny_listings).median_price_by_district()
    # short stays: Mitte 120 and 60, Pankow 90 and 150, Neukölln 30
    assert medians.to_dict() == {"Neukölln": 30.0, "Mitte": 90.0, "Pankow": 120.0}
    assert list(medians.index) == ["Neukölln", "Mitte", "Pankow"]


@pytest.mark.skip(reason="exercise 3: delete this line when the method is done")
def test_review_summary(tiny_listings):
    reviews = ListingTable(tiny_listings).review_summary()
    # sorted reviews 0, 0, 0, 1, 3, 12, 40, 150: the median lies between 1 and 3
    assert reviews["median_reviews"] == pytest.approx(2.0)
    assert reviews["share_without_reviews"] == pytest.approx(3 / 8)


@pytest.mark.skip(reason="exercise 4: delete this line when summary() is extended")
def test_summary_contains_all_answers(tiny_listings):
    summary = ListingTable(tiny_listings).summary()
    assert "median_price_by_district" in summary
    assert "reviews" in summary
