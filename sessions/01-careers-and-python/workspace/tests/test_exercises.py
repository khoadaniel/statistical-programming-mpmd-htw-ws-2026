"""Tests for the exercises. Delete a ``skip`` line once you have done the exercise."""

import pytest

from btitools import DecisionTable


@pytest.mark.skip(reason="exercise 2: delete this line when the method is done")
def test_median_description_length_by_language(tiny_decisions):
    medians = DecisionTable(tiny_decisions).median_description_length_by_language()
    # de: 10, 13 and 13 characters; fr: 3 and 6; pl: 5
    assert medians.to_dict() == {"de": 13.0, "fr": 4.5, "pl": 5.0}


@pytest.mark.skip(reason="exercise 3: delete this line when the method is done")
def test_valid_share_by_year(tiny_decisions):
    shares = DecisionTable(tiny_decisions).valid_share_by_year()
    # 2021: none of three is valid; 2023: two of three are valid
    assert shares[2021] == pytest.approx(0.0)
    assert shares[2023] == pytest.approx(2 / 3)


@pytest.mark.skip(reason="exercise 4: delete this line when summary() is extended")
def test_summary_contains_all_answers(tiny_decisions):
    summary = DecisionTable(tiny_decisions).summary()
    assert "median_description_length_by_language" in summary
    assert "valid_share_by_year" in summary
