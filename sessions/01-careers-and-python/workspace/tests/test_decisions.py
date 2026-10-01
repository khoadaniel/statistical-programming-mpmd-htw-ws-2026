"""Tests for the finished parts of DecisionTable. They pass in the starter state."""

import pandas as pd
import pytest

from btitools import DecisionTable


def test_table_has_length_and_repr(tiny_decisions):
    table = DecisionTable(tiny_decisions)
    assert len(table) == 6
    assert repr(table) == "DecisionTable(6 decisions)"


def test_missing_column_is_rejected(tiny_decisions):
    with pytest.raises(ValueError, match="heading"):
        DecisionTable(tiny_decisions.drop(columns="heading"))


def test_date_range(tiny_decisions):
    first, last = DecisionTable(tiny_decisions).date_range()
    assert first == pd.Timestamp("2021-01-04")
    assert last == pd.Timestamp("2023-11-30")


def test_counts_by_year_and_country(tiny_decisions):
    counts = DecisionTable(tiny_decisions).counts_by_year_and_country()
    assert counts.loc[2021, "DE"] == 2
    assert counts.loc[2023, "PL"] == 1
    assert counts.to_numpy().sum() == 6


def test_language_shares_sum_to_one(tiny_decisions):
    shares = DecisionTable(tiny_decisions).language_shares()
    assert shares["de"] == pytest.approx(3 / 6)
    assert shares["fr"] == pytest.approx(2 / 6)
    assert shares.sum() == pytest.approx(1.0)


def test_top_headings(tiny_decisions):
    top = DecisionTable(tiny_decisions).top_headings(2)
    # heading 9503 occurs four times, 4202 and 3926 once each
    assert top.loc[0, "heading"] == "9503"
    assert top.loc[0, "n_decisions"] == 4
    assert top.loc[0, "share"] == pytest.approx(0.667)
    assert len(top) == 2


def test_top_headings_with_names(tiny_decisions):
    nomenclature = pd.DataFrame({"heading": ["9503"], "heading_description": ["Toys"]})
    top = DecisionTable(tiny_decisions).top_headings(1, nomenclature=nomenclature)
    assert top.loc[0, "heading_description"] == "Toys"


def test_summary_keys(tiny_decisions):
    summary = DecisionTable(tiny_decisions).summary()
    assert summary["n_decisions"] == 6
    assert summary["decisions_per_year"] == {2021: 3, 2023: 3}
    assert summary["top_headings"][0] == {"heading": "9503", "n_decisions": 4}


def test_from_parquet_round_trip(tiny_decisions, tmp_path):
    path = tmp_path / "decisions.parquet"
    tiny_decisions.to_parquet(path)
    assert len(DecisionTable.from_parquet(path)) == 6
