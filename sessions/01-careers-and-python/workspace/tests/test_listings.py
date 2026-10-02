"""Tests for the finished parts of ListingTable. They pass in the starter state."""

import pytest

from listingtools import ListingTable


def test_table_has_length_and_repr(tiny_listings):
    table = ListingTable(tiny_listings)
    assert len(table) == 8
    assert repr(table) == "ListingTable(8 listings)"


def test_missing_column_is_rejected(tiny_listings):
    with pytest.raises(ValueError, match="price"):
        ListingTable(tiny_listings.drop(columns="price"))


def test_n_hosts(tiny_listings):
    # hosts 10, 11, 12, 13 and 14; host 13 has three listings
    assert ListingTable(tiny_listings).n_hosts() == 5


def test_by_district(tiny_listings):
    districts = ListingTable(tiny_listings).by_district()
    assert districts.index[0] == "Mitte"
    assert districts.loc["Mitte", "n_listings"] == 4
    assert districts.loc["Pankow", "share"] == pytest.approx(0.25)
    assert districts["n_listings"].sum() == 8


def test_room_type_shares_sum_to_one(tiny_listings):
    shares = ListingTable(tiny_listings).room_type_shares()
    assert shares["Entire home/apt"] == pytest.approx(5 / 8)
    assert shares["Shared room"] == pytest.approx(1 / 8)
    assert shares.sum() == pytest.approx(1.0)


def test_short_stays_drop_missing_prices_and_monthly_rentals(tiny_listings):
    short = ListingTable(tiny_listings).short_stays()
    assert isinstance(short, ListingTable)
    assert sorted(short.data["id"]) == [1, 2, 3, 6, 7]


def test_price_by_room_type(tiny_listings):
    prices = ListingTable(tiny_listings).price_by_room_type()
    # entire homes among the short stays: 120, 90 and 150 -> median 120
    assert prices.loc["Entire home/apt", "n_listings"] == 3
    assert prices.loc["Entire home/apt", "median_price"] == pytest.approx(120.0)
    assert prices.loc["Shared room", "median_price"] == pytest.approx(30.0)
    assert prices.index[0] == "Entire home/apt"


def test_summary_keys(tiny_listings):
    summary = ListingTable(tiny_listings).summary()
    assert summary["n_listings"] == 8
    assert summary["n_short_stays"] == 5
    assert summary["median_short_stay_price"] == pytest.approx(90.0)
    assert summary["listings_per_district"]["Mitte"] == 4


def test_from_parquet_round_trip(tiny_listings, tmp_path):
    path = tmp_path / "listings.parquet"
    tiny_listings.to_parquet(path)
    assert len(ListingTable.from_parquet(path)) == 8
