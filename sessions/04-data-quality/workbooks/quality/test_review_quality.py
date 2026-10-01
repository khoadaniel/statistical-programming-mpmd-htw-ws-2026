"""Validation rules for the review data, written as pytest tests.

Run from the repository root:
    uv run --with pytest pytest sessions/04-data-quality/workbooks/quality -v

Tests on the raw data encode hard rules (keys, ranges, consistency). Known problems of the raw data are
marked `xfail` with the reason: they are documented, and pytest reports them as "expected failures".
Tests on the cleaned table run after workbook 14 has written `reviews_clean.parquet`; otherwise they are
skipped. Author: course team, licence CC-BY-4.0.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "case-study").is_dir())
DATA = ROOT / "case-study" / "data"
CLEAN = Path(os.environ.get("CLEAN_OUT", DATA / "reviews_clean.parquet"))

if not (DATA / "train.parquet").exists():
    pytest.skip("case-study data missing: run case-study/prepare_data.py", allow_module_level=True)


@pytest.fixture(scope="module")
def reviews():
    return pd.read_parquet(DATA / "train.parquet")


@pytest.fixture(scope="module")
def products():
    return pd.read_parquet(DATA / "products.parquet")


@pytest.fixture(scope="module")
def clean():
    if not CLEAN.exists():
        pytest.skip("cleaned table not found: run workbook 14 first")
    return pd.read_parquet(CLEAN)


# ---------- raw data: hard rules ----------

def test_review_id_is_primary_key(reviews):
    assert reviews["review_id"].notna().all()
    assert reviews["review_id"].is_unique


def test_product_id_is_primary_key(products):
    assert products["parent_asin"].is_unique


def test_foreign_key(reviews, products):
    missing = ~reviews["parent_asin"].isin(products["parent_asin"])
    assert missing.sum() == 0, f"{missing.sum()} reviews refer to unknown products"


def test_rating_range(reviews):
    assert reviews["rating"].between(1, 5).all()


def test_label_matches_rating(reviews):
    expected = pd.cut(reviews["rating"], bins=[0, 2, 3, 5], labels=["neg", "neu", "pos"]).astype(str)
    assert (reviews["label"] == expected).all()


def test_counts_not_negative(reviews):
    assert (reviews[["helpful_vote", "n_images"]] >= 0).all().all()


def test_dates_in_training_period(reviews):
    assert reviews["date"].between("2001-01-01", "2021-12-31 23:59:59").all()


def test_price_positive_where_known(products):
    assert (products["price"].dropna() > 0).all()


@pytest.mark.parametrize("column", ["rating", "text", "parent_asin", "user_id", "date", "label"])
def test_required_columns_complete(reviews, column):
    assert reviews[column].notna().all()


# ---------- raw data: known problems (documented, expected to fail) ----------

@pytest.mark.xfail(strict=True, reason="known: 95 reviews have an empty text")
def test_raw_text_not_empty(reviews):
    assert reviews["text"].str.strip().ne("").all()


@pytest.mark.xfail(strict=True, reason="known: 537 user-product pairs have more than one review")
def test_raw_one_review_per_user_and_product(reviews):
    assert not reviews.duplicated(["user_id", "parent_asin"]).any()


# ---------- cleaned table (after workbook 14) ----------

def test_clean_text_not_empty(clean):
    assert clean["text"].str.strip().ne("").all()


def test_clean_one_review_per_user_and_product(clean):
    assert not clean.duplicated(["user_id", "parent_asin"]).any()


def test_clean_no_html_breaks(clean):
    assert not clean["text"].str.contains("<br", regex=False).any()


def test_clean_has_flags_and_indicators(clean):
    for col in ["is_duplicate_text", "outlier_length_mad", "outlier_mahalanobis", "price_missing"]:
        assert col in clean.columns
        assert clean[col].dtype == bool


def test_clean_price_indicator_consistent(clean):
    assert (clean["price_missing"] == clean["price"].isna()).all()
