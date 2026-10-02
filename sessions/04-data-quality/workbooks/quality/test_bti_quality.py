"""Validation rules for the BTI decisions of the case study, written as pytest tests.

Run from the repository root:
    uv run --with pytest pytest sessions/04-data-quality/workbooks/quality -v

Tests on the prepared training table encode hard rules (keys, formats, consistency). Known problems of the
data are marked `xfail` with the reason: they are documented, and pytest reports them as "expected
failures". Tests on the cleaned table run after workbook 14 has written `decisions_clean.parquet`;
otherwise they are skipped. Author: course team, licence CC-BY-4.0.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "case-study").is_dir())
DATA = ROOT / "case-study" / "data"
CLEAN = Path(os.environ.get("CLEAN_OUT", DATA / "decisions_clean.parquet"))

if not (DATA / "train.parquet").exists():
    pytest.skip("case-study data missing: run case-study/prepare_data.py", allow_module_level=True)


@pytest.fixture(scope="module")
def decisions():
    return pd.read_parquet(DATA / "train.parquet")


@pytest.fixture(scope="module")
def nomenclature():
    return pd.read_parquet(DATA / "nomenclature.parquet")


@pytest.fixture(scope="module")
def clean():
    if not CLEAN.exists():
        pytest.skip("cleaned table not found: run workbook 14 first")
    return pd.read_parquet(CLEAN)


# ---------- training table: hard rules ----------

def test_bti_reference_is_primary_key(decisions):
    assert decisions["bti_reference"].notna().all()
    assert decisions["bti_reference"].is_unique


def test_heading_is_primary_key_of_nomenclature(nomenclature):
    assert nomenclature["heading"].is_unique


def test_heading_has_four_digits(decisions):
    assert decisions["heading"].str.fullmatch(r"\d{4}").all()


def test_chapter_matches_heading(decisions):
    assert (decisions["chapter"] == decisions["heading"].str[:2]).all()


def test_cn_code_starts_with_heading(decisions):
    assert (decisions["cn_code"].str[:4] == decisions["heading"]).all()


def test_codes_have_two_letters(decisions):
    assert decisions["issuing_country"].str.fullmatch(r"[A-Z]{2}").all()
    assert decisions["language"].str.fullmatch(r"[a-z]{2}").all()


def test_status_values(decisions):
    assert decisions["status"].isin(["VALID", "INVALID"]).all()


def test_valid_decisions_have_no_invalidation_reason(decisions):
    valid = decisions["status"] == "VALID"
    assert decisions.loc[valid, "invalidation_reason"].isna().all()


def test_start_dates_in_training_period(decisions):
    assert decisions["start_date"].between("2017-01-01", "2023-12-31").all()


def test_issued_before_validity_starts(decisions):
    assert (decisions["date_of_issue"] <= decisions["start_date"]).all()


@pytest.mark.parametrize("column", ["description", "heading", "start_date", "issuing_country", "language"])
def test_required_columns_complete(decisions, column):
    assert decisions[column].notna().all()


# ---------- training table: known problems (documented, expected to fail) ----------

@pytest.mark.xfail(strict=True, reason="known: 510 annulled decisions carry the placeholder end date 1900-01-01")
def test_raw_end_not_before_start(decisions):
    assert (decisions["end_date"] >= decisions["start_date"]).all()


@pytest.mark.xfail(strict=True, reason="known: 51 decisions use heading 8803, deleted in HS 2022")
def test_raw_heading_in_nomenclature(decisions, nomenclature):
    assert decisions["heading"].isin(nomenclature["heading"]).all()


@pytest.mark.xfail(strict=True, reason="known: 1,040 CN codes have only 4 or 6 digits")
def test_raw_cn_code_has_eight_digits(decisions):
    assert decisions["cn_code"].str.fullmatch(r"\d{8}").all()


@pytest.mark.xfail(strict=True, reason="known: 73 descriptions are a database template (SQL{...})")
def test_raw_no_template_descriptions(decisions):
    assert not decisions["description"].str.upper().str.contains("SQL{", regex=False).any()


# ---------- cleaned table (after workbook 14) ----------

def test_clean_no_template_descriptions(clean):
    assert not clean["description"].str.upper().str.contains("SQL{", regex=False).any()


def test_clean_no_placeholder_dates(clean):
    assert not clean["end_date"].dt.year.eq(1900).any()
    known = clean["end_date"].notna()
    assert (clean.loc[known, "end_date"] >= clean.loc[known, "start_date"]).all()


def test_clean_annulled_flag_matches_code_55(clean):
    assert (clean["annulled"] == clean["invalidation_reason"].eq("55")).all()


def test_clean_text_normalised(clean):
    assert not clean["description"].str.contains(" ", regex=False).any()
    assert not clean["description"].str.contains("\r", regex=False).any()


def test_clean_has_flags_and_indicators(clean):
    for col in ["annulled", "heading_outdated", "cn_code_short", "is_duplicate_description",
                "language_unexpected", "keywords_missing", "outlier_length_mad", "outlier_mahalanobis"]:
        assert col in clean.columns
        assert clean[col].dtype == bool


def test_clean_keywords_indicator_consistent(clean):
    assert (clean["keywords_missing"] == clean["keywords"].isna()).all()
