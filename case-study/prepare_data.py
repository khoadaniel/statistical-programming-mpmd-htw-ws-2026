"""Prepare the course dataset: European Binding Tariff Information (EBTI) decisions.

A Binding Tariff Information (BTI) decision is issued by the customs authority of an EU member
state on request of a trader: it states how a described product is classified in the customs
nomenclature. The European Commission publishes all decisions in the EBTI database. The task of
the course is to predict the four-digit HS heading of a decision from its description of goods.

The script downloads the official full export (about 400 MB, once), builds the course tables and
the leaderboard split, and writes them to case-study/data/.

    uv run python case-study/prepare_data.py
    uv run python case-study/prepare_data.py --postgres postgresql+psycopg://postgres:course@localhost/postgres

Split (by start date of validity):
    train          decisions 2017-2023, with CN code, heading and the customs' justification
    test           decisions 2024-2026, description and metadata only; new ids
    solution.csv   (lecturer only) headings of the test set; 2024 = public, 2025-2026 = private
Metric: accuracy (share of correct headings), with macro-F1 reported alongside.

Sources: European Commission, EBTI database (reuse notice: Commission Decision 2011/833/EU);
HS nomenclature from https://github.com/datasets/harmonized-system (ODC-PDDL-1.0).
"""

from __future__ import annotations

import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

EBTI_URL = "https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_export_management.jsp?message=extractFull"
HS_URL = "https://raw.githubusercontent.com/datasets/harmonized-system/main/data/{}.csv"
HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "data" / "raw", HERE / "data"
FIRST_YEAR, TEST_YEAR, PRIVATE_YEAR, LAST_YEAR = 2017, 2024, 2025, 2026
SEED = 2026

COLUMNS = {
    "BTI_REFERENCE": "bti_reference", "ISSUING_COUNTRY": "issuing_country", "LANGUAGE": "language",
    "START_DATE_OF_VALIDITY": "start_date", "END_DATE_OF_VALIDITY": "end_date", "DATE_OF _ISSUE": "date_of_issue",
    "STATUS": "status", "INVALIDATION_REASON": "invalidation_reason", "NOMENCLATURE_CODE": "nomenclature_code",
    "DESCRIPTION_OF_GOODS": "description", "KEYWORDS": "keywords",
    "CLASSIFICATION_JUSTIFICATION": "classification_justification",
}
CODE_LIKE = re.compile(r"\d[\d .]{2,}\d")


def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "DDS2-EBTI_Full.zip"
    if not target.exists():
        print("downloading the EBTI full export (about 400 MB) ...")
        urllib.request.urlretrieve(EBTI_URL, target)
    for name in ("harmonized-system", "sections"):
        if not (RAW / f"hs_{name}.csv").exists():
            urllib.request.urlretrieve(HS_URL.format(name), RAW / f"hs_{name}.csv")
    return target


def read_export(path: Path) -> pd.DataFrame:
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            df = pd.read_csv(z.open(name), usecols=list(COLUMNS), dtype=str, encoding="utf-8-sig")
            parts.append(df.rename(columns=COLUMNS))
    d = pd.concat(parts, ignore_index=True)
    for col in ("start_date", "end_date", "date_of_issue"):
        d[col] = pd.to_datetime(d[col], format="%d/%m/%Y", errors="coerce")
    digits = d["nomenclature_code"].str.replace(r"\D", "", regex=True)
    d["cn_code"], d["heading"], d["chapter"] = digits.str[:8], digits.str[:4], digits.str[:2]
    d["description"] = d["description"].fillna("").str.strip()
    return d.drop(columns="nomenclature_code")


def mask_own_code(description: str, cn_code: str) -> str:
    """Replace numbers that repeat the decision's own code (e.g. a quoted CN subheading) by <CODE>."""
    def repl(m: re.Match) -> str:
        num = re.sub(r"\D", "", m.group())
        return "<CODE>" if len(num) >= 4 and cn_code.startswith(num) else m.group()
    return CODE_LIKE.sub(repl, description)


def monthly_counts(d: pd.DataFrame) -> pd.DataFrame:
    month = d["start_date"].dt.to_period("M").dt.to_timestamp()
    return (d.assign(month=month).dropna(subset=["month"])
             .groupby(["month", "issuing_country", "chapter"]).size().rename("n_decisions").reset_index())


def nomenclature() -> pd.DataFrame:
    hs = pd.read_csv(RAW / "hs_harmonized-system.csv", dtype=str)
    sections = pd.read_csv(RAW / "hs_sections.csv", dtype=str, encoding="utf-8-sig")
    chapters = hs[hs["level"] == "2"][["hscode", "description"]].rename(columns={"hscode": "chapter", "description": "chapter_description"})
    headings = hs[hs["level"] == "4"][["section", "hscode", "description", "parent"]]
    headings = headings.rename(columns={"hscode": "heading", "description": "heading_description", "parent": "chapter"})
    return (headings.merge(chapters, on="chapter", how="left")
                    .merge(sections.rename(columns={"name": "section_name"}), on="section", how="left")
                    [["heading", "heading_description", "chapter", "chapter_description", "section", "section_name"]])


def build(path: Path) -> dict[str, pd.DataFrame]:
    d = read_export(path)
    counts = monthly_counts(d)
    # a few start dates are typing errors far in the future (e.g. 2200); keep 2017 up to the export year
    d = d[d["start_date"].dt.year.between(FIRST_YEAR, LAST_YEAR)].copy()
    d = d[d["heading"].str.len() == 4]
    d["description"] = [mask_own_code(t, c) for t, c in zip(d["description"], d["cn_code"])]
    d = d[d["description"].str.len() > 0].sort_values("start_date").reset_index(drop=True)

    train = d[d["start_date"].dt.year < TEST_YEAR].copy()
    test = d[d["start_date"].dt.year >= TEST_YEAR].copy()
    # a test description identical to a training description (renewed decisions) could be looked up
    norm = lambda s: s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()  # noqa: E731
    test = test[~norm(test["description"]).isin(set(norm(train["description"])))]
    rng = np.random.default_rng(SEED)
    test.insert(0, "id", [f"t{i:06d}" for i in rng.permutation(len(test))])
    solution = pd.DataFrame({"id": test["id"], "heading": test["heading"],
                             "Usage": np.where(test["start_date"].dt.year >= PRIVATE_YEAR, "Private", "Public")})
    # the test set holds only what a trader's request contains, plus where and when it was decided
    test = test[["id", "issuing_country", "language", "start_date", "description"]]
    train = train[["bti_reference", "issuing_country", "language", "start_date", "end_date", "date_of_issue",
                   "status", "invalidation_reason", "description", "keywords", "classification_justification",
                   "cn_code", "heading", "chapter"]]
    sample = train.sample(50_000, random_state=SEED).sort_values("start_date")
    return {"train": train, "train_sample": sample, "test": test, "nomenclature": nomenclature(),
            "monthly_counts": counts, "solution": solution}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres", help="SQLAlchemy URL; also load the tables into PostgreSQL")
    args = parser.parse_args()
    tables = build(download())
    for name, df in tables.items():
        if name == "solution":
            (OUT / "instructor").mkdir(exist_ok=True)
            df.to_csv(OUT / "instructor" / "solution.csv", index=False)
        else:
            df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:15s} {len(df):>9,} rows")
    if args.postgres:
        from sqlalchemy import create_engine
        engine = create_engine(args.postgres)
        for table, name in (("train", "decisions"), ("test", "decisions_test"), ("nomenclature", "nomenclature")):
            tables[table].to_sql(name, engine, if_exists="replace", index=False, chunksize=10_000)
        print("loaded decisions, decisions_test and nomenclature into PostgreSQL")


if __name__ == "__main__":
    main()
