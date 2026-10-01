"""Load and summarise the course data: EU Binding Tariff Information (BTI) decisions.

Reference solution, for self-checking only: copy this file over
src/btitools/decisions.py after you have tried the exercises yourself.

This module is the notebook analysis of Session 1 turned into reusable code:
one class, ``DecisionTable``, holds the decisions and answers the questions of the
case-study exercise through methods.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Columns every decision table must have; the class refuses tables without them.
REQUIRED_COLUMNS = (
    "bti_reference",
    "issuing_country",
    "language",
    "start_date",
    "end_date",
    "status",
    "description",
    "heading",
)


class DecisionTable:
    """A table of BTI decisions with methods that summarise it.

    Attributes:
        data: the decisions as a pandas DataFrame, one row per decision.

    Example:
        >>> table = DecisionTable.from_parquet("case-study/data/train_sample.parquet")
        >>> len(table)
        50000
    """

    def __init__(self, data: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"the decision table is missing the columns {missing}")
        self.data = data

    @classmethod
    def from_parquet(cls, path: str | Path) -> DecisionTable:
        """Read a Parquet file of decisions and return a ``DecisionTable``."""
        return cls(pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"DecisionTable({len(self)} decisions)"

    # Question 1 -------------------------------------------------------------
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Earliest and latest start date of validity."""
        return self.data["start_date"].min(), self.data["start_date"].max()

    def counts_by_year_and_country(self) -> pd.DataFrame:
        """Number of decisions per start year (rows) and issuing country (columns)."""
        year = self.data["start_date"].dt.year.rename("year")
        return pd.crosstab(year, self.data["issuing_country"])

    # Question 2 -------------------------------------------------------------
    def language_shares(self) -> pd.Series:
        """Share of decisions per language of the description, largest first."""
        return self.data["language"].value_counts(normalize=True)

    # Question 3 -------------------------------------------------------------
    def top_headings(self, n: int = 5, nomenclature: pd.DataFrame | None = None) -> pd.DataFrame:
        """The ``n`` most frequent headings with their count and share.

        If a nomenclature table (columns ``heading``, ``heading_description``) is given,
        the English description of each heading is joined to the result.
        """
        counts = self.data["heading"].value_counts()
        top = pd.DataFrame(
            {
                "heading": counts.index[:n].astype(str),
                "n_decisions": counts.to_numpy()[:n],
                "share": (counts.to_numpy()[:n] / len(self)).round(3),
            }
        )
        if nomenclature is not None:
            names = nomenclature[["heading", "heading_description"]]
            top = top.merge(names, on="heading", how="left")
        return top

    # Question 4 -------------------------------------------------------------
    def median_description_length_by_language(self) -> pd.Series:
        """Median number of characters of the description within each language."""
        lengths = self.data["description"].str.len()
        return lengths.groupby(self.data["language"]).median()

    # Question 5 -------------------------------------------------------------
    def valid_share_by_year(self) -> pd.Series:
        """Share of decisions with status ``VALID`` within each start year, indexed by year."""
        is_valid = self.data["status"] == "VALID"
        return is_valid.groupby(self.data["start_date"].dt.year.rename("year")).mean()

    # Summary ------------------------------------------------------------------
    def summary(self) -> dict[str, object]:
        """The answers to the case-study questions as a dictionary."""
        first, last = self.date_range()
        return {
            "n_decisions": len(self),
            "n_columns": self.data.shape[1],
            "first_start_date": first,
            "last_start_date": last,
            "decisions_per_year": self.counts_by_year_and_country().sum(axis=1).to_dict(),
            "language_shares": self.language_shares().head(5).round(3).to_dict(),
            "top_headings": self.top_headings(5)[["heading", "n_decisions"]].to_dict("records"),
            "median_description_length_by_language": (
                self.median_description_length_by_language().to_dict()
            ),
            "valid_share_by_year": self.valid_share_by_year().round(3).to_dict(),
        }
