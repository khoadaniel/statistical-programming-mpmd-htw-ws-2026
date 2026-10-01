"""Command-line entry point: print the summary of a file of BTI decisions.

Run from the repository root (the default paths are relative to it):

    uv run --project sessions/01-careers-and-python/workspace decision-summary
    uv run --project sessions/01-careers-and-python/workspace decision-summary other.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import pandas as pd

from btitools.decisions import DecisionTable

DEFAULT_PATH = "case-study/data/train_sample.parquet"
NOMENCLATURE_PATH = "case-study/data/nomenclature.parquet"


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, load the decisions and print their summary."""
    parser = argparse.ArgumentParser(description="Summarise a Parquet file of BTI decisions.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="Parquet file of decisions")
    args = parser.parse_args(argv)

    table = DecisionTable.from_parquet(args.path)
    print(table)
    pprint(table.summary(), sort_dicts=False)
    if Path(NOMENCLATURE_PATH).exists():
        top = table.top_headings(5, nomenclature=pd.read_parquet(NOMENCLATURE_PATH))
        top["heading_description"] = top["heading_description"].str.slice(0, 50)
        print("\nMost frequent headings:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
