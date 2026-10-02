"""Lecturer only: release the 2024 labels as feedback data for the Session 16 practice.

    uv run python sessions/16-deployment-and-monitoring/workbooks/make_feedback_2024.py
    uv run python sessions/16-deployment-and-monitoring/workbooks/make_feedback_2024.py --out /tmp/feedback_2024.csv

Reads the hidden solution (case-study/data/instructor/solution.csv) and writes the columns id,heading for
the decisions of 2024 (the public leaderboard part) to case-study/data/feedback_2024.csv, or to the path
given with --out. Share this file with the students; the 2025-2026 labels stay hidden for the final ranking.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "case-study" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solution", type=Path, default=DATA / "instructor" / "solution.csv")
    parser.add_argument("--out", type=Path, default=DATA / "feedback_2024.csv")
    args = parser.parse_args()
    solution = pd.read_csv(args.solution, dtype=str)
    feedback = solution.loc[solution["Usage"] == "Public", ["id", "heading"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feedback.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(feedback):,} labelled decisions of 2024")
    print("most frequent headings:", feedback["heading"].value_counts(normalize=True).head(3).round(3).to_dict())


if __name__ == "__main__":
    main()
