"""Lecturer only: release the 2022 labels as feedback data for the Session 16 practice.

    uv run --with pandas --with pyarrow python sessions/16-deployment-and-monitoring/workbooks/make_feedback_2022.py

Reads the hidden solution (case-study/data/instructor/solution.csv) and writes
case-study/data/feedback_2022.csv with the columns review_id,label for the 2022 reviews (the public
leaderboard part). Share this file with the students; the 2023 labels stay hidden for the final ranking.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
solution = pd.read_csv(ROOT / "case-study" / "data" / "instructor" / "solution.csv")
feedback = solution.loc[solution["Usage"] == "Public", ["review_id", "label"]]
out = ROOT / "case-study" / "data" / "feedback_2022.csv"
feedback.to_csv(out, index=False)
print(f"wrote {out} with {len(feedback):,} labelled reviews of 2022")
print(feedback["label"].value_counts(normalize=True).round(3).to_dict())
