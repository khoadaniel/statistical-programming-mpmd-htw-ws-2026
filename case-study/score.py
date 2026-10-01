"""Score a leaderboard submission (CSV with columns id,heading) against the hidden solution.

    uv run python case-study/score.py submission.csv
Prints accuracy and macro-F1 on the public (2024) and private (2025-2026) part of the test set.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SOLUTION = Path(__file__).resolve().parent / "data" / "instructor" / "solution.csv"


def score(submission: pd.DataFrame, solution: pd.DataFrame) -> dict[str, float]:
    if set(submission.columns) != {"id", "heading"}:
        raise ValueError("submission needs exactly the columns id,heading")
    submission = submission.assign(heading=submission["heading"].astype(str).str.zfill(4))
    if not submission["heading"].str.fullmatch(r"\d{4}").all():
        raise ValueError("heading must be a four-digit HS heading such as 3926")
    if submission["id"].duplicated().any():
        raise ValueError("each id may appear only once")
    merged = solution.merge(submission, on="id", how="left", suffixes=("", "_pred"))
    if merged["heading_pred"].isna().any():
        raise ValueError(f"{merged['heading_pred'].isna().sum()} test decisions have no prediction")
    out = {}
    for usage, part in merged.groupby("Usage"):
        out[f"{usage.lower()}_accuracy"] = accuracy_score(part["heading"], part["heading_pred"])
        out[f"{usage.lower()}_macro_f1"] = f1_score(part["heading"], part["heading_pred"], average="macro")
    return out


if __name__ == "__main__":
    result = score(pd.read_csv(sys.argv[1], dtype=str), pd.read_csv(SOLUTION, dtype=str))
    for k, v in result.items():
        print(f"{k:20s} {v:.4f}")
