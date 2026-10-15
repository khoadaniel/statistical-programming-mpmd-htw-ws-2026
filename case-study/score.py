"""Score a leaderboard submission (CSV with columns id,heading) for the EBTI test set.

    uv run python case-study/score.py submission.csv

The test labels are not part of the student data: only the lecturer has them, in
case-study/lecturer/held_out/ebti/test_labels.csv, and scores the submissions for the leaderboard. Students
check their models on the validation set (case-study/data/ebti/val.parquet), which has labels.

score() also works for any table of true labels with the columns id and heading, and reports the
Public and Private halves separately when the table has a Usage column.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TEST_LABELS = Path(__file__).resolve().parent / "lecturer" / "held_out" / "ebti" / "test_labels.csv"


def score(submission: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    if set(submission.columns) != {"id", "heading"}:
        raise ValueError("submission needs exactly the columns id,heading")
    submission = submission.assign(heading=submission["heading"].astype(str).str.zfill(4))
    if not submission["heading"].str.fullmatch(r"\d{4}").all():
        raise ValueError("heading must be a four-digit HS heading such as 3926")
    if submission["id"].duplicated().any():
        raise ValueError("each id may appear only once")
    merged = labels.merge(submission, on="id", how="left", suffixes=("", "_pred"))
    if merged["heading_pred"].isna().any():
        raise ValueError(f"{merged['heading_pred'].isna().sum()} decisions have no prediction")
    if "Usage" not in merged:
        merged["Usage"] = "all"
    out = {}
    for usage, part in merged.groupby("Usage"):
        out[f"{usage.lower()}_accuracy"] = accuracy_score(part["heading"], part["heading_pred"])
        out[f"{usage.lower()}_macro_f1"] = f1_score(part["heading"], part["heading_pred"], average="macro")
    return out


if __name__ == "__main__":
    if not TEST_LABELS.exists():
        sys.exit("The test labels are held by the lecturer: upload your submission to the leaderboard. "
                 "To check a model yourself, score it on case-study/data/ebti/val.parquet.")
    result = score(pd.read_csv(sys.argv[1], dtype=str), pd.read_csv(TEST_LABELS, dtype=str))
    for k, v in result.items():
        print(f"{k:20s} {v:.4f}")
