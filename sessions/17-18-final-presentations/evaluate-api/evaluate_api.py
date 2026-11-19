"""Evaluate a team's subheading API on labelled decisions, the same way the final evaluation does.

    uv run evaluate_api.py --url http://localhost:8000 --data my_validation.csv

The data file (CSV is easiest to check by eye; Parquet works too) needs three columns:

    id            a unique name for each decision (for example its `bti_reference`)
    description   the text sent to the service: its only input
    subheading    the right answer, six digits, as text

With `--label heading` the right answer is in a column `heading` (four digits) and the service must answer
`heading`: this is the check of the practice sessions, which work on the four-digit heading.

Every description is sent to `POST /predict` one by
one. The script prints accuracy, macro-F1, the accuracy of the heading (the first four digits), the failed
requests with their reasons, and the 95th percentile of the response times, and it writes one row per
decision to a results file (`--output`, default `evaluation_results.csv`), so you can see which `id`
went wrong and what the service answered.

Students run it on their own validation decisions to check that the service answers the interface of the
final project (sessions/17-18-final-presentations/final-project.md). The lecturer runs it on the
independent evaluation set.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DIGITS = {"subheading": 6, "heading": 4}  # the label to score and its number of digits


def load(path: Path, label: str) -> pd.DataFrame:
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
    required = ("id", "description", label)
    missing = [column for column in required if column not in data.columns]
    if missing:
        sys.exit(f"{path} needs the columns {', '.join(required)}; missing: {missing}")
    if data["id"].duplicated().any():
        sys.exit(f"{path}: the column id must be unique, but {data['id'].duplicated().sum():,} ids repeat")
    return data.assign(id=data["id"].astype(str), **{label: data[label].astype(str).str.zfill(DIGITS[label])})


def ask(client: httpx.Client, description: str, label: str) -> tuple[str | None, str]:
    """Send one description; return the answer (or None) and a status that says what happened."""
    try:
        answer = client.post("/predict", json={"description": description})   # the description is the only input
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError:
        return None, "connection error"
    if answer.status_code != 200:
        return None, f"http {answer.status_code}"
    try:
        code = str(answer.json()[label]).zfill(DIGITS[label])
    except (KeyError, ValueError, TypeError):
        return None, f"no {label} in the answer"
    if len(code) != DIGITS[label] or not code.isdigit():
        return None, f"{label} is not {DIGITS[label]} digits"
    return code, "ok"


def predict_all(url: str, data: pd.DataFrame, label: str, timeout: float) -> pd.DataFrame:
    """One row per decision: id, the answer, the status and the seconds the request took."""
    rows = []
    with httpx.Client(base_url=url, timeout=timeout) as client:
        health = client.get("/health")
        if health.status_code != 200:
            sys.exit(f"GET /health answered {health.status_code}: is the service running at {url}?")
        for i, row in enumerate(data.itertuples(index=False), start=1):
            start = time.perf_counter()
            predicted, status = ask(client, row.description, label)
            rows.append({"id": row.id, "predicted": predicted, "status": status,
                         "seconds": time.perf_counter() - start})
            if i % 500 == 0:
                print(f"  {i:,} of {len(data):,} decisions", file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the service")
    parser.add_argument("--data", type=Path, required=True, help="CSV (or Parquet) with id, description and the label")
    parser.add_argument("--label", choices=sorted(DIGITS), default="subheading",
                        help="what the service answers: subheading (six digits, the final project) or heading (four)")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.csv"),
                        help="where to write one row per decision (default: evaluation_results.csv)")
    parser.add_argument("--limit", type=int, help="evaluate only the first N decisions (a quick check)")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for one answer")
    args = parser.parse_args()

    label = args.label
    data = load(args.data, label)
    if args.limit:
        data = data.head(args.limit)
    answers = predict_all(args.url, data, label, args.timeout)

    # one table, one row per decision; a failed request has no answer and counts as wrong
    results = data[["id", label]].rename(columns={label: "true"}).merge(answers, on="id")
    results["correct"] = results["predicted"] == results["true"]
    if label == "subheading":
        results["heading_correct"] = results["predicted"].str[:4] == results["true"].str[:4]
    results.round({"seconds": 4}).to_csv(args.output, index=False)

    scored = results["predicted"].fillna("none")
    print(f"decisions          {len(results):,}")
    print(f"failed requests    {(results['status'] != 'ok').sum():,}")
    print(f"accuracy           {results['correct'].mean():.4f}")
    print(f"macro-F1           {f1_score(results['true'], scored, average='macro', zero_division=0):.4f}")
    if label == "subheading":
        print(f"heading accuracy   {results['heading_correct'].mean():.4f}")
    print(f"latency P95        {np.percentile(results['seconds'], 95):.3f} s")

    print("\nwhat happened to the requests:")
    print(results["status"].value_counts().to_string())

    wrong = results[(results["status"] == "ok") & ~results["correct"]]
    if len(wrong):
        extra = f"; {wrong['heading_correct'].mean():.0%} of them have the right heading" if label == "subheading" else ""
        print(f"\n{len(wrong):,} wrong answers{extra}")
        print("most frequent confusions (true -> answered):")
        print(wrong.groupby(["true", "predicted"]).size().sort_values(ascending=False).head(5).to_string())
    print(f"\nevery decision with its answer: {args.output}")


if __name__ == "__main__":
    main()
