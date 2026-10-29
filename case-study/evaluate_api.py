"""Evaluate a team's heading API on labelled decisions, the same way the final evaluation does.

    uv run python case-study/evaluate_api.py --url http://localhost:8000 --data my_validation.parquet

The data file (Parquet or CSV) needs a `description` and a `heading` column. Every description is sent to
`POST /predict` one by one, as the only input; the script prints accuracy and macro-F1 against `heading`,
how many requests failed, and the 95th percentile of the response times.

Students run it on their own validation decisions to check that the service answers the interface
of the final project (sessions/17-18-final-presentations/final-project.md). The lecturer runs it on the
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


def load(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
    missing = {"description", "heading"} - set(data.columns)
    if missing:
        sys.exit(f"{path} needs the columns description and heading; missing: {sorted(missing)}")
    return data.assign(heading=data["heading"].astype(str).str.zfill(4))


def predict_all(url: str, data: pd.DataFrame, timeout: float) -> tuple[list[str | None], list[float]]:
    headings, seconds = [], []
    with httpx.Client(base_url=url, timeout=timeout) as client:
        health = client.get("/health")
        if health.status_code != 200:
            sys.exit(f"GET /health answered {health.status_code}: is the service running at {url}?")
        for i, row in enumerate(data.itertuples(index=False), start=1):
            body = {"description": row.description}                # the description is the only input
            start = time.perf_counter()
            try:
                answer = client.post("/predict", json=body)
                answer.raise_for_status()
                heading = str(answer.json()["heading"]).zfill(4)
            except (httpx.HTTPError, KeyError, ValueError):
                heading = None
            seconds.append(time.perf_counter() - start)
            headings.append(heading)
            if i % 500 == 0:
                print(f"  {i:,} of {len(data):,} decisions", file=sys.stderr)
    return headings, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the service")
    parser.add_argument("--data", type=Path, required=True, help="Parquet or CSV with description and heading")
    parser.add_argument("--limit", type=int, help="evaluate only the first N decisions (a quick check)")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for one answer")
    args = parser.parse_args()

    data = load(args.data)
    if args.limit:
        data = data.head(args.limit)
    predicted, seconds = predict_all(args.url, data, args.timeout)
    # a failed request counts as a wrong answer
    pred = pd.Series(predicted, index=data.index).fillna("none")
    print(f"decisions          {len(data):,}")
    print(f"failed requests    {(pred == 'none').sum():,}")
    print(f"accuracy           {accuracy_score(data['heading'], pred):.4f}")
    print(f"macro-F1           {f1_score(data['heading'], pred, average='macro', zero_division=0):.4f}")
    print(f"latency P95        {np.percentile(seconds, 95):.3f} s")


if __name__ == "__main__":
    main()
