"""Drift report: compare a batch of new reviews with the reference stored in the model metadata.

    uv run python -m sentiment_service.monitor --current ../../../case-study/data/test.parquet
    uv run python -m sentiment_service.monitor --current new.parquet --labels feedback_2022.csv --f1-tolerance 0.03

Input drift (text length, PSI) and prediction drift need no labels; label shift and the F1 check need
the labels of the new period, which usually arrive later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import LABELS
from .drift import bin_shares, label_shift, psi_from_shares, retrain_needed
from .model import load_model, review_text


def drift_report(pipe, meta: dict, texts: list[str], labels: list[str] | None = None,
                 f1_tolerance: float = 0.03) -> dict:
    ref = meta["reference"]
    lengths = np.log1p([len(t) for t in texts])
    edges = np.array(ref["log_length_quantiles"][1:-1])
    psi_length = psi_from_shares(np.full(len(edges) + 1, 1 / (len(edges) + 1)), bin_shares(lengths, edges))
    pred = pipe.predict(texts).tolist()
    pred_shares = {c: round(pred.count(c) / len(pred), 4) for c in LABELS}
    report = {"model_version": meta["model_version"], "n_current": len(texts),
              "input": {"psi_log_length": round(psi_length, 4),
                        "median_length": float(np.median([len(t) for t in texts]))},
              "prediction_shares": {"reference_labels": ref["label_shares"], "current_predicted": pred_shares}}
    f1_recent = None
    if labels is not None:
        ref_labels = [c for c in LABELS for _ in range(round(ref["label_shares"][c] * 10_000))]
        report["label_shift"] = label_shift(ref_labels, labels).as_dict()
        f1_recent = round(float(f1_score(labels, pred, average="macro")), 4)
        report["macro_f1_recent"] = f1_recent
    decision, reasons = retrain_needed(f1_recent, meta["validation"]["macro_f1"], psi_length, tolerance=f1_tolerance)
    report["retrain"] = {"decision": decision, "reasons": reasons}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--current", type=Path, required=True, help="parquet with review_id, title, text")
    parser.add_argument("--labels", type=Path, help="CSV review_id,label for (part of) the current reviews")
    parser.add_argument("--f1-tolerance", type=float, default=0.03)
    args = parser.parse_args()
    pipe, meta = load_model(args.model_dir)
    df = pd.read_parquet(args.current, columns=["review_id", "title", "text"])
    labels = None
    if args.labels:
        df = df.merge(pd.read_csv(args.labels), on="review_id")
        labels = df["label"].tolist()
    texts = [review_text(t, x) for t, x in zip(df["title"], df["text"], strict=True)]
    print(json.dumps(drift_report(pipe, meta, texts, labels, args.f1_tolerance), indent=2))


if __name__ == "__main__":
    main()
