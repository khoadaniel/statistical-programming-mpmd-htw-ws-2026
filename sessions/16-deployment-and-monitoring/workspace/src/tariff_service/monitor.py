"""Drift report: compare a batch of new decisions with the reference stored in the model metadata.

    uv run python -m tariff_service.monitor --current ../../../case-study/data/ebti/decisions.parquet \\
        --since 2026-01-01 --hide-labels                  # inputs only, as before the labels arrive
    uv run python -m tariff_service.monitor --current ../../../case-study/data/ebti/decisions.parquet \\
        --since 2026-01-01                                # with the labels of 2026

Input drift (description length, language and country shares) and prediction drift (predicted chapter
shares) need no labels; label shift, unseen headings and the accuracy check need the labels of the new
period, which arrive later (here: the headings of the decisions of 2026, for a model trained with
`--before 2026-01-01`). A file with a `heading` column brings its own labels; `--hide-labels` ignores them.
`--since DATE` keeps only decisions that started on or after DATE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import bin_shares, label_shift, psi_categories, psi_from_shares, retrain_needed, unseen_share
from .model import decision_text, load_model


def drift_report(pipe, meta: dict, texts: list[str], labels: list[str] | None = None,
                 languages: list[str] | None = None, countries: list[str] | None = None,
                 tolerance: float = 0.03) -> dict:
    ref = meta["reference"]
    lengths = np.log1p([len(t) for t in texts])
    edges = np.array(ref["log_length_quantiles"][1:-1])
    psi_length = psi_from_shares(np.full(len(edges) + 1, 1 / (len(edges) + 1)), bin_shares(lengths, edges))
    inputs = {"psi_log_length": round(psi_length, 4), "median_length": float(np.median([len(t) for t in texts]))}
    if languages is not None and "language_shares" in ref:
        inputs["psi_language"] = round(psi_categories(ref["language_shares"], languages), 4)
    if countries is not None and "country_shares" in ref:
        inputs["psi_country"] = round(psi_categories(ref["country_shares"], countries), 4)
        inputs["countries_missing_now"] = sorted(set(ref["country_shares"]) - set(countries))[:10]
    pred = [str(p) for p in pipe.predict(texts)]
    report = {"model_version": meta["model_version"], "n_current": len(texts), "input": inputs,
              "prediction": {"psi_predicted_chapters": round(psi_categories(ref["chapter_shares"],
                                                                            [p[:2] for p in pred]), 4)}}
    acc_recent = None
    if labels is not None:
        reference_chapters = [c for c, s in ref["chapter_shares"].items() for _ in range(round(s * 10_000))]
        report["label_shift_chapters"] = label_shift(reference_chapters, [y[:2] for y in labels]).as_dict(top=5)
        report["unseen_headings_share"] = round(unseen_share(meta["classes"], labels), 4)
        acc_recent = round(float(np.mean([p == y for p, y in zip(pred, labels, strict=True)])), 4)
        report["accuracy_recent"] = acc_recent
    psi_input = max(v for k, v in inputs.items() if k.startswith("psi_"))
    decision, reasons = retrain_needed(acc_recent, meta["validation"]["accuracy"], psi_input, tolerance=tolerance)
    report["retrain"] = {"decision": decision, "reasons": reasons}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--current", type=Path, required=True, help="parquet with description, language, ...")
    parser.add_argument("--hide-labels", action="store_true", help="ignore the heading column: inputs only")
    parser.add_argument("--since", help="keep only decisions that started on or after this date (YYYY-MM-DD)")
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()
    pipe, meta = load_model(args.model_dir)
    df = pd.read_parquet(args.current)
    if args.since is not None:
        df = df[df["start_date"] >= pd.Timestamp(args.since)]
    labels = None
    if "heading" in df.columns and not args.hide_labels:  # e.g. decisions.parquet: labelled decisions
        labels = df["heading"].tolist()
    texts = [decision_text(t) for t in df["description"]]
    report = drift_report(pipe, meta, texts, labels, df["language"].tolist(), df["issuing_country"].tolist(),
                          args.tolerance)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
