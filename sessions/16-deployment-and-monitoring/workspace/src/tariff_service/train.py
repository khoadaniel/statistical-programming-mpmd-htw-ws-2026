"""Train version N of the heading model and write it to models/ with its metadata.

    uv run python -m tariff_service.train --data ../../../case-study/data/train_sample.parquet --version 1.0.0
    uv run python -m tariff_service.train --data ... --feedback ../../../case-study/data/feedback_2024.csv \\
        --version 2.0.0                                            # retraining with the released 2024 labels
    uv run python -m tariff_service.train --fixture                # tiny model for tests and CI smoke tests

Validation is time-based: the most recent 20 % of the decisions are held out, as in Session 7.
Inputs are the description only: justification, keywords and CN code are decided with the heading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import (
    build_pipeline,
    decision_text,
    evaluate,
    file_sha256,
    prune_coefficients,
    reference_stats,
    save_model,
    train_fixture_model,
)

COLUMNS = ["description", "language", "issuing_country", "start_date", "heading"]


def load_training_data(data: Path, feedback: Path | None = None, test: Path | None = None) -> pd.DataFrame:
    """Training decisions; optionally plus feedback labels (id,heading) joined to the test decisions."""
    df = pd.read_parquet(data, columns=COLUMNS)
    if feedback is not None:
        test = test or data.with_name("test.parquet")
        new = pd.read_parquet(test, columns=["id", *COLUMNS[:-1]]).merge(
            pd.read_csv(feedback, dtype=str), on="id", how="inner").drop(columns="id")
        df = pd.concat([df, new], ignore_index=True)
    return df.sort_values("start_date", kind="stable").reset_index(drop=True)


def load_heading_texts(data: Path) -> dict[str, str]:
    path = data.with_name("nomenclature.parquet")
    if not path.exists():
        return {}
    nom = pd.read_parquet(path, columns=["heading", "heading_description"])
    return dict(zip(nom["heading"], nom["heading_description"], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="parquet file with description, language, ..., heading")
    parser.add_argument("--feedback", type=Path, help="CSV id,heading of newly labelled test decisions")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--version", default="1.0.0", help="semantic version of the model")
    parser.add_argument("--format", choices=["joblib", "skops"], default="joblib")
    parser.add_argument("--prune", type=float, default=0.05, help="drop coefficients with |w| below this value")
    parser.add_argument("--fixture", action="store_true", help="train the tiny built-in fixture model")
    args = parser.parse_args()

    if args.fixture:
        print("wrote", train_fixture_model(args.out))
        return
    if args.data is None:
        parser.error("--data is required unless --fixture is given")

    df = load_training_data(args.data, args.feedback)
    texts = [decision_text(t) for t in df["description"]]
    labels = df["heading"].tolist()
    cut = int(len(df) * 0.8)  # time-based hold-out: the newest 20 %
    pipe = build_pipeline().fit(texts[:cut], labels[:cut])
    prune_coefficients(pipe, args.prune)
    scores = evaluate(pipe, texts[cut:], labels[cut:])
    print("validation on the newest 20 %:", scores)
    pipe = build_pipeline().fit(texts, labels)  # final model on all data
    kept = prune_coefficients(pipe, args.prune)  # small file, same accuracy (see model.prune_coefficients)
    heading_texts = load_heading_texts(args.data)
    meta = {"model_version": args.version,
            "training_data": {"file": args.data.name, "sha256": file_sha256(args.data), "n": len(df),
                              "feedback": args.feedback.name if args.feedback else None,
                              "date_min": str(df["start_date"].min()), "date_max": str(df["start_date"].max())},
            "validation": {"scheme": "time-based, newest 20 % held out", **scores},
            "pruning": {"threshold": args.prune, "weights_kept": round(kept, 4)},
            "reference": reference_stats(texts, labels, df["language"].tolist(), df["issuing_country"].tolist()),
            "headings": {h: heading_texts.get(h, "") for h in map(str, pipe.classes_)}}
    path = save_model(pipe, args.out, meta, fmt=args.format)
    print("wrote", path, f"({path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta["reference"]["chapter_shares"])[:200])


if __name__ == "__main__":
    main()
