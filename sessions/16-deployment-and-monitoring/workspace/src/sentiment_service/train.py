"""Train version N of the sentiment model and write it to models/ with its metadata.

    uv run python -m sentiment_service.train --data ../../../case-study/data/train_sample.parquet --version 1.0.0
    uv run python -m sentiment_service.train --data ... --feedback feedback_2022.csv --version 2.0.0   # retraining
    uv run python -m sentiment_service.train --fixture            # tiny model for tests and CI smoke tests

Validation is time-based: the most recent 20 % of the reviews are held out, as in Session 7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import build_pipeline, evaluate, file_sha256, reference_stats, review_text, save_model, train_fixture_model


def load_training_data(data: Path, feedback: Path | None = None, test: Path | None = None) -> pd.DataFrame:
    """Training reviews; optionally plus feedback labels (review_id,label) joined to the test reviews."""
    df = pd.read_parquet(data, columns=["review_id", "title", "text", "date", "label"])
    if feedback is not None:
        test = test or data.with_name("test.parquet")
        new = pd.read_parquet(test, columns=["review_id", "title", "text", "date"]).merge(
            pd.read_csv(feedback), on="review_id", how="inner")
        df = pd.concat([df, new], ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="parquet file with review_id, title, text, date, label")
    parser.add_argument("--feedback", type=Path, help="CSV review_id,label of newly labelled reviews")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--version", default="1.0.0", help="semantic version of the model")
    parser.add_argument("--format", choices=["joblib", "skops"], default="joblib")
    parser.add_argument("--fixture", action="store_true", help="train the tiny built-in fixture model")
    args = parser.parse_args()

    if args.fixture:
        print("wrote", train_fixture_model(args.out))
        return
    if args.data is None:
        parser.error("--data is required unless --fixture is given")

    df = load_training_data(args.data, args.feedback)
    texts = [review_text(t, x) for t, x in zip(df["title"], df["text"], strict=True)]
    labels = df["label"].tolist()
    cut = int(len(df) * 0.8)  # time-based hold-out: the newest 20 %
    pipe = build_pipeline().fit(texts[:cut], labels[:cut])
    scores = evaluate(pipe, texts[cut:], labels[cut:])
    print("validation on the newest 20 %:", scores)
    pipe = build_pipeline().fit(texts, labels)  # final model on all data
    meta = {"model_version": args.version,
            "training_data": {"file": args.data.name, "sha256": file_sha256(args.data), "n": len(df),
                              "feedback": args.feedback.name if args.feedback else None,
                              "date_min": str(df["date"].min()), "date_max": str(df["date"].max())},
            "validation": {"scheme": "time-based, newest 20 % held out", **scores},
            "reference": reference_stats(texts, labels)}
    path = save_model(pipe, args.out, meta, fmt=args.format)
    print("wrote", path, f"({path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta["reference"]["label_shares"]))


if __name__ == "__main__":
    main()
