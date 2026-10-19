"""Train version N of the heading model and write it to models/ with its metadata.

    uv run python -m tariff_service.train --data ../../../case-study/data/ebti/decisions.parquet --version 1.0.0
    uv run python -m tariff_service.train --data ../../../case-study/data/ebti/decisions.parquet \\
        --before 2026-01-01 --version 1.0.0               # simulate a model released at the start of 2026
    uv run python -m tariff_service.train --data ../../../case-study/data/ebti/decisions.parquet \\
        --version 2.0.0                                   # retraining with all decisions, 2026 included
    uv run python -m tariff_service.train --fixture                # tiny model for tests and CI smoke tests

Validation is time-based: the most recent 20 % of the decisions are held out, as in Session 7. This is one
possible validation design; it simulates deployment, where a model meets newer requests. Your team reports
the score of its own design (final project). `--before DATE` keeps only decisions that started before DATE,
to simulate an older model (Session 16). The English heading texts stored with the model come from the
nomenclature file next to the data (nomenclature/Nomenclature EN.xlsx).
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


def load_training_data(data: Path, before: str | None = None) -> pd.DataFrame:
    """Labelled decisions; with `before` (YYYY-MM-DD), only those that started before that date."""
    df = pd.read_parquet(data, columns=COLUMNS)
    if before is not None:
        df = df[df["start_date"] < pd.Timestamp(before)]
    return df.sort_values("start_date", kind="stable").reset_index(drop=True)


def load_heading_texts(data: Path, headings: list[str]) -> dict[str, str]:
    """English text of each heading, from the nomenclature file next to the data ({} if it is missing)."""
    path = data.parent / "nomenclature" / "Nomenclature EN.xlsx"
    if not path.exists():
        return {}
    taric = pd.read_excel(path, dtype=str).set_index("Goods code")["Description"]
    return {h: taric.get(f"{h}000000 80", "") for h in headings}  # the line of heading 6404 is "6404000000 80"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="decisions.parquet or a file with the same columns")
    parser.add_argument("--before", help="keep only training decisions that started before this date (YYYY-MM-DD)")
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

    df = load_training_data(args.data, before=args.before)
    texts = [decision_text(t) for t in df["description"]]
    labels = df["heading"].tolist()
    cut = int(len(df) * 0.8)  # time-based hold-out: the newest 20 %
    pipe = build_pipeline().fit(texts[:cut], labels[:cut])
    prune_coefficients(pipe, args.prune)
    scores = evaluate(pipe, texts[cut:], labels[cut:])
    print("validation on the newest 20 %:", scores)
    pipe = build_pipeline().fit(texts, labels)  # final model on all data
    kept = prune_coefficients(pipe, args.prune)  # small file, same accuracy (see model.prune_coefficients)
    heading_texts = load_heading_texts(args.data, [str(h) for h in pipe.classes_])
    meta = {"model_version": args.version,
            "training_data": {"file": args.data.name, "sha256": file_sha256(args.data), "n": len(df),
                              "before": args.before,
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
