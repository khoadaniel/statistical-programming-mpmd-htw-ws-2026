"""The model artefact: a fitted scikit-learn pipeline plus a metadata file.

    models/
      model.joblib     the fitted pipeline (TF-IDF vocabulary, idf weights, coefficients)
      metadata.json    version, library versions, training data, validation scores, reference statistics

The metadata make the file reproducible and give monitoring its reference: the label shares and the
text-length distribution of the training data.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from . import LABELS

MODEL_FILE, META_FILE, SKOPS_FILE = "model.joblib", "metadata.json", "model.skops"


def review_text(title: str | None, text: str | None) -> str:
    """The model input: title and text, joined as in the case-study notebooks.

    Training and serving must call the same function: a different preprocessing at serving time is one
    of the most common deployment bugs (training-serving skew).
    """
    return f"{title}. {text or ''}" if title else (text or "")


def build_pipeline(min_df: int = 2, C: float = 4.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df, max_features=200_000)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
    ])


def evaluate(pipe: Pipeline, texts: list[str], labels: list[str]) -> dict[str, float]:
    pred = pipe.predict(texts)
    per_class = f1_score(labels, pred, average=None, labels=list(LABELS), zero_division=0)
    return {"macro_f1": round(float(f1_score(labels, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(labels, pred)), 4),
            **{f"f1_{c}": round(float(v), 4) for c, v in zip(LABELS, per_class, strict=True)}}


def reference_stats(texts: list[str], labels: list[str]) -> dict[str, Any]:
    """What monitoring compares new data with (Session 16, block 3)."""
    lengths = np.log1p([len(t) for t in texts])
    shares = {c: round(sum(lab == c for lab in labels) / len(labels), 4) for c in LABELS}
    return {"label_shares": shares,
            "log_length_quantiles": np.quantile(lengths, np.linspace(0, 1, 11)).round(4).tolist(),
            "n": len(texts)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_model(pipe: Pipeline, out_dir: Path, metadata: dict[str, Any], fmt: str = "joblib") -> Path:
    """Write the pipeline and its metadata; returns the model file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "classes": [str(c) for c in pipe.classes_], **metadata}
    if fmt == "skops":
        import skops.io as sio  # optional dependency: uv sync --extra skops

        path = out_dir / SKOPS_FILE
        sio.dump(pipe, path)
    else:
        path = out_dir / MODEL_FILE
        joblib.dump(pipe, path)
    meta["model_file"], meta["model_sha256"] = path.name, file_sha256(path)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return path


def load_model(model_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load pipeline and metadata; refuses a file whose checksum does not match the metadata."""
    meta = json.loads((model_dir / META_FILE).read_text())
    path = model_dir / meta["model_file"]
    if file_sha256(path) != meta["model_sha256"]:
        raise ValueError(f"{path} does not match the checksum in {META_FILE}")
    if path.suffix == ".skops":
        import skops.io as sio

        pipe = sio.load(path, trusted=sio.get_untrusted_types(file=path))
    else:
        pipe = joblib.load(path)  # only load joblib/pickle files you created yourself
    if meta.get("scikit_learn") != sklearn.__version__:
        import warnings

        warnings.warn(f"model saved with scikit-learn {meta.get('scikit_learn')}, running {sklearn.__version__}",
                      stacklevel=2)
    return pipe, meta


# ---------------------------------------------------------------- a tiny model for tests and CI

FIXTURE_REVIEWS = [
    ("Broke after two days. Waste of money.", "neg"),
    ("Terrible quality, it stopped working and I want a refund.", "neg"),
    ("Awful smell and it leaked everywhere. Do not buy.", "neg"),
    ("Cheap plastic, broke the first time I used it.", "neg"),
    ("It is okay. Does the job but nothing special.", "neu"),
    ("Average product, some good points and some bad points.", "neu"),
    ("Not bad, not great. It is okay for the price.", "neu"),
    ("Works as described, okay quality, average.", "neu"),
    ("Excellent product, works great and I love it.", "pos"),
    ("Great value, my whole family loves it. Highly recommend.", "pos"),
    ("Love it! Works perfectly and arrived fast.", "pos"),
    ("Best purchase this year, great quality.", "pos"),
]


def train_fixture_model(out_dir: Path, version: str = "0.0.0-fixture") -> Path:
    """A model trained on twelve sentences: only for tests and container smoke tests, never for users."""
    texts, labels = zip(*FIXTURE_REVIEWS, strict=True)
    pipe = build_pipeline(min_df=1, C=10.0).fit(list(texts), list(labels))
    return save_model(pipe, out_dir, {"model_version": version, "training_data": "built-in fixture (12 sentences)",
                                      "validation": evaluate(pipe, list(texts), list(labels)),
                                      "reference": reference_stats(list(texts), list(labels))})
