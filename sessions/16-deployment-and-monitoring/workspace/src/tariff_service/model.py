"""The model artefact: a fitted scikit-learn pipeline plus a metadata file.

    models/
      model.joblib     the fitted pipeline (TF-IDF vocabulary, idf weights, coefficients)
      metadata.json    version, library versions, training data, validation scores, reference statistics,
                       and the English texts of the headings the model knows

The metadata make the file reproducible and give monitoring its reference: the chapter, language and
country shares and the description-length distribution of the training data.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

MODEL_FILE, META_FILE, SKOPS_FILE = "model.joblib", "metadata.json", "model.skops"


def decision_text(description: str | None) -> str:
    """The model input: the description of goods with normalised white space.

    Training, the API, the dashboard and the monitor call the same function: a different preprocessing at
    serving time is one of the most common deployment bugs (training-serving skew).
    """
    return " ".join((description or "").split())


def build_pipeline(min_df: int = 2, alpha: float = 1e-5) -> Pipeline:
    """Word TF-IDF + a linear support vector machine fitted by stochastic gradient descent (Session 13).

    The hinge loss gives no probabilities; the top-3 list reports the decision scores (margins) instead.
    (loss="modified_huber" offers predict_proba, but on the case study its values were mostly 1.0 and 0.0
    and its accuracy one point lower.)"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=min_df, max_features=400_000)),
        ("clf", SGDClassifier(loss="hinge", alpha=alpha, max_iter=20, tol=None, random_state=0, n_jobs=-1)),
    ])


def prune_coefficients(pipe: Pipeline, threshold: float = 0.05) -> float:
    """Set coefficients with |w| < threshold to zero and store them as a sparse matrix.

    With about 900 headings and 80,000 words the dense coefficient matrix takes about 600 MB; on the case
    study, pruning at 0.05 keeps about 1 % of the weights, the same validation accuracy and a model file of
    about 8 MB. Returns the share of weights kept."""
    clf = pipe.named_steps["clf"]
    coef = np.asarray(clf.coef_.todense()) if hasattr(clf.coef_, "todense") else clf.coef_.copy()
    coef[np.abs(coef) < threshold] = 0.0
    clf.coef_ = coef
    clf.sparsify()
    return float(clf.coef_.nnz / np.prod(clf.coef_.shape))


def top_k(pipe: Pipeline, texts: list[str], k: int = 3) -> list[list[tuple[str, float]]]:
    """The k highest-scoring headings with their decision scores, for every text.

    A score is the margin of the linear SVM: higher is better, above 0 the model votes for the heading, below
    0 it is not convinced of any heading. It is not a probability (Session 8, calibration)."""
    scores = pipe.decision_function(texts)
    order = np.argsort(-scores, axis=1)[:, :k]
    return [[(str(pipe.classes_[j]), round(float(row[j]), 4)) for j in idx]
            for row, idx in zip(scores, order, strict=True)]


def evaluate(pipe: Pipeline, texts: list[str], labels: list[str]) -> dict[str, float]:
    pred = pipe.predict(texts)
    ranked = top_k(pipe, texts, k=3)
    in_top3 = [y in [h for h, _ in r] for y, r in zip(labels, ranked, strict=True)]
    return {"accuracy": round(float(accuracy_score(labels, pred)), 4),
            "macro_f1": round(float(f1_score(labels, pred, average="macro", zero_division=0)), 4),
            "top3_accuracy": round(float(np.mean(in_top3)), 4),
            "chapter_accuracy": round(float(np.mean([p[:2] == y[:2] for p, y in zip(pred, labels, strict=True)])), 4)}


def shares(values: list[str], top: int | None = None) -> dict[str, float]:
    counts = Counter(values)
    n = sum(counts.values())
    items = counts.most_common(top)
    return {k: round(v / n, 4) for k, v in items}


def reference_stats(texts: list[str], labels: list[str], languages: list[str] | None = None,
                    countries: list[str] | None = None) -> dict[str, Any]:
    """What monitoring compares new data with (Session 16, block 3)."""
    lengths = np.log1p([len(t) for t in texts])
    ref = {"chapter_shares": shares([y[:2] for y in labels]),
           "log_length_quantiles": np.quantile(lengths, np.linspace(0, 1, 11)).round(4).tolist(),
           "n": len(texts)}
    if languages is not None:
        ref["language_shares"] = shares(languages)
    if countries is not None:
        ref["country_shares"] = shares(countries)
    return ref


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
        joblib.dump(pipe, path, compress=3)
    meta["model_file"], meta["model_sha256"] = path.name, file_sha256(path)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
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

FIXTURE_DECISIONS = [
    ("Damenstiefel mit Oberteil aus Rindleder und Laufsohle aus Gummi", "6403"),
    ("Chaussures pour hommes, dessus en cuir naturel, semelle en caoutchouc", "6403"),
    ("Ladies' shoes with leather uppers and rubber soles", "6403"),
    ("Herrenschuhe, Oberteil aus Leder, Laufsohle aus Kunststoff", "6403"),
    ("Sportschuhe mit Oberteil aus Spinnstoff und Laufsohle aus Kunststoff", "6404"),
    ("Chaussures de sport, dessus en matière textile, semelle en caoutchouc", "6404"),
    ("Trainers with textile uppers and plastic soles", "6404"),
    ("Hausschuhe mit Oberteil aus Textil und Laufsohle aus Gummi", "6404"),
    ("Spielzeugauto aus Kunststoff für Kinder", "9503"),
    ("Voiture jouet en matière plastique pour enfants", "9503"),
    ("Plastic toy car for children with wheels", "9503"),
    ("Puppe aus Kunststoff, Spielzeug für Kinder", "9503"),
    ("Mund-Nasen-Schutzmaske aus Vliesstoff zum Einmalgebrauch", "6307"),
    ("Masque de protection en non-tissé, à usage unique", "6307"),
    ("Disposable face mask made of nonwoven fabric", "6307"),
    ("Reinigungstuch aus Vliesstoff, konfektioniert", "6307"),
]
FIXTURE_HEADINGS = {
    "6307": "Textiles; made up articles (including dress patterns), n.e.c. in chapter 63",
    "6403": "Footwear; with outer soles of rubber, plastics, leather or composition leather and uppers of leather",
    "6404": "Footwear; with outer soles of rubber, plastics, leather or composition leather and uppers of textile",
    "9503": "Tricycles, scooters, pedal cars and similar wheeled toys; dolls; other toys",
}


def train_fixture_model(out_dir: Path, version: str = "0.0.0-fixture") -> Path:
    """A model trained on 16 sentences: only for tests and container smoke tests, never for users."""
    texts, labels = zip(*FIXTURE_DECISIONS, strict=True)
    texts = [decision_text(t) for t in texts]
    pipe = build_pipeline(min_df=1, alpha=1e-4).fit(texts, list(labels))
    return save_model(pipe, out_dir, {"model_version": version, "training_data": "built-in fixture (16 sentences)",
                                      "validation": evaluate(pipe, texts, list(labels)),
                                      "reference": reference_stats(texts, list(labels)),
                                      "headings": FIXTURE_HEADINGS})

