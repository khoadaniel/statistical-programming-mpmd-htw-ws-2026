"""Figures for Session 13 (classical NLP).

Run from the repository root:
    uv run --with pandas --with pyarrow --with scikit-learn --with matplotlib \
        python sessions/13-classical-nlp/theory/figures/make_figures.py

Writes dtm_sketch.png, top_ngrams.png and confusion_matrix.png next to this script.
Deterministic: fixed split (random_state=0) and fixed toy documents.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

OUT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[4] / "case-study" / "data" / "train_sample.parquet"
LABELS = ["neg", "neu", "pos"]
COLORS = {"neg": "#c0392b", "neu": "#7f8c8d", "pos": "#2471a3"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def dtm_sketch() -> None:
    """A small document-term matrix: rows are reviews, columns are vocabulary words."""
    docs = [
        "great pump, works great",
        "pump stopped working",
        "not great, not bad",
        "smells great",
        "bad smell, stopped using it",
    ]
    cv = CountVectorizer()
    X = cv.fit_transform(docs).toarray()
    words = cv.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=130)
    ax.imshow(X > 0, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ax.text(j, i, str(X[i, j]), ha="center", va="center",
                    color="black" if X[i, j] else "#b0b0b0", fontsize=11)
    ax.set_xticks(range(len(words)), words, rotation=40, ha="right")
    ax.set_yticks(range(len(docs)), [f"d{i + 1}: {d}" for i, d in enumerate(docs)])
    ax.set_xlabel("vocabulary (one column per word type)")
    ax.set_title(f"Document-term matrix: {X.shape[0]} documents x {X.shape[1]} words, "
                 f"{(X > 0).mean():.0%} of cells non-zero")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "dtm_sketch.png")
    plt.close(fig)


def fit_model():
    reviews = pd.read_parquet(DATA)
    texts = reviews["title"] + " " + reviews["text"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        texts, reviews["label"], test_size=0.2, stratify=reviews["label"], random_state=0
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
        LogisticRegression(C=4, class_weight="balanced", max_iter=2000),
    )
    model.fit(X_tr, y_tr)
    return model, X_va, y_va


def top_ngrams(model, k: int = 12) -> None:
    vec, clf = model.named_steps.values()
    names = vec.get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4), dpi=130)
    for ax, label in zip(axes, clf.classes_):
        coef = clf.coef_[list(clf.classes_).index(label)]
        top = np.argsort(coef)[::-1][:k][::-1]
        ax.barh(range(k), coef[top], color=COLORS[label])
        ax.set_yticks(range(k), names[top])
        ax.set_title(f"class '{label}'")
        ax.set_xlabel("coefficient")
    fig.suptitle("Most informative n-grams per class (TF-IDF 1-2-grams + logistic regression)")
    fig.tight_layout()
    fig.savefig(OUT / "top_ngrams.png")
    plt.close(fig)


def confusion(model, X_va, y_va) -> None:
    pred = model.predict(X_va)
    cm = confusion_matrix(y_va, pred, labels=LABELS)
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{share[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title("Confusion matrix, validation set\n(row shares = recall; counts in brackets)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    dtm_sketch()
    model, X_va, y_va = fit_model()
    top_ngrams(model)
    confusion(model, X_va, y_va)
    print("figures written to", OUT)
