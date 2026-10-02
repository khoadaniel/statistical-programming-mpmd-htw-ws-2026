"""Figures for Session 13 (classical NLP) on the EBTI case study.

Run from the repository root:
    uv run python sessions/13-classical-nlp/theory/figures/make_figures.py

Writes dtm_sketch.png, top_ngrams.png and confusion_matrix.png next to this script.
Deterministic: fixed toy documents, time-based split (train 2017-2021, validate 2022-2023), random_state=0.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

OUT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[4] / "case-study" / "data"
COLORS = ["#2471a3", "#c0392b", "#1e8449"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def dtm_sketch() -> None:
    """A small document-term matrix: rows are descriptions in three languages, columns are words."""
    docs = [
        "toy car of plastic",
        "plastic toy, plastic box",
        "Spielzeugauto aus Kunststoff",
        "Spielzeug aus Holz",
        "voiture jouet en plastique",
    ]
    cv = CountVectorizer()
    X = cv.fit_transform(docs).toarray()
    words = cv.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=120)
    ax.imshow(X > 0, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ax.text(j, i, str(X[i, j]), ha="center", va="center",
                    color="black" if X[i, j] else "#b0b0b0", fontsize=10)
    ax.set_xticks(range(len(words)), words, rotation=40, ha="right")
    ax.set_yticks(range(len(docs)), [f"d{i + 1}: {d}" for i, d in enumerate(docs)])
    ax.set_xlabel("vocabulary (one column per word type)")
    ax.set_title(f"Document-term matrix: {X.shape[0]} documents x {X.shape[1]} words, "
                 f"{(X > 0).mean():.0%} of cells non-zero\n(d1, d3 and d5 describe the same toy)")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "dtm_sketch.png")
    plt.close(fig)


def fit_model():
    d = pd.read_parquet(DATA / "train_sample.parquet")
    year = d["start_date"].dt.year
    train, valid = d[year <= 2021], d[year >= 2022]
    model = make_pipeline(
        TfidfVectorizer(min_df=2, sublinear_tf=True),
        SGDClassifier(loss="hinge", alpha=1e-5, max_iter=20, tol=None, random_state=0, n_jobs=-1),
    )
    model.fit(train["description"], train["heading"])
    return model, valid


def top_ngrams(model, k: int = 12) -> None:
    vec, clf = model.named_steps.values()
    names = vec.get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), dpi=120)
    for ax, heading, color in zip(axes, ["6403", "6404", "9503"], COLORS):
        coef = clf.coef_[list(clf.classes_).index(heading)]
        top = np.argsort(coef)[::-1][:k][::-1]
        ax.barh(range(k), coef[top], color=color)
        ax.set_yticks(range(k), names[top])
        short = {"6403": "footwear, uppers of leather", "6404": "footwear, uppers of textile",
                 "9503": "toys"}[heading]
        ax.set_title(f"{heading}: {short}", fontsize=10)
        ax.set_xlabel("coefficient")
    fig.suptitle("Most informative words per heading (word TF-IDF + linear SVM, 2017-2021 sample)")
    fig.tight_layout()
    fig.savefig(OUT / "top_ngrams.png")
    plt.close(fig)


def confusion(model, valid) -> None:
    shoes = ["6401", "6402", "6403", "6404", "6405", "6406"]
    keep = valid["heading"].isin(shoes).to_numpy()
    pred = model.predict(valid["description"])[keep]
    labels = shoes + ["other"]
    pred = np.where(np.isin(pred, shoes), pred, "other")
    cm = confusion_matrix(valid["heading"][keep], pred, labels=labels)[: len(shoes)]
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=120)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{share[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(shoes)), shoes)
    ax.set_xlabel("predicted heading")
    ax.set_ylabel("true heading")
    ax.set_title("Footwear (chapter 64), validation years 2022-2023\n(row shares = recall; counts in brackets)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    dtm_sketch()
    model, valid = fit_model()
    top_ngrams(model)
    confusion(model, valid)
    print("figures written to", OUT)
