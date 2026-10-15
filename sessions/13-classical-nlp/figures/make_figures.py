"""Figures for Session 13 (classical NLP) on the EBTI case study.

Run from the repository root:
    uv run python sessions/13-classical-nlp/figures/make_figures.py

Writes dtm_sketch.png next to this script (deterministic: fixed toy documents). The other figures of
the session (most informative words, the footwear confusion matrix) are drawn inside the notebooks.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

OUT = Path(__file__).resolve().parent
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


if __name__ == "__main__":
    dtm_sketch()
    print("figures written to", OUT)
