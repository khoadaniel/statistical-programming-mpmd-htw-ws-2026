"""Figures for the Session 9 theory pages.

Run from the repository root:
    uv run python sessions/09-feature-engineering-and-imbalance/theory/figures/make_figures.py

Deterministic (seeded). The third figure needs the case-study sample
(case-study/data/train_sample.parquet): footwear decisions of chapter 64.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def cyclical_month_encoding():
    months = np.arange(1, 13)
    angle = 2 * np.pi * months / 12
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRID, lw=2, zorder=1)
    ax.scatter(np.cos(angle), np.sin(angle), s=90, color=BLUE, edgecolor="white", lw=2, zorder=3)
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m, a, name in zip(months, angle, names):
        ax.annotate(name, (np.cos(a), np.sin(a)), xytext=(1.18 * np.cos(a), 1.18 * np.sin(a)),
                    ha="center", va="center", color=INK)
    for m in (12, 1):
        a = 2 * np.pi * m / 12
        ax.scatter([np.cos(a)], [np.sin(a)], s=130, color=ORANGE, edgecolor="white", lw=2, zorder=4)
    ax.set_xlabel("month_cos = cos(2π · month / 12)")
    ax.set_ylabel("month_sin = sin(2π · month / 12)")
    ax.set_title("Cyclical encoding: December and January are neighbours", fontsize=11, color=INK)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "cyclical_month_encoding.png", dpi=130)
    plt.close(fig)


def smote_illustration():
    rng = np.random.default_rng(3)
    majority = rng.normal([0, 0], [1.3, 1.3], size=(160, 2))
    minority = rng.normal([4.2, 3.8], [1.0, 1.0], size=(14, 2))
    i = int(np.argmin(np.linalg.norm(minority - minority.mean(axis=0), axis=1)))
    d = np.linalg.norm(minority - minority[i], axis=1)
    nn = np.argsort(d)[1:6]
    synth = []
    for j in nn:
        lam = rng.uniform()
        synth.append(minority[i] + lam * (minority[j] - minority[i]))
    synth = np.array(synth)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(*majority.T, s=22, color=GRID, edgecolor=MUTED, lw=0.4, label="majority class")
    ax.scatter(*minority.T, s=60, color=BLUE, edgecolor="white", lw=1.5, label="minority class", zorder=3)
    for j in nn:
        ax.plot(*np.c_[minority[i], minority[j]], color=BLUE, lw=1.2, ls="--", zorder=2)
    ax.scatter(*minority[i], s=150, color=BLUE, edgecolor=INK, lw=1.5, zorder=4)
    ax.annotate("x: chosen minority row", minority[i], xytext=(minority[i][0] - 4.5, minority[i][1] + 1.8),
                arrowprops=dict(arrowstyle="->", color=MUTED), color=INK)
    ax.scatter(*synth.T, s=90, marker="D", color=ORANGE, edgecolor="white", lw=1.5, zorder=5,
               label=r"synthetic rows $x + \lambda\,(x_{nn} - x)$")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("SMOTE: new rows on the lines to the 5 nearest minority neighbours", fontsize=11,
                 color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "smote_illustration.png", dpi=130)
    plt.close(fig)


def resampling_precision_recall():
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import make_pipeline
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    root = OUT.parents[3]
    decisions = pd.read_parquet(root / "case-study/data/train_sample.parquet")
    shoes = decisions[decisions["chapter"] == "64"].reset_index(drop=True)   # footwear task, 6 headings
    y = shoes["heading"]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    X = pd.DataFrame(TruncatedSVD(50, random_state=0).fit_transform(tfidf.fit_transform(shoes["description"])),
                     columns=[f"svd{i}" for i in range(50)])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def lr(cw=None):
        return LogisticRegression(max_iter=2000, class_weight=cw)

    candidates = {
        "no correction": make_pipeline(StandardScaler(), lr()),
        "undersampling": make_pipeline(StandardScaler(), RandomUnderSampler(random_state=0), lr()),
        "oversampling": make_pipeline(StandardScaler(), RandomOverSampler(random_state=0), lr()),
        "SMOTE": make_pipeline(StandardScaler(), SMOTE(random_state=0), lr()),
        "class weights": make_pipeline(StandardScaler(), lr("balanced")),
    }
    labels = sorted(y.unique())
    rows = []
    for name, model in candidates.items():
        pred = cross_val_predict(model, X, y, cv=cv)
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        rows.append((name, accuracy_score(y, pred), f.mean(), p[labels.index("6406")], r[labels.index("6406")]))
    res = pd.DataFrame(rows, columns=["method", "accuracy", "macro_f1", "precision", "recall"])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(res))
    width = 0.2
    series = [("accuracy", "accuracy (6 headings)", BLUE), ("macro_f1", "macro-F1 (6 headings)", YELLOW),
              ("precision", "precision of 6406", AQUA), ("recall", "recall of 6406", ORANGE)]
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * width, res[col], width - 0.02, color=color, label=label)
        for b, v in zip(bars, res[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
    ax.set_xticks(x, res["method"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (5-fold cross-validation)")
    ax.set_title("Footwear headings: corrections raise recall of the rare heading 6406 but lower macro-F1",
                 fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "resampling_precision_recall.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cyclical_month_encoding()
    smote_illustration()
    resampling_precision_recall()
    print("figures written to", OUT)
