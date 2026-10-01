"""Figures for the Session 9 theory pages.

Run from the repository root:
    uv run python sessions/09-feature-engineering-and-imbalance/theory/figures/make_figures.py

Deterministic (seeded). The third figure needs the case-study sample
(case-study/data/train_sample.parquet).
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    root = OUT.parents[3]
    reviews = pd.read_parquet(root / "case-study/data/train_sample.parquet")
    text = reviews["text"].fillna("")
    neg = r"\b(?:not|no|never|don't|doesn't|didn't|isn't|wasn't|won't|can't)\b"
    X = pd.DataFrame({
        "log_chars": np.log1p(text.str.len()), "n_exclaim": text.str.count("!"),
        "n_question": text.str.count(r"\?"), "n_negations": text.str.lower().str.count(neg),
        "upper_share": text.str.count(r"[A-Z]") / text.str.len().clip(lower=1),
        "title_words": reviews["title"].fillna("").str.split().str.len(),
        "verified": reviews["verified_purchase"].astype(int),
        "log_helpful": np.log1p(reviews["helpful_vote"]), "n_images": reviews["n_images"],
    })
    y = reviews["label"]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def lr(cw=None):
        return LogisticRegression(max_iter=1000, class_weight=cw)

    candidates = {
        "no correction": make_pipeline(StandardScaler(), lr()),
        "undersampling": make_pipeline(StandardScaler(), RandomUnderSampler(random_state=0), lr()),
        "oversampling": make_pipeline(StandardScaler(), RandomOverSampler(random_state=0), lr()),
        "SMOTE": make_pipeline(StandardScaler(), SMOTE(random_state=0), lr()),
        "class weights": make_pipeline(StandardScaler(), lr("balanced")),
    }
    rows = []
    for name, model in candidates.items():
        pred = cross_val_predict(model, X, y, cv=cv)
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=["neg", "neu", "pos"],
                                                     zero_division=0)
        rows.append((name, p[1], r[1], f[1], f.mean()))
    res = pd.DataFrame(rows, columns=["method", "precision", "recall", "f1", "macro_f1"])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(res))
    width = 0.2
    series = [("precision", "neutral precision", BLUE), ("recall", "neutral recall", ORANGE),
              ("f1", "neutral F1", AQUA), ("macro_f1", "macro-F1 (3 classes)", YELLOW)]
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * width, res[col], width - 0.02, color=color, label=label)
        for b, v in zip(bars, res[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    ax.set_xticks(x, res["method"])
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("score (5-fold cross-validation)")
    ax.set_title("Imbalance corrections trade neutral precision for recall (logistic regression)",
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
