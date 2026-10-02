"""Figures for the Session 9 theory pages.

Run from the repository root:
    uv run python sessions/09-feature-engineering-and-imbalance/theory/figures/make_figures.py

Deterministic (seeded). The third figure downloads the IBM Telco churn data from GitHub.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
TELCO = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
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
                arrowprops={"arrowstyle": "->", "color": MUTED}, color=INK)
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
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    telco = pd.read_csv(TELCO)
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce").fillna(0)
    y = (telco["Churn"] == "Yes").astype(int)
    X = telco.drop(columns=["customerID", "Churn"])
    cat_cols = X.select_dtypes("object").columns.tolist()
    prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                                   remainder=StandardScaler())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def lr(cw=None):
        return LogisticRegression(max_iter=2000, class_weight=cw)

    candidates = {
        "no correction": make_pipeline(prep, lr()),
        "undersampling": make_pipeline(prep, RandomUnderSampler(random_state=0), lr()),
        "oversampling": make_pipeline(prep, RandomOverSampler(random_state=0), lr()),
        "SMOTE": make_pipeline(prep, SMOTE(random_state=0), lr()),
        "class weights": make_pipeline(prep, lr("balanced")),
    }
    rows = []
    for name, model in candidates.items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        if name == "no correction":
            plain = proba
        pred = (proba >= 0.5).astype(int)
        rows.append((name, accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred), f1_score(y, pred)))
    pred = (plain >= 0.3).astype(int)
    rows.append(("no correction,\nthreshold 0.3", accuracy_score(y, pred), precision_score(y, pred),
                 recall_score(y, pred), f1_score(y, pred)))
    res = pd.DataFrame(rows, columns=["method", "accuracy", "precision", "recall", "f1"])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(res))
    width = 0.2
    series = [("accuracy", "accuracy", BLUE), ("precision", "precision (churn)", AQUA),
              ("recall", "recall (churn)", ORANGE), ("f1", "F1 (churn)", YELLOW)]
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * width, res[col], width - 0.02, color=color, label=label)
        for bar, v in zip(bars, res[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
    ax.set_xticks(x, res["method"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (5-fold cross-validation)")
    ax.set_title("Telco churn: corrections trade precision for recall, like a lower threshold",
                 fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "resampling_precision_recall.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cyclical_month_encoding()
    smote_illustration()
    resampling_precision_recall()
    print("figures written to", OUT)
