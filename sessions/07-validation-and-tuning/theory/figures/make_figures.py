"""Make the figures of the Session 7 theory pages.

Run from the repository root:
    uv run --with pandas --with pyarrow --with scikit-learn --with matplotlib \
        python sessions/07-validation-and-tuning/theory/figures/make_figures.py
Deterministic: fixed seeds, built-in data (diabetes) and the IBM Telco churn sample (downloaded).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TELCO = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def cv_splits():
    """k-fold (left) and time-series split (right) as coloured index bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    k = 5
    for ax, kind in zip(axes, ["k-fold (k = 5)", "Time-series split (5 folds)"]):
        for i in range(k):
            yy = k - 1 - i
            if kind.startswith("k-fold"):
                for j in range(k):
                    c = ORANGE if j == i else BLUE
                    ax.add_patch(Rectangle((j / k + 0.003, yy + 0.1), 1 / k - 0.006, 0.8, color=c))
            else:
                n = k + 1
                for j in range(n):
                    if j <= i:
                        c = BLUE
                    elif j == i + 1:
                        c = ORANGE
                    else:
                        c = GRID
                    ax.add_patch(Rectangle((j / n + 0.003, yy + 0.1), 1 / n - 0.006, 0.8, color=c))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, k)
        ax.set_yticks([k - 0.5 - i for i in range(k)], [f"split {i + 1}" for i in range(k)])
        ax.set_xticks([])
        ax.grid(False)
        ax.set_title(kind, color=INK, loc="left")
        ax.set_xlabel("rows (left to right: in time order for the time-series split)", color=MUTED)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRID)]
    fig.legend(handles, ["training", "validation", "not used"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, "cv_splits.png")


def validation_curve_fig():
    """Ridge on degree-2 polynomial features of the diabetes data: R² against the penalty alpha."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    alphas = np.logspace(-3, 4, 22)
    pipe = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge())
    tr, va = validation_curve(pipe, X, y, param_name="ridge__alpha", param_range=alphas,
                              cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(alphas, tr.mean(1), color=BLUE, lw=2, label="training score")
    ax.plot(alphas, va.mean(1), color=ORANGE, lw=2, label="validation score (5-fold CV)")
    ax.fill_between(alphas, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=ORANGE, alpha=0.15, lw=0)
    best = alphas[va.mean(1).argmax()]
    ax.axvline(best, color=MUTED, lw=1, ls="--")
    ax.text(best * 1.15, 0.05, f"best alpha ≈ {best:.0f}", color=MUTED)
    ax.text(2e-3, 0.69, "overfitting: large gap", color=MUTED, va="top")
    ax.text(1.2e3, 0.69, "underfitting: both low", color=MUTED, va="top")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("ridge penalty alpha (log scale; larger = simpler model)")
    ax.set_ylabel("R²")
    ax.set_title("Validation curve: ridge regression, diabetes data", color=INK, loc="left")
    ax.legend(loc="lower left")
    save(fig, "validation_curve.png")


def learning_curve_fig():
    """Learning curves on Telco churn: a flexible model (5-NN) and a simple one (logistic regression)."""
    df = pd.read_csv(TELCO)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]]
    models = [("5-nearest neighbours (flexible)", KNeighborsClassifier(5)),
              ("Logistic regression (simple)", LogisticRegression())]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, est) in zip(axes, models):
        sizes, tr, va = learning_curve(make_pipeline(StandardScaler(), est), X, y,
                                       train_sizes=np.linspace(0.05, 1, 10),
                                       cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
        ax.plot(sizes, tr.mean(1), color=BLUE, lw=2, marker="o", ms=4, label="training score")
        ax.plot(sizes, va.mean(1), color=ORANGE, lw=2, marker="o", ms=4, label="validation score")
        ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=ORANGE, alpha=0.15, lw=0)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("number of training customers")
        ax.set_ylim(0.65, 1.0)
    axes[0].set_ylabel("ROC AUC")
    axes[0].legend(loc="upper right")
    fig.suptitle("Learning curves on the Telco churn data (4 numeric features)", x=0.07, y=1.03, ha="left", color=INK)
    save(fig, "learning_curve.png")


if __name__ == "__main__":
    cv_splits()
    validation_curve_fig()
    learning_curve_fig()
