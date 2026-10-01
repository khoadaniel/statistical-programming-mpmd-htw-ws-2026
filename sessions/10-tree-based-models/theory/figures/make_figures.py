"""Figures for the Session 10 theory pages.

Run from the repository root:
    uv run python sessions/10-tree-based-models/theory/figures/make_figures.py

Deterministic (seeded). Reads the IBM Telco churn data from GitHub.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

OUT = Path(__file__).resolve().parent
URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
})


def telco():
    df = pd.read_csv(URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    cat = X.select_dtypes(exclude="number").columns.tolist()
    X_oh = pd.get_dummies(X, columns=cat, drop_first=True, dtype=int)
    return train_test_split(X_oh, y, test_size=0.25, stratify=y, random_state=0)


def tree_depth2(X_tr, y_tr):
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_tree(tree, feature_names=list(X_tr.columns), class_names=["stays", "churns"], filled=True,
              impurity=True, proportion=False, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Decision tree of depth 2 on the Telco churn training data", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "tree_depth2_churn.png", dpi=120)
    plt.close(fig)


def depth_vs_score(X_tr, y_tr):
    depths = range(1, 21)
    train_acc, cv_acc = [], []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=0)
        cv_acc.append(cross_val_score(tree, X_tr, y_tr, cv=5).mean())
        train_acc.append(tree.fit(X_tr, y_tr).score(X_tr, y_tr))
    best = int(np.argmax(cv_acc))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(depths, train_acc, color=BLUE, lw=2, marker="o", ms=5, label="training accuracy")
    ax.plot(depths, cv_acc, color=ORANGE, lw=2, marker="o", ms=5, label="5-fold CV accuracy")
    ax.axvline(depths[best], color=MUTED, ls=":", lw=1.2)
    ax.annotate(f"best CV depth = {depths[best]}", (depths[best], cv_acc[best]),
                xytext=(depths[best] + 2, cv_acc[best] - 0.05), color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(20, train_acc[-1] - 0.012, "training", color=BLUE, ha="right", va="top")
    ax.text(20, cv_acc[-1] + 0.008, "cross-validation", color=ORANGE, ha="right", va="bottom")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_title("Deeper trees fit the training data better and generalise worse", color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "tree_depth_vs_score.png", dpi=130)
    plt.close(fig)


def boosting_stages():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 6, 200)).reshape(-1, 1)
    target = np.sin(x[:, 0]) + rng.normal(0, 0.2, 200)
    pred = np.full(200, target.mean())
    snapshots = {}
    for step in range(1, 51):
        tree = DecisionTreeRegressor(max_depth=2).fit(x, target - pred)
        pred += 0.3 * tree.predict(x)
        if step in (1, 5, 20, 50):
            snapshots[step] = pred.copy()
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)
    for ax, (step, p) in zip(axes, snapshots.items()):
        ax.scatter(x[:, 0], target, s=10, color=GRID, edgecolor=MUTED, lw=0.3)
        ax.plot(x[:, 0], np.sin(x[:, 0]), color=AQUA, lw=1.5, ls="--", label="true function")
        ax.plot(x[:, 0], p, color=BLUE, lw=2, label="boosted model")
        mse = np.mean((target - p) ** 2)
        ax.set_title(f"{step} tree{'s' if step > 1 else ''}, training MSE {mse:.3f}", fontsize=10,
                     color=INK)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "boosting_stages.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = telco()
    tree_depth2(X_tr, y_tr)
    depth_vs_score(X_tr, y_tr)
    boosting_stages()
    print("figures written to", OUT)
