"""Figures for the Session 10 theory pages.

Run from the repository root:
    uv run python sessions/10-tree-based-models/theory/figures/make_figures.py

Deterministic (seeded). Reads the IBM Telco churn data from GitHub and the Inside Airbnb Berlin listings
(case-study/data/airbnb/, `uv run python case-study/prepare_airbnb.py`). LightGBM needs libomp on macOS
(`brew install libomp`).
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
                arrowprops={"arrowstyle": "->", "color": MUTED})
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


def price_model_interpretation():
    """Permutation importance and SHAP dependence of distance for the LightGBM price model (page 4)."""
    import json
    import re

    import shap
    from lightgbm import LGBMRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupShuffleSplit

    lst = pd.read_parquet(OUT.parents[3] / "case-study/data/airbnb/listings.parquet")
    bnb = lst[(lst["minimum_nights"] < 28) & lst["price"].between(10, 1000)].reset_index(drop=True)
    bnb["dist_km"] = np.hypot((bnb["latitude"] - 52.5219) * 111.2, (bnb["longitude"] - 13.4132) * 68.0)
    amen_lists = bnb["amenities"].map(json.loads)
    common = amen_lists.explode().value_counts().loc[lambda c: c >= 200].index
    amen = pd.DataFrame({"am_" + re.sub(r"\W+", "_", a).strip("_").lower(): amen_lists.map(lambda x, a=a: a in x).astype(int)
                         for a in common})
    rich_num = ["accommodates", "bedrooms", "beds", "bathrooms", "dist_km", "latitude", "longitude",
                "minimum_nights", "availability_365", "number_of_reviews", "review_scores_rating"]
    X = pd.concat([bnb[rich_num], bnb[["room_type", "district", "property_type"]].astype("category"),
                   amen.assign(n_amenities=amen_lists.map(len))], axis=1)
    y = np.log(bnb["price"])
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, groups=bnb["host_id"]))
    model = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, verbose=-1, random_state=0).fit(X.iloc[tr], y.iloc[tr])
    perm = permutation_importance(model, X.iloc[te], y.iloc[te], scoring="r2", n_repeats=5, random_state=0)
    imp = pd.Series(perm.importances_mean, index=X.columns).nlargest(10)[::-1]
    sv = shap.TreeExplainer(model)(X.iloc[te])
    j = list(X.columns).index("dist_km")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].barh(imp.index, imp.values, color=BLUE)
    axes[0].set_xlabel("drop in R² when shuffled (held-out hosts)")
    axes[0].set_title("Permutation importance", color=INK, loc="left")
    axes[0].grid(axis="x", color=GRID, lw=0.8)
    factor = np.exp(sv.values[:, j]) - 1
    axes[1].scatter(X.iloc[te]["dist_km"], 100 * factor, s=8, alpha=0.5, color=ORANGE, edgecolor="none")
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xlabel("distance to Alexanderplatz (km)")
    axes[1].set_ylabel("SHAP contribution as price change (%)")
    axes[1].set_title("SHAP: what the model adds for location", color=INK, loc="left")
    axes[1].grid(color=GRID, lw=0.8)
    fig.suptitle("LightGBM price model for Berlin Airbnb listings", x=0.06, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "price_model_interpretation.png", dpi=110)
    plt.close(fig)
    near = X.iloc[te]["dist_km"] < 3
    print("dist SHAP factor: share > 0 within 3 km", round(float((factor[near] > 0).mean()), 3),
          "| 95th pct within 3 km", round(float(np.percentile(factor[near], 95)), 3),
          "| max", round(float(factor.max()), 3), "| median beyond 8 km", round(float(np.median(factor[~(X.iloc[te]["dist_km"] < 8)])), 3))


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = telco()
    tree_depth2(X_tr, y_tr)
    depth_vs_score(X_tr, y_tr)
    boosting_stages()
    price_model_interpretation()
    print("figures written to", OUT)
