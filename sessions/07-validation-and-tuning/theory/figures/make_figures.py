"""Make the figures of the Session 7 theory pages.

Run from the repository root:
    uv run --with pandas --with pyarrow --with scikit-learn --with matplotlib \
        python sessions/07-validation-and-tuning/theory/figures/make_figures.py
Deterministic: fixed seeds. Uses the Inside Airbnb Berlin listings
(case-study/data/airbnb/, created with `uv run python case-study/prepare_airbnb.py`).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, learning_curve, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

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


def price_data():
    """Berlin Airbnb price model of theory page 02: 6,675 short-term listings, 480 columns, host groups."""
    root = OUT.parents[3]
    lst = pd.read_parquet(root / "case-study/data/airbnb/listings.parquet")
    bnb = lst[(lst["minimum_nights"] < 28) & lst["price"].between(10, 1000)].reset_index(drop=True)
    bnb["dist_km"] = np.hypot((bnb["latitude"] - 52.5219) * 111.2, (bnb["longitude"] - 13.4132) * 68.0)
    mlb = MultiLabelBinarizer()
    amen = pd.DataFrame(mlb.fit_transform(bnb["amenities"].map(json.loads)), columns=["am_" + a for a in mlb.classes_])
    amen = amen.loc[:, amen.sum() >= 20]
    num = ["accommodates", "bedrooms", "beds", "bathrooms", "dist_km", "minimum_nights",
           "availability_365", "number_of_reviews", "review_scores_rating"]
    cat = ["room_type", "district", "neighbourhood", "property_type"]
    X = pd.concat([bnb[num + cat], amen], axis=1)
    prep = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        (OneHotEncoder(handle_unknown="ignore"), cat),
        (StandardScaler(), list(amen.columns)))
    return X, np.log(bnb["price"]), bnb["host_id"], prep


def validation_curve_fig(X, y, hosts, prep):
    """Ridge penalty against R², with all listings and with a random subset of 1,000."""
    alphas = np.logspace(-2, 4, 19)
    small = np.random.default_rng(0).choice(len(X), 1000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, rows) in zip(axes, [("All 6,675 listings", np.arange(len(X))), ("1,000 listings", small)]):
        tr, va = validation_curve(make_pipeline(prep, Ridge()), X.iloc[rows], y.iloc[rows], groups=hosts.iloc[rows],
                                  param_name="ridge__alpha", param_range=alphas, cv=GroupKFold(5), scoring="r2")
        ax.plot(alphas, tr.mean(1), color=BLUE, lw=2, label="training score")
        ax.plot(alphas, va.mean(1), color=ORANGE, lw=2, label="validation score (5 host-grouped folds)")
        ax.fill_between(alphas, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=ORANGE, alpha=0.15, lw=0)
        best = alphas[va.mean(1).argmax()]
        ax.axvline(best, color=MUTED, lw=1, ls="--")
        ax.text(best * 1.2, 0.82, f"best alpha ≈ {best:.0f}", color=MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("ridge penalty alpha (log scale; larger = simpler model)")
        ax.set_title(title, color=INK, loc="left")
    axes[0].set_ylabel("R² (log price)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Validation curves: ridge regression of Berlin Airbnb prices on 480 columns",
                 x=0.07, y=1.02, ha="left", color=INK)
    save(fig, "validation_curve.png")


def learning_curve_fig(X, y, hosts, prep):
    """Learning curves of the price model: ridge and gradient boosting."""
    models = [("Ridge regression (alpha = 10)", make_pipeline(prep, Ridge(alpha=10))),
              ("Gradient boosting (defaults)", make_pipeline(prep, HistGradientBoostingRegressor(random_state=0)))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, est) in zip(axes, models):
        sizes, tr, va = learning_curve(est, X, y, groups=hosts, train_sizes=np.linspace(0.1, 1, 8),
                                       cv=GroupKFold(5), scoring="r2")
        ax.plot(sizes, tr.mean(1), color=BLUE, lw=2, marker="o", ms=4, label="training score")
        ax.plot(sizes, va.mean(1), color=ORANGE, lw=2, marker="o", ms=4, label="validation score")
        ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=ORANGE, alpha=0.15, lw=0)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("number of training listings")
        ax.set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("R² (log price)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Learning curves of the Berlin price model (host-grouped 5-fold CV)", x=0.07, y=1.03,
                 ha="left", color=INK)
    save(fig, "learning_curve.png")


if __name__ == "__main__":
    cv_splits()
    data = price_data()
    validation_curve_fig(*data)
    learning_curve_fig(*data)
