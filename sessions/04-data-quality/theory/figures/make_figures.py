"""Figures for the Session 4 theory pages (deterministic).

Run from the repository root:
    uv run python sessions/04-data-quality/theory/figures/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "case-study" / "data"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(HERE / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def missingness_pattern():
    p = pd.read_parquet(DATA / "products.parquet")
    miss = pd.DataFrame({
        "price": p["price"].isna(),
        "features (empty)": p["features"].eq(""),
        "description (empty)": p["description"].eq(""),
        "train_avg_rating": p["train_avg_rating"].isna(),
        "store": p["store"].isna(),
        "details": p["details"].isna(),
    })
    n_reviews = p["train_n_reviews"].fillna(0)
    sample = miss.assign(n=n_reviews).sample(400, random_state=0).sort_values("n", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained", width_ratios=[1.2, 1])
    ax = axes[0]
    ax.imshow(sample.drop(columns="n").to_numpy().T, aspect="auto", interpolation="nearest",
              cmap=plt.matplotlib.colors.ListedColormap(["#f1f0ec", BLUE]))
    ax.set_yticks(range(miss.shape[1]), [f"{c}  {miss[c].mean():.0%}" for c in miss.columns])
    ax.set_xticks([0, len(sample) - 1], ["most reviews", "fewest"])
    ax.set_title("Missing (blue) in 400 random products, sorted by number of reviews", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    bins = pd.cut(n_reviews, [-1, 0, 1, 5, 20, 100, np.inf], labels=["0", "1", "2-5", "6-20", "21-100", ">100"])
    share = miss["price"].groupby(bins, observed=True).mean()
    ax = axes[1]
    ax.bar(share.index.astype(str), share.values, color=BLUE, width=0.65)
    ax.axhline(miss["price"].mean(), color=MUTED, ls="--", lw=1)
    ax.text(5.4, miss["price"].mean() + 0.02, f"all: {miss['price'].mean():.0%}", ha="right", color=MUTED, fontsize=9)
    for x, v in enumerate(share.values):
        ax.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set(ylim=(0, 1), xlabel="training reviews per product", ylabel="share with missing price",
           title="Price missing less often for popular products")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "missingness-pattern.png")


def fences(x):
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return {
        "IQR rule": (q1 - 1.5 * iqr, q3 + 1.5 * iqr),
        "z-score |z| > 3": (x.mean() - 3 * x.std(), x.mean() + 3 * x.std()),
        "MAD rule |z*| > 3.5": (med - 3.5 * mad / 0.6745, med + 3.5 * mad / 0.6745),
    }


def univariate_outliers():
    r = pd.read_parquet(DATA / "train_sample.parquet", columns=["text"])
    length = r["text"].str.len().to_numpy()
    length = length[length > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), layout="constrained")
    styles = {"IQR rule": (ORANGE, "-"), "z-score |z| > 3": (AQUA, "--"), "MAD rule |z*| > 3.5": (INK, ":")}
    # raw scale
    ax = axes[0]
    ax.hist(length[length < 3000], bins=80, color="#a9c7ee")
    for name, (lo, hi) in fences(length).items():
        share = np.mean((length < lo) | (length > hi))
        ax.axvline(hi, color=styles[name][0], ls=styles[name][1], lw=1.8, label=f"{name}: {share:.1%} flagged")
    ax.set(title="Review length, raw scale (cut at 3,000)", xlabel="characters", ylabel="reviews")
    ax.legend(fontsize=8.5)
    # log scale
    ax = axes[1]
    loglen = np.log10(length)
    ax.hist(loglen, bins=80, color="#a9c7ee")
    for name, (lo, hi) in fences(loglen).items():
        share = np.mean((loglen < lo) | (loglen > hi))
        for v in (lo, hi):
            ax.axvline(v, color=styles[name][0], ls=styles[name][1], lw=1.8,
                       label=f"{name}: {share:.1%} flagged" if v == lo else None)
    ax.set_xticks([0, 1, 2, 3, 4], ["1", "10", "100", "1,000", "10,000"])
    ax.set(title="Review length, log scale: rules flag both tails", xlabel="characters (log scale)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True, facecolor="white", edgecolor="white", framealpha=1)
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    save(fig, "univariate-outliers.png")


def box_cox():
    r = pd.read_parquet(DATA / "train_sample.parquet", columns=["text"])
    length = r["text"].str.len()
    length = length[length > 0].to_numpy()
    transformed, lam = stats.boxcox(length)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    for col, (x, title) in enumerate([(length, "before: review length"),
                                      (transformed, f"after: Box–Cox, λ = {lam:.2f}")]):
        ax = axes[0, col]
        ax.hist(x if col else x[x < 3000], bins=70, color=BLUE)
        ax.set(title=f"{title}  (skewness {stats.skew(x):.2f})", ylabel="reviews")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax = axes[1, col]
        rng = np.random.default_rng(0)
        sub = rng.choice(x, 2000, replace=False)
        (osm, osr), (slope, intercept, _) = stats.probplot(sub, dist="norm")
        ax.scatter(osm, osr, s=6, color=BLUE, alpha=0.5, linewidths=0)
        ax.plot(osm, slope * osm + intercept, color=ORANGE, lw=1.5)
        ax.set(xlabel="normal quantiles", ylabel="sample quantiles", title="normal Q–Q plot (2,000 reviews)")
    save(fig, "box-cox-before-after.png")


def mahalanobis():
    rng = np.random.default_rng(0)
    height = rng.normal(172, 9, 500)
    weight = 0.9 * height - 85 + rng.normal(0, 6, 500)
    X = np.vstack([np.column_stack([height, weight]),
                   [[192, 55]],                                   # tall and light
                   np.column_stack([rng.normal(150, 3, 25), rng.normal(95, 4, 25)])])  # a contaminating group
    cut = stats.chi2.ppf(0.999, df=2)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.scatter(X[:500, 0], X[:500, 1], s=10, color="#a9c7ee", label="typical people")
    ax.scatter(X[501:, 0], X[501:, 1], s=14, color=MUTED, marker="s", label="a contaminating group")
    ax.scatter(*X[500], s=80, color=ORANGE, zorder=3, label="192 cm, 55 kg")
    xx, yy = np.meshgrid(np.linspace(130, 205, 300), np.linspace(35, 120, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for est, color, ls, name in [(EmpiricalCovariance().fit(X), MUTED, "--", "classical"),
                                 (MinCovDet(random_state=0).fit(X), BLUE, "-", "robust (MCD)")]:
        d2 = est.mahalanobis(grid).reshape(xx.shape)
        ax.contour(xx, yy, d2, levels=[cut], colors=color, linestyles=ls, linewidths=1.8)
        ax.plot([], [], color=color, ls=ls, label=f"{name}: d² = χ²₂(0.999)")
    ax.set(xlabel="height (cm)", ylabel="weight (kg)",
           title="Mahalanobis distance: the robust ellipse is not pulled by the outliers")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "mahalanobis-ellipses.png")


if __name__ == "__main__":
    missingness_pattern()
    univariate_outliers()
    box_cox()
    mahalanobis()
