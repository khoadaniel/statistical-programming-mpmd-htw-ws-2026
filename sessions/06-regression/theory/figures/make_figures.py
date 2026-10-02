"""Make the figures for the Session 6 theory pages.

Run from the repository root:
    uv run python sessions/06-regression/theory/figures/make_figures.py

Uses seeded toy data and the case-study sample in case-study/data/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

OUT = Path(__file__).parent
DATA = Path("case-study/data")
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8C8C8C"
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
})


def fig_residuals() -> None:
    """Residual-versus-fitted plots: a good fit and the description-length model."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 300)
    y = 2 + 0.8 * x + rng.normal(0, 1, 300)
    lin = LinearRegression().fit(x.reshape(-1, 1), y)
    fitted = lin.predict(x.reshape(-1, 1))

    decisions = pd.read_parquet(DATA / "train_sample.parquet")
    nomenclature = pd.read_parquet(DATA / "nomenclature.parquet")
    decisions = decisions.merge(nomenclature[["heading", "section"]], on="heading")
    top = decisions["language"].value_counts().index[:8]
    decisions["language"] = decisions["language"].where(decisions["language"].isin(top), "other")
    decisions["year"] = decisions["start_date"].dt.year - 2017
    lx = pd.get_dummies(decisions[["year", "language", "section"]], drop_first=True, dtype=float)
    ly = np.log(decisions["description"].str.len())
    model = LinearRegression().fit(lx, ly)
    f2 = model.predict(lx)
    res2 = ly - f2
    keep = rng.choice(len(lx), 4000, replace=False)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    left.scatter(fitted, y - fitted, s=10, alpha=0.6, color=BLUE)
    left.axhline(0, color="black", lw=1)
    left.set(xlabel="fitted value ŷ", ylabel="residual y − ŷ",
             title="Good: a structureless band around zero")
    right.scatter(f2[keep], res2.iloc[keep], s=5, alpha=0.3, color=GREY)
    right.axhline(0, color="black", lw=1)
    right.annotate("descriptions much shorter\nthan expected: a long\nlower tail", xy=(6.08, -3.1),
                   xytext=(5.15, -3.4), arrowprops={"arrowstyle": "->"}, fontsize=9)
    right.set(xlabel="fitted log(characters)", ylabel="residual",
              title="Case study: centred on zero, but a long lower tail")
    fig.savefig(OUT / "residual-plot.png")
    plt.close(fig)


def fig_under_overfitting() -> None:
    """Polynomial fits of degree 1, 4 and 15, and training versus test error."""
    rng = np.random.default_rng(0)
    # 20 noisy training points; a large test set from the same source
    x_tr = np.sort(rng.uniform(0, 1, 20))
    y_tr = np.cos(1.5 * np.pi * x_tr) + rng.normal(0, 0.15, 20)
    x_te = rng.uniform(0, 1, 1000)
    y_te = np.cos(1.5 * np.pi * x_te) + rng.normal(0, 0.15, 1000)
    grid = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    for ax, degree, name in zip(axes[:3], (1, 4, 15), ("underfit", "good fit", "overfit"),
                                strict=True):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_tr.reshape(-1, 1), y_tr)
        ax.plot(grid, np.cos(1.5 * np.pi * grid), color=GREY, lw=1, ls="--", label="true")
        ax.plot(grid, model.predict(grid.reshape(-1, 1)), color=ORANGE, lw=2, label="model")
        ax.scatter(x_tr, y_tr, s=14, color=BLUE, label="training data", zorder=3)
        ax.set(ylim=(-1.6, 1.6), xticks=[], yticks=[], title=f"degree {degree}: {name}")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    degrees = np.arange(1, 16)
    err_tr, err_te = [], []
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_tr.reshape(-1, 1), y_tr)
        err_tr.append(root_mean_squared_error(y_tr, model.predict(x_tr.reshape(-1, 1))))
        err_te.append(root_mean_squared_error(y_te, model.predict(x_te.reshape(-1, 1))))
    ax = axes[3]
    ax.plot(degrees, err_tr, color=BLUE, marker="o", ms=4, lw=2, label="training error")
    ax.plot(degrees, err_te, color=ORANGE, marker="s", ms=4, lw=2, ls="--",
            label="test error")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 3)
    best = degrees[int(np.argmin(err_te))]
    ax.axvline(best, color=GREY, lw=1)
    ax.text(best + 0.3, 0.3, f"lowest test error:\ndegree {best}", fontsize=8.5)
    ax.text(0.02, 0.02, "← high bias", fontsize=8.5, color=GREY, transform=ax.transAxes)
    ax.text(0.98, 0.02, "high variance →", fontsize=8.5, color=GREY, ha="right",
            transform=ax.transAxes)
    ax.set(xlabel="polynomial degree (model complexity)", ylabel="RMSE (log scale)",
           title="Training vs test error")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(OUT / "under-overfitting.png")
    plt.close(fig)


def fig_sigmoid() -> None:
    """The sigmoid maps log-odds to probabilities."""
    z = np.linspace(-6, 6, 400)
    p = 1 / (1 + np.exp(-z))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    left.plot(z, p, color=BLUE, lw=2.5)
    left.axhline(0.5, color=GREY, lw=1, ls=":")
    left.axvline(0, color=GREY, lw=1, ls=":")
    for zz in (-2, 0, 2):
        pp = 1 / (1 + np.exp(-zz))
        left.scatter([zz], [pp], color=ORANGE, zorder=3, s=30)
        left.annotate(f"z = {zz}: p = {pp:.2f}", (zz, pp), textcoords="offset points",
                      xytext=(8, -12 if zz > 0 else (-4 if zz < 0 else 6)), fontsize=9)
    left.set(xlabel="z = b₀ + b₁x₁ + … (log-odds)", ylabel="probability p = σ(z)",
             ylim=(-0.03, 1.03), title="σ(z) = 1 / (1 + e^(−z))")

    # churn example: log-odds fall linearly with tenure (coefficients from the Telco fit)
    tenure = np.linspace(0, 72, 200)
    b0, b1 = 0.008, -0.0382
    right.plot(tenure, 1 / (1 + np.exp(-(b0 + b1 * tenure))), color=ORANGE, lw=2.5)
    for months in (1, 12, 60):
        pp = 1 / (1 + np.exp(-(b0 + b1 * months)))
        right.scatter([months], [pp], color=BLUE, zorder=3, s=30)
        right.annotate(f"{months} month{'s' if months > 1 else ''}: {pp:.0%}", (months, pp), textcoords="offset points",
                       xytext=(6, 6), fontsize=9)
    right.yaxis.set_major_formatter("{x:.0%}")
    right.set(xlabel="tenure (months)", ylabel="predicted churn probability", ylim=(0, 0.6),
              title="Telco: odds of churn × 0.96 per month of tenure")
    fig.savefig(OUT / "sigmoid.png")
    plt.close(fig)


def fig_robust() -> None:
    """Least squares versus Huber with a few gross outliers."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 80)
    y = 2 + 0.5 * x + rng.normal(0, 0.5, 80)
    bad = np.argsort(x)[-8:]
    y[bad] -= rng.uniform(6, 9, 8)
    X = x.reshape(-1, 1)
    grid = np.linspace(0, 10, 2).reshape(-1, 1)
    ols = LinearRegression().fit(X, y)
    hub = HuberRegressor().fit(X, y)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    mask = np.zeros(len(x), bool)
    mask[bad] = True
    ax.scatter(x[~mask], y[~mask], s=16, color=BLUE, label="data")
    ax.scatter(x[mask], y[mask], s=28, color=ORANGE, marker="x", label="gross errors")
    ax.plot(grid, ols.predict(grid), color=GREY, lw=2, ls="--",
            label=f"least squares, slope {ols.coef_[0]:.2f}")
    ax.plot(grid, hub.predict(grid), color=GREEN, lw=2,
            label=f"Huber, slope {hub.coef_[0]:.2f}")
    ax.set(xlabel="x", ylabel="y", title="Eight bad points tilt least squares (true slope 0.5)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(OUT / "robust-vs-ols.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_residuals()
    fig_under_overfitting()
    fig_sigmoid()
    fig_robust()
    for png in sorted(OUT.glob("*.png")):
        print(png.name, png.stat().st_size // 1024, "KB")
