"""Make the figures for the Session 5 theory pages.

Run from the repository root:
    uv run python sessions/05-eda-and-statistics/theory/figures/make_figures.py

Uses the case-study data in case-study/data/ and small built-in tables.
Deterministic: every random step is seeded.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
DATA = Path("case-study/data")

# Okabe-Ito colours (colour-blind safe); grey for context
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


def load_sample() -> pd.DataFrame:
    reviews = pd.read_parquet(DATA / "train_sample.parquet")
    reviews["n_words"] = reviews["text"].str.split().str.len()
    reviews["year"] = reviews["date"].dt.year
    return reviews


def fig_skewed_distribution(reviews: pd.DataFrame) -> None:
    """Mean versus median on a right-skewed variable (words per review)."""
    words = reviews["n_words"].clip(lower=1)
    fig, (lin, log) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    lin.hist(words.clip(upper=300), bins=60, color=GREY, edgecolor="white", linewidth=0.5)
    for ax in (lin, log):
        ax.axvline(words.mean(), color=ORANGE, lw=2)
        ax.axvline(words.median(), color=BLUE, lw=2)
    lin.text(words.mean() + 4, lin.get_ylim()[1] * 0.9, f"mean {words.mean():.0f}", color=ORANGE)
    lin.text(words.median() + 4, lin.get_ylim()[1] * 0.75, f"median {words.median():.0f}",
             color=BLUE)
    lin.set(xlabel="words per review (values above 300 shown at 300)", ylabel="reviews",
            title="Linear axis: a long right tail")
    # integer-aligned bin edges avoid empty bins for the short (1-3 word) reviews
    bins = np.unique(np.round(np.logspace(0, np.log10(words.max()), 22))) + 0.5
    bins = np.concatenate([[0.5], bins])
    log.hist(words, bins=bins, color=GREY, edgecolor="white", linewidth=0.5)
    log.set_xscale("log")
    log.set(xlabel="words per review (log scale)", ylabel="reviews",
            title="Log axis: the shape becomes readable")
    fig.suptitle("The mean (35) is pulled into the tail; the median (20) is not",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT / "skewed-distribution.png")
    plt.close(fig)


def fig_good_vs_poor(reviews: pd.DataFrame) -> None:
    """The same data as a poorly designed and a well-designed chart."""
    yearly = reviews.query("year >= 2014").groupby("year")["rating"].agg(
        avg="mean", share_neg=lambda r: (r <= 2).mean())
    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")

    # poor: truncated axis, rainbow colours, no units, title that says nothing
    colours = plt.cm.rainbow(np.linspace(0, 1, len(yearly)))
    bad.bar(yearly.index, yearly["avg"], color=colours)
    bad.set_ylim(3.85, 4.22)
    bad.set_title("Ratings")
    bad.spines[["top", "right"]].set_visible(True)
    bad.grid(True, color="0.6")
    bad.text(0.02, -0.2, "Poor: bars start at 3.85, rainbow colours, no units, vague title",
             transform=bad.transAxes, color=ORANGE, fontsize=9)

    # good: the measure the reader cares about, full axis, one colour, finding as title
    good.plot(yearly.index, yearly["share_neg"], marker="o", ms=6, lw=2, color=ORANGE)
    good.set_ylim(0, 0.3)
    good.yaxis.set_major_formatter("{x:.0%}")
    good.set(ylabel="share of 1–2 star reviews",
             title="Negative reviews rose from 16 % (2014) to 23 % (2021)")
    for year in (2014, 2021):
        value = yearly.loc[year, "share_neg"]
        good.annotate(f"{value:.1%}", (year, value), textcoords="offset points",
                      xytext=(0, 8), ha="center")
    good.grid(axis="y", color="0.9")
    good.text(0.02, -0.2, "Better: full axis from 0, one colour, labelled values, finding as title",
              transform=good.transAxes, color=BLUE, fontsize=9)
    fig.savefig(OUT / "good-vs-poor-chart.png")
    plt.close(fig)


def fig_simpsons_paradox() -> None:
    """Simpson's paradox: Berkeley admissions 1973 and verified purchases by year."""
    # UC Berkeley graduate admissions, six largest departments (Bickel et al. 1975;
    # R dataset UCBAdmissions): (admitted, applied)
    berkeley = pd.DataFrame(
        {"men": [(512, 825), (353, 560), (120, 325), (138, 417), (53, 191), (22, 373)],
         "women": [(89, 108), (17, 25), (202, 593), (131, 375), (94, 393), (24, 341)]},
        index=list("ABCDEF"))
    rate = berkeley.map(lambda t: t[0] / t[1])
    overall = {g: sum(a for a, _ in berkeley[g]) / sum(n for _, n in berkeley[g])
               for g in ("men", "women")}

    reviews = pd.read_parquet(DATA / "train.parquet", columns=["rating", "verified_purchase",
                                                               "date"])
    reviews = reviews[reviews["date"].dt.year >= 2012]
    by_year = reviews.groupby([reviews["date"].dt.year, "verified_purchase"])["rating"].mean()
    by_year = by_year.unstack()
    pooled = reviews.groupby("verified_purchase")["rating"].mean()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(rate) + 1)
    labels = list(rate.index) + ["All"]
    left.scatter(x[:-1], rate["men"], color=BLUE, s=45, label="men", marker="o", zorder=3)
    left.scatter(x[:-1], rate["women"], color=ORANGE, s=45, label="women", marker="s", zorder=3)
    left.scatter([x[-1]], [overall["men"]], color=BLUE, s=90, marker="o", zorder=3)
    left.scatter([x[-1]], [overall["women"]], color=ORANGE, s=90, marker="s", zorder=3)
    left.axvline(x[-1] - 0.5, color="0.7", lw=1)
    left.set_xticks(x, labels)
    left.yaxis.set_major_formatter("{x:.0%}")
    left.set(xlabel="department", ylabel="admission rate", ylim=(0, 0.9),
             title="Berkeley 1973: women ahead in 4 of 6\ndepartments, behind overall")
    left.legend(frameon=False, loc="upper right")

    years = by_year.index.to_numpy()
    right.plot(years, by_year[True], color=BLUE, marker="o", lw=2, label="verified")
    right.plot(years, by_year[False], color=ORANGE, marker="s", lw=2, ls="--",
               label="not verified")
    right.axhline(pooled[True], color=BLUE, lw=1, ls=":",
                  label=f"all years pooled: verified {pooled[True]:.3f}")
    right.axhline(pooled[False], color=ORANGE, lw=1, ls=":",
                  label=f"all years pooled: not verified {pooled[False]:.3f}")
    right.set(xlabel="year of review", ylabel="mean star rating", ylim=(3.6, 4.6),
              title="Reviews: verified purchases rate higher in 8 of 10 years,\n"
                    "lower when pooled (unverified reviews cluster in 2015–2016)")
    right.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.savefig(OUT / "simpsons-paradox.png")
    plt.close(fig)


def fig_regression_line(reviews: pd.DataFrame) -> None:
    """From a scatter plot to the least-squares line, with residuals."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 30)
    y = 1.5 + 0.6 * x + rng.normal(0, 1.0, 30)
    b1, b0 = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    fitted = b0 + b1 * x
    left.vlines(x, fitted, y, color=GREY, lw=1, label="residual = y − ŷ")
    left.scatter(x, y, color=BLUE, s=30, zorder=3, label="observation")
    grid = np.linspace(0, 10, 2)
    left.plot(grid, b0 + b1 * grid, color=ORANGE, lw=2, label=f"ŷ = {b0:.2f} + {b1:.2f}·x")
    left.scatter([x.mean()], [y.mean()], color="black", marker="x", s=60, zorder=4,
                 label="point of means (x̄, ȳ)")
    left.set(xlabel="x", ylabel="y",
             title=f"Least squares: r = {r:.2f}, slope = r · s_y / s_x = {b1:.2f}")
    left.legend(frameon=False, fontsize=8.5, loc="upper left")

    sub = reviews.sample(4000, random_state=1)
    lx = np.log1p(sub["n_words"])
    ly = np.log1p(sub["helpful_vote"])
    jitter = rng.uniform(-0.08, 0.08, len(sub))
    right.scatter(lx, ly + jitter, s=4, alpha=0.25, color=GREY, label="review (jittered)")
    bins = pd.cut(lx, np.arange(0, 8.5, 0.5))
    means = pd.DataFrame({"x": lx, "y": ly}).groupby(bins, observed=True).mean()
    right.plot(means["x"], means["y"], color=BLUE, marker="o", lw=0, ms=6,
               label="mean per length bin")
    c1, c0 = np.polyfit(lx, ly, 1)
    grid = np.linspace(lx.min(), lx.max(), 2)
    right.plot(grid, c0 + c1 * grid, color=ORANGE, lw=2, label=f"OLS line, slope {c1:.2f}")
    rr = np.corrcoef(lx, ly)[0, 1]
    right.set(xlabel="log(1 + words)", ylabel="log(1 + helpful votes)",
              title=f"Case study: longer reviews get more votes (r = {rr:.2f})")
    right.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(OUT / "regression-line.png")
    plt.close(fig)


if __name__ == "__main__":
    sample = load_sample()
    fig_skewed_distribution(sample)
    fig_good_vs_poor(sample)
    fig_simpsons_paradox()
    fig_regression_line(sample)
    for png in sorted(OUT.glob("*.png")):
        print(png.name, png.stat().st_size // 1024, "KB")
