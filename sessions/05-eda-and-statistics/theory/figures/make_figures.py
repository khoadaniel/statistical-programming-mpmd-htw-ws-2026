"""Make the figures for the Session 5 theory pages.

Run from the repository root:
    uv run python sessions/05-eda-and-statistics/theory/figures/make_figures.py

Uses the Inside Airbnb Berlin data in case-study/data/airbnb/ and small built-in tables.
Deterministic: every random step is seeded.
"""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
DATA = Path("case-study/data/airbnb")

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


def load_listings() -> pd.DataFrame:
    return pd.read_parquet(DATA / "listings.parquet")


def short_stay(listings: pd.DataFrame) -> pd.DataFrame:
    """Listings with a price and a minimum stay below 28 nights (comparable prices, Session 4)."""
    return listings[listings["price"].notna() & listings["minimum_nights"].lt(28)]


def fig_skewed_distribution(listings: pd.DataFrame) -> None:
    """Mean versus median on a right-skewed variable (price per night)."""
    price = short_stay(listings)["price"]
    fig, (lin, log) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    lin.hist(price.clip(upper=1000), bins=60, color=GREY, edgecolor="white", linewidth=0.5)
    for ax in (lin, log):
        ax.axvline(price.mean(), color=ORANGE, lw=2)
        ax.axvline(price.median(), color=BLUE, lw=2)
    lin.text(price.mean() + 15, lin.get_ylim()[1] * 0.9, f"mean €{price.mean():.0f}", color=ORANGE,
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    lin.text(price.median() - 15, lin.get_ylim()[1] * 0.75, f"median €{price.median():.0f}",
             color=BLUE, ha="right", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    lin.set(xlabel="EUR per night (values above 1,000 shown at 1,000)",
            ylabel="listings", title="Linear axis: a long right tail")
    bins = np.logspace(np.log10(price.min()), np.log10(price.max()), 45)
    log.hist(price, bins=bins, color=GREY, edgecolor="white", linewidth=0.5)
    log.set_xscale("log")
    log.set(xlabel="EUR per night (log scale)", ylabel="listings",
            title="Log axis: the shape becomes readable")
    fig.suptitle(f"Short-stay listings in Berlin: the mean (€{price.mean():.0f}) is pulled into the tail; "
                 f"the median (€{price.median():.0f}) is not", x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT / "skewed-distribution.png")
    plt.close(fig)


def fig_good_vs_poor() -> None:
    """The same data as a poorly designed and a well-designed chart (reviews per year)."""
    monthly = pd.read_parquet(DATA / "reviews_monthly.parquet")
    yearly = monthly.groupby(monthly["month"].dt.year)["n_reviews"].sum().loc[2015:2025]
    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")

    # poor: truncated axis, rainbow colours, no units, title that says nothing
    colours = plt.cm.rainbow(np.linspace(0, 1, len(yearly)))
    bad.bar(yearly.index, yearly, color=colours)
    bad.set_ylim(20000, 135000)
    bad.set_title("Reviews")
    bad.spines[["top", "right"]].set_visible(True)
    bad.grid(True, color="0.6")
    bad.text(0.02, -0.2, "Poor: bars start at 20,000, rainbow colours, no units, vague title",
             transform=bad.transAxes, color=ORANGE, fontsize=9)

    # good: full axis, colour for one distinction, finding as title
    colours = [ORANGE if year in (2020, 2021) else GREY for year in yearly.index]
    good.bar(yearly.index, yearly, color=colours)
    good.yaxis.set_major_formatter("{x:,.0f}")
    drop = 1 - yearly[2020] / yearly[2019]
    good.set(ylabel="reviews per year (today's listings)",
             title=f"Reviews fell by {drop:.0%} in 2020 and passed\nthe 2019 level only in 2022")
    good.annotate("pandemic years", xy=(2020.5, yearly[2021] + 3000), xytext=(2016, 95000),
                  arrowprops={"arrowstyle": "->", "color": ORANGE}, color=ORANGE, fontsize=9)
    good.grid(axis="y", color="0.9")
    good.text(0.02, -0.2, "Better: full axis from 0, one highlight colour, finding as title",
              transform=good.transAxes, color=BLUE, fontsize=9)
    fig.savefig(OUT / "good-vs-poor-chart.png")
    plt.close(fig)


def fig_simpsons_paradox() -> None:
    """Simpson's paradox (Berkeley 1973) and confounding by the type of stay in the case study."""
    # UC Berkeley graduate admissions, six largest departments (Bickel et al. 1975;
    # R dataset UCBAdmissions): (admitted, applied)
    berkeley = pd.DataFrame(
        {"men": [(512, 825), (353, 560), (120, 325), (138, 417), (53, 191), (22, 373)],
         "women": [(89, 108), (17, 25), (202, 593), (131, 375), (94, 393), (24, 341)]},
        index=list("ABCDEF"))
    rate = berkeley.map(lambda t: t[0] / t[1])
    overall = {g: sum(a for a, _ in berkeley[g]) / sum(n for _, n in berkeley[g])
               for g in ("men", "women")}

    listings = load_listings()
    priced = listings[listings["price"].notna()].copy()
    priced["stay"] = np.where(priced["minimum_nights"].ge(28), "medium-term\n(28+ nights)", "short stay")
    pair = ["Friedrichshain-Kreuzberg", "Charlottenburg-Wilm."]
    d = priced[priced["district"].isin(pair)]
    by_stay = d.groupby(["district", "stay"])["price"].mean().unstack()
    pooled = d.groupby("district")["price"].mean()
    medium_share = d.groupby("district")["stay"].apply(lambda s: s.str.startswith("medium").mean())

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.3]})
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

    groups = ["short stay", "medium-term\n(28+ nights)", "all listings"]
    colours = {pair[0]: BLUE, pair[1]: ORANGE}
    markers = {pair[0]: "o", pair[1]: "s"}
    names = {pair[0]: "Friedrichshain-Kreuzberg", pair[1]: "Charlottenburg-Wilmersdorf"}
    for k, district in enumerate(pair):
        values = [by_stay.loc[district, groups[0]], by_stay.loc[district, groups[1]], pooled[district]]
        xs = np.arange(3) + (k - 0.5) * 0.18
        right.scatter(xs, values, color=colours[district], marker=markers[district], s=60, zorder=3,
                      label=f"{names[district]} ({medium_share[district]:.0%} medium-term)")
        for xx, v in zip(xs, values):
            right.text(xx + (0.1 if k else -0.1), v, f"€{v:.0f}", ha="left" if k else "right",
                       va="center", fontsize=8.5, color=colours[district])
    right.axvline(1.5, color="0.7", lw=1)
    right.set_yscale("log")
    right.set_yticks([25, 50, 100, 200], ["€25", "€50", "€100", "€200"])
    right.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    right.set_xticks(np.arange(3), groups)
    right.set_xlim(-0.6, 2.6)
    right.set(ylabel="mean price per night (log scale)",
              title="Case study: a €10 gap between two districts vanishes\nwithin each type of stay")
    right.legend(frameon=False, loc="center", fontsize=8.5)
    fig.savefig(OUT / "simpsons-paradox.png")
    plt.close(fig)


def fig_regression_line() -> None:
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

    short = short_stay(load_listings())
    lx = short["accommodates"].to_numpy(dtype=float)
    ly = np.log(short["price"].to_numpy())
    jitter = rng.uniform(-0.25, 0.25, len(short))
    right.scatter(lx + jitter, ly, s=4, alpha=0.15, color=GREY, label="listing (jittered)")
    means = pd.DataFrame({"x": lx, "y": ly}).groupby("x")["y"].agg(["mean", "size"])
    means = means[means["size"] >= 30]
    right.plot(means.index, means["mean"], color=BLUE, marker="o", lw=0, ms=6,
               label="mean log price per number of guests")
    c1, c0 = np.polyfit(lx, ly, 1)
    grid = np.array([1, 12])
    right.plot(grid, c0 + c1 * grid, color=ORANGE, lw=2, label=f"OLS line, slope {c1:.3f}")
    rr = np.corrcoef(lx, ly)[0, 1]
    ticks = [25, 50, 100, 200, 400, 800]
    right.set_yticks(np.log(ticks), [f"€{t}" for t in ticks])
    right.set(xlabel="guests (accommodates)", ylabel="price per night (log scale)", xlim=(0.3, 12.7),
              ylim=(np.log(15), np.log(1500)),
              title=f"Case study: larger listings cost more (r = {rr:.2f}, log scale)")
    right.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(OUT / "regression-line.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_skewed_distribution(load_listings())
    fig_good_vs_poor()
    fig_simpsons_paradox()
    fig_regression_line()
    for png in sorted(OUT.glob("*.png")):
        print(png.name, png.stat().st_size // 1024, "KB")
