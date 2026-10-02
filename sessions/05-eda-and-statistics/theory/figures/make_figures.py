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
    decisions = pd.read_parquet(DATA / "train_sample.parquet")
    decisions["n_chars"] = decisions["description"].str.len()
    decisions["n_keywords"] = decisions["keywords"].str.split(",").str.len()
    return decisions


def fig_skewed_distribution(decisions: pd.DataFrame) -> None:
    """Mean versus median on a right-skewed variable (characters per description)."""
    chars = decisions["n_chars"]
    fig, (lin, log) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    lin.hist(chars.clip(upper=3000), bins=60, color=GREY, edgecolor="white", linewidth=0.5)
    for ax in (lin, log):
        ax.axvline(chars.mean(), color=ORANGE, lw=2)
        ax.axvline(chars.median(), color=BLUE, lw=2)
    lin.text(chars.mean() + 40, lin.get_ylim()[1] * 0.9, f"mean {chars.mean():.0f}", color=ORANGE)
    lin.text(chars.median() - 40, lin.get_ylim()[1] * 0.75, f"median {chars.median():.0f}",
             color=BLUE, ha="right")
    lin.set(xlabel="characters per description (values above 3,000 shown at 3,000)",
            ylabel="decisions", title="Linear axis: a long right tail")
    bins = np.logspace(np.log10(chars.min()), np.log10(chars.max()), 45)
    log.hist(chars, bins=bins, color=GREY, edgecolor="white", linewidth=0.5)
    log.set_xscale("log")
    log.set(xlabel="characters per description (log scale)", ylabel="decisions",
            title="Log axis: the shape becomes readable")
    fig.suptitle(f"The mean ({chars.mean():.0f}) is pulled into the tail; "
                 f"the median ({chars.median():.0f}) is not", x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT / "skewed-distribution.png")
    plt.close(fig)


def fig_good_vs_poor() -> None:
    """The same data as a poorly designed and a well-designed chart (decisions per year)."""
    counts = pd.read_parquet(DATA / "monthly_counts.parquet")
    counts["year"] = counts["month"].dt.year
    counts = counts[counts["year"].between(2015, 2025)]
    counts["uk"] = counts["issuing_country"].eq("GB")
    yearly = counts.pivot_table(index="year", columns="uk", values="n_decisions", aggfunc="sum",
                                fill_value=0)
    total = yearly.sum(axis=1)
    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")

    # poor: truncated axis, rainbow colours, no units, title that says nothing
    colours = plt.cm.rainbow(np.linspace(0, 1, len(total)))
    bad.bar(total.index, total, color=colours)
    bad.set_ylim(38000, 52500)
    bad.set_title("Decisions")
    bad.spines[["top", "right"]].set_visible(True)
    bad.grid(True, color="0.6")
    bad.text(0.02, -0.2, "Poor: bars start at 38,000, rainbow colours, no units, vague title",
             transform=bad.transAxes, color=ORANGE, fontsize=9)

    # good: full axis, colour for one distinction, finding as title
    good.bar(yearly.index, yearly[False], color=GREY, label="other countries")
    good.bar(yearly.index, yearly[True], bottom=yearly[False], color=ORANGE,
             label="United Kingdom")
    good.yaxis.set_major_formatter("{x:,.0f}")
    fall = total[2017] - total[2021]
    good.set(ylabel="decisions per year (start of validity)",
             title=f"Decisions fell by a fifth from 2017 to 2021;\n"
                   f"a third of the fall ({yearly.loc[2017, True]:,} of {fall:,}) is the UK leaving")
    good.legend(frameon=False, loc="lower left", fontsize=9)
    good.grid(axis="y", color="0.9")
    good.text(0.02, -0.2, "Better: full axis from 0, one highlight colour, finding as title",
              transform=good.transAxes, color=BLUE, fontsize=9)
    fig.savefig(OUT / "good-vs-poor-chart.png")
    plt.close(fig)


def fig_simpsons_paradox() -> None:
    """Simpson's paradox (Berkeley 1973) and milder confounding by language in the case study."""
    # UC Berkeley graduate admissions, six largest departments (Bickel et al. 1975;
    # R dataset UCBAdmissions): (admitted, applied)
    berkeley = pd.DataFrame(
        {"men": [(512, 825), (353, 560), (120, 325), (138, 417), (53, 191), (22, 373)],
         "women": [(89, 108), (17, 25), (202, 593), (131, 375), (94, 393), (24, 341)]},
        index=list("ABCDEF"))
    rate = berkeley.map(lambda t: t[0] / t[1])
    overall = {g: sum(a for a, _ in berkeley[g]) / sum(n for _, n in berkeley[g])
               for g in ("men", "women")}

    decisions = load_sample().dropna(subset=["keywords"])
    decisions = decisions[decisions["language"].isin(["de", "fr", "en"])]
    decisions["log_chars"] = np.log10(decisions["n_chars"])

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

    colours = {"de": BLUE, "fr": ORANGE, "en": GREEN}
    names = {"de": "German", "fr": "French", "en": "English"}
    edges = np.arange(1.0, 3.81, 0.2)
    for lang, g in decisions.groupby("language"):
        bins = pd.cut(g["log_chars"], edges)
        means = g.groupby(bins, observed=True)[["log_chars", "n_keywords"]].mean()
        means = means[g.groupby(bins, observed=True).size() >= 30]
        right.scatter(means["log_chars"], means["n_keywords"], color=colours[lang], s=22, zorder=3)
        b1, b0 = np.polyfit(g["log_chars"], g["n_keywords"], 1)
        grid = np.array([g["log_chars"].quantile(0.02), g["log_chars"].quantile(0.98)])
        right.plot(grid, b0 + b1 * grid, color=colours[lang], lw=2,
                   label=f"{names[lang]}: slope {b1:.1f}")
    b1, b0 = np.polyfit(decisions["log_chars"], decisions["n_keywords"], 1)
    grid = np.array([2.0, 3.4])
    right.plot(grid, b0 + b1 * grid, color="black", lw=2, ls="--",
               label=f"three languages pooled: slope {b1:.1f}")
    right.set(xlabel="log10(characters in description)", ylabel="mean number of keywords",
              title="Case study: German and French slopes are steeper than\n"
                    "the pooled slope (long German texts, few extra keywords)")
    right.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.savefig(OUT / "simpsons-paradox.png")
    plt.close(fig)


def fig_regression_line(decisions: pd.DataFrame) -> None:
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

    sub = decisions.dropna(subset=["n_keywords"]).sample(4000, random_state=1)
    lx = np.log(sub["n_chars"])
    ly = sub["n_keywords"]
    jitter = rng.uniform(-0.3, 0.3, len(sub))
    right.scatter(lx, ly + jitter, s=4, alpha=0.25, color=GREY, label="decision (jittered)")
    bins = pd.cut(lx, np.arange(2, 9.5, 0.5))
    means = pd.DataFrame({"x": lx, "y": ly}).groupby(bins, observed=True).mean()
    right.plot(means["x"], means["y"], color=BLUE, marker="o", lw=0, ms=6,
               label="mean per length bin")
    full = decisions.dropna(subset=["n_keywords"])
    c1, c0 = np.polyfit(np.log(full["n_chars"]), full["n_keywords"], 1)   # all 50,000, as on the page
    grid = np.linspace(lx.min(), lx.max(), 2)
    right.plot(grid, c0 + c1 * grid, color=ORANGE, lw=2, label=f"OLS line, slope {c1:.2f}")
    rr = np.corrcoef(np.log(full["n_chars"]), full["n_keywords"])[0, 1]
    right.set(xlabel="log(characters in description)", ylabel="number of keywords",
              ylim=(0, 20),
              title=f"Case study: longer descriptions, more keywords (r = {rr:.2f})")
    right.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(OUT / "regression-line.png")
    plt.close(fig)


if __name__ == "__main__":
    sample = load_sample()
    fig_skewed_distribution(sample)
    fig_good_vs_poor()
    fig_simpsons_paradox()
    fig_regression_line(sample)
    for png in sorted(OUT.glob("*.png")):
        print(png.name, png.stat().st_size // 1024, "KB")
