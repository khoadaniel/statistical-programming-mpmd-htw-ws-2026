"""Draw the figures of the two dataset introductions (case-study/airbnb/ and case-study/ebti/).

    uv run python case-study/make_dataset_figures.py

Reads the student data in case-study/data/ (the course-drive download) and writes PNG files to
case-study/airbnb/figures/ and case-study/ebti/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})
thousands = FuncFormatter(lambda x, _: f"{x:,.0f}")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path.relative_to(HERE.parent))


# ---------------------------------------------------------------- Airbnb

def airbnb() -> None:
    a = DATA / "airbnb"
    listings = pd.read_parquet(a / "listings.parquet")
    reviews = pd.read_parquet(a / "reviews_monthly.parquet")
    out = HERE / "airbnb" / "figures"
    short = listings[(listings["minimum_nights"] < 28) & listings["price"].notna()]

    # 1 · map of the listings by room type
    fig, ax = plt.subplots(figsize=(7, 5.6))
    groups = [("Entire home/apt", "Entire home or flat", BLUE),
              ("Private room", "Private room", ORANGE)]
    for key, label, colour in groups:
        d = listings[listings["room_type"] == key]
        ax.scatter(d["longitude"], d["latitude"], s=3, c=colour, alpha=0.55, linewidths=0, label=f"{label} ({len(d):,})")
    other = listings[~listings["room_type"].isin([g[0] for g in groups])]
    ax.scatter(other["longitude"], other["latitude"], s=6, c=AQUA, alpha=0.9, linewidths=0,
               label=f"Hotel or shared room ({len(other):,})")
    ax.set_title(f"{len(listings):,} Berlin listings, one dot each")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect(1 / np.cos(np.radians(52.5)))
    ax.legend(loc="lower left", frameon=False, markerscale=4)
    save(fig, out / "map-room-types.png")

    # 2 · price per night of short stays, log scale
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(np.log10(10), np.log10(2000), 60)
    ax.hist(short["price"].clip(upper=2000), bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000], ["€20", "€50", "€100", "€200", "€500", "€1,000", "€2,000+"])
    med = short["price"].median()
    ax.axvline(med, color=INK, linewidth=1)
    ax.annotate(f"median €{med:,.0f}", (med, ax.get_ylim()[1] * 0.92), xytext=(6, 0), textcoords="offset points", color=INK)
    ax.set_title(f"Price per night of {len(short):,} short-stay listings (log scale)")
    ax.set_xlabel("price per night"); ax.set_ylabel("listings")
    ax.grid(axis="x", visible=False)
    save(fig, out / "price-distribution.png")

    # 3 · median price by district
    by = short.groupby("district")["price"].agg(["median", "size"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(by.index, by["median"], color=BLUE, height=0.65)
    for i, (m, n) in enumerate(zip(by["median"], by["size"])):
        ax.text(m + 2, i, f"€{m:,.0f}  ({n:,} listings)", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, by["median"].max() * 1.45)
    ax.set_title("Median price per night by district, short stays")
    ax.set_xlabel("median price per night (€)")
    ax.grid(axis="y", visible=False)
    save(fig, out / "price-by-district.png")

    # 4 · minimum stay
    edges = [0, 1, 2, 3, 7, 27, 91, 92, np.inf]
    labels = ["1", "2", "3", "4–7", "8–27", "28–91", "92", "93+"]
    cat = pd.cut(listings["minimum_nights"], edges, labels=labels)
    counts = cat.value_counts().reindex(labels)
    colours = [BLUE] * 5 + [ORANGE] * 3
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, counts.values, color=colours, width=0.7)
    for x, c in zip(labels, counts.values):
        ax.text(x, c, f"{c:,}", ha="center", va="bottom", color=INK_2, fontsize=9)
    long_share = (listings["minimum_nights"] >= 28).mean()
    ax.set_title(f"Minimum stay: {long_share:.0%} of listings ask for 28 nights or more")
    ax.set_xlabel("minimum stay (nights)"); ax.set_ylabel("listings")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    save(fig, out / "minimum-nights.png")

    # 5 · reviews per month
    monthly = reviews.groupby("month")["n_reviews"].sum()
    monthly = monthly[monthly.index < monthly.index.max()]  # the snapshot month is incomplete
    monthly = monthly[monthly.index >= "2015-01-01"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(monthly.index, monthly.values, color=BLUE, linewidth=2)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color=GRID, alpha=0.6, linewidth=0)
    ax.text(pd.Timestamp("2020-04-01"), monthly.max() * 0.9, "pandemic\n2020–21", color=INK_2, fontsize=9)
    ax.set_title("Reviews per month, a proxy for stays")
    ax.set_ylabel("reviews"); ax.yaxis.set_major_formatter(thousands)
    save(fig, out / "reviews-per-month.png")


# ---------------------------------------------------------------- EBTI

def ebti() -> None:
    e = DATA / "ebti"
    labelled = pd.concat([pd.read_parquet(e / "train.parquet"), pd.read_parquet(e / "val.parquet")])
    nomenclature = pd.read_parquet(e / "nomenclature.parquet")
    out = HERE / "ebti" / "figures"

    # 1 · languages
    names = {"de": "German", "fr": "French", "nl": "Dutch", "es": "Spanish", "cs": "Czech", "pl": "Polish",
             "it": "Italian", "en": "English", "sv": "Swedish", "hu": "Hungarian"}
    share = labelled["language"].value_counts(normalize=True)
    top = share.head(7)
    top["other"] = share.iloc[7:].sum()
    top.index = [names.get(i, i) if i != "other" else f"{share.size - 7} other languages" for i in top.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(top.index[::-1], top.values[::-1] * 100, color=BLUE, height=0.65)
    for i, v in enumerate(top.values[::-1] * 100):
        ax.text(v + 0.8, i, f"{v:.1f} %", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, top.max() * 100 * 1.15)
    ax.set_title("Language of the description")
    ax.set_xlabel("share of decisions (%)")
    ax.grid(axis="y", visible=False)
    save(fig, out / "languages.png")

    # 2 · long tail of headings
    counts = labelled["heading"].value_counts()
    half = int((counts.cumsum() / counts.sum() < 0.5).sum() + 1)
    rare = int((counts < 10).sum())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(np.arange(1, len(counts) + 1), counts.values, width=1.0, color=BLUE, linewidth=0)
    ax.axvline(half, color=INK, linewidth=1)
    ax.annotate(f"the {half} largest headings\nhold half of all decisions", (half, counts.iloc[0] * 0.55),
                xytext=(8, 0), textcoords="offset points", color=INK)
    ax.annotate(f"{rare} headings have\nfewer than 10 decisions", (len(counts) - rare / 2, 12),
                xytext=(-40, 60), textcoords="offset points", color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    ax.set_yscale("log")
    ax.set_title(f"{len(counts):,} headings, sorted by number of decisions (log scale)")
    ax.set_xlabel("headings, from most to least frequent"); ax.set_ylabel("decisions")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    save(fig, out / "long-tail.png")

    # 3 · the ten most frequent headings
    short_names = {
        "3926": "other articles of plastics", "9503": "toys, dolls, puzzles, scale models",
        "6307": "other made-up textile articles", "4202": "bags, cases and suitcases",
        "2106": "food preparations n.e.s.", "7326": "other articles of iron or steel",
        "8536": "switches and connectors (≤ 1,000 V)", "6404": "footwear with textile uppers",
        "9405": "lamps and light fittings", "7318": "screws, bolts, nuts, washers",
    }
    top10 = counts.head(10)
    lab = [f"{h}  {short_names.get(h, nomenclature.set_index('heading').heading_description.get(h, '')[:35])}"
           for h in top10.index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(lab[::-1], top10.values[::-1], color=BLUE, height=0.65)
    for i, v in enumerate(top10.values[::-1]):
        ax.text(v + 40, i, f"{v:,}", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, top10.max() * 1.15)
    ax.set_title("The ten most frequent headings")
    ax.set_xlabel("decisions (train + val)"); ax.xaxis.set_major_formatter(thousands)
    ax.grid(axis="y", visible=False)
    save(fig, out / "top-headings.png")

    # 4 · description length
    words = labelled["description"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(words.clip(upper=400), bins=80, color=BLUE, edgecolor="white", linewidth=0.4)
    med = words.median()
    ax.axvline(med, color=INK, linewidth=1)
    ax.annotate(f"median {med:.0f} words", (med, ax.get_ylim()[1] * 0.9), xytext=(6, 0), textcoords="offset points")
    ax.set_title("Length of the description of goods")
    ax.set_xlabel("words (descriptions above 400 words shown at 400)"); ax.set_ylabel("decisions")
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(axis="x", visible=False)
    save(fig, out / "description-length.png")


if __name__ == "__main__":
    airbnb()
    ebti()
