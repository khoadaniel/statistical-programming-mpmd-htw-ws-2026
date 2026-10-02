"""Figures for the Session 1 theory pages.

Run from the repository root (needs the case-study data, `uv run python case-study/prepare_airbnb.py`):

    uv run python sessions/01-careers-and-python/theory/figures/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")  # write files only, no window

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data/airbnb")
ROOM_TYPES = ["Entire home/apt", "Private room"]  # the two room types with enough listings
COLOUR = "#2d7d74"


def short_stays() -> pd.DataFrame:
    """Listings with a price and a minimum stay below 28 nights (the course convention)."""
    listings = pd.read_parquet(DATA / "listings.parquet")
    return listings[listings["price"].notna() & (listings["minimum_nights"] < 28)]


def price_by_room_type() -> None:
    """Histogram of the price per night (log scale) per room type, with mean and median."""
    short = short_stays()
    bins = np.logspace(np.log10(short["price"].min()), np.log10(short["price"].max()), 50)

    fig, axes = plt.subplots(len(ROOM_TYPES), 1, figsize=(8, 5.5), sharex=True, dpi=130)
    for ax, room_type in zip(axes, ROOM_TYPES, strict=True):
        values = short.loc[short["room_type"] == room_type, "price"]
        ax.hist(values, bins=bins, color=COLOUR, alpha=0.85)
        ax.axvline(values.median(), color="black", lw=1.5, label=f"median €{values.median():.0f}")
        ax.axvline(values.mean(), color="black", lw=1.5, ls="--", label=f"mean €{values.mean():.0f}")
        ax.set_ylabel(f"{room_type}\n(listings)")
        ax.legend(frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("Price per night in EUR, short-stay listings (log scale)")
    fig.suptitle("Price per night: a few expensive listings pull the mean above the median")
    fig.tight_layout()
    fig.savefig(OUT / "price_by_room_type.png")
    plt.close(fig)


def listings_by_district() -> None:
    """Listings per district and the median short-stay price, side by side."""
    listings = pd.read_parquet(DATA / "listings.parquet")
    counts = listings["district"].value_counts().sort_values()
    median = short_stays().groupby("district")["price"].median().reindex(counts.index)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True, dpi=130)
    left.barh(counts.index, counts.to_numpy(), color=COLOUR, height=0.7)
    left.set_xlabel("Listings (all 12,776)")
    right.barh(median.index, median.to_numpy(), color=COLOUR, height=0.7)
    right.set_xlabel("Median price per night of short stays (EUR)")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    for y, value in enumerate(counts.to_numpy()):
        left.text(value + 40, y, f"{value:,}", va="center", fontsize=8)
    for y, value in enumerate(median.to_numpy()):
        right.text(value + 3, y, f"{value:.0f}", va="center", fontsize=8)
    fig.suptitle("Where the listings are, and what a night costs there (snapshot of June 2026)")
    fig.tight_layout()
    fig.savefig(OUT / "listings_by_district.png")
    plt.close(fig)


if __name__ == "__main__":
    price_by_room_type()
    listings_by_district()
