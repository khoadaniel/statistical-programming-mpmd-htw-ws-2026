"""Figures for the Session 3 lecture notebooks.

Run from the repository root:
    uv run python sessions/03-sql/figures/make_figures.py
Data: Inside Airbnb Berlin (case-study/data/airbnb/), snapshot of 26 June 2026, CC BY 4.0.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "case-study" / "data"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(HERE / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def draw_table(ax, x, y, title, header, rows, highlight=None, w=1.0, h=0.32):
    """Draw a small table with its top-left corner at (x, y)."""
    ax.text(x, y + 0.12, title, fontsize=10, fontweight="bold", color=INK, va="bottom")
    cols = len(header)
    for j, label in enumerate(header):
        ax.add_patch(plt.Rectangle((x + j * w, y - h), w, h, facecolor="#f1f0ec", edgecolor=MUTED, lw=0.8))
        ax.text(x + j * w + w / 2, y - h / 2, label, ha="center", va="center", fontsize=8, color=INK)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            face = "white"
            if highlight and i in highlight:
                face = highlight[i]
            ax.add_patch(plt.Rectangle((x + j * w, y - (i + 2) * h), w, h, facecolor=face, edgecolor=MUTED, lw=0.8))
            color = MUTED if value == "NULL" else INK
            ax.text(x + j * w + w / 2, y - (i + 1.5) * h, value, ha="center", va="center", fontsize=9, color=color)
    return x + cols * w


def join_types():
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(-2.6, 0.6)
    ax.axis("off")
    listings = [["1", "Mitte"], ["2", "Neukölln"], ["3", "Mitte"]]
    reviews = [["1", "2026-04", "3"], ["1", "2026-05", "2"], ["2", "2026-05", "1"]]
    draw_table(ax, 0.0, 0.3, "listings", ["id", "district"], listings, highlight={2: "#fde3d8"}, w=1.55)
    draw_table(ax, 0.0, -1.3, "reviews", ["listing_id", "month", "n_reviews"], reviews, w=1.1)
    inner = [["1", "Mitte", "2026-04", "3"], ["1", "Mitte", "2026-05", "2"], ["2", "Neukölln", "2026-05", "1"]]
    left = inner + [["3", "Mitte", "NULL", "NULL"]]
    hdr = ["id", "district", "month", "n_reviews"]
    draw_table(ax, 3.9, 0.3, "listings JOIN reviews: 3 rows", hdr, inner, w=1.15)
    draw_table(ax, 9.6, 0.3, "listings LEFT JOIN reviews: 4 rows", hdr, left, w=1.15,
               highlight={3: "#fde3d8"})
    ax.text(9.6, -1.75, "Listing 3 has no review: kept, with NULL\nin the review columns. "
            "WHERE month\nIS NULL keeps only such rows (anti-join).", fontsize=9, color=MUTED, va="top")
    ax.text(3.9, -1.75, "Listing 3 has no review: dropped.\nListing 1 appears twice: one row per\nmatching review month.",
            fontsize=9, color=MUTED, va="top")
    save(fig, "join-types.png")


def reviews_per_year():
    reviews = pd.read_parquet(DATA / "airbnb" / "reviews_monthly.parquet")
    n = reviews.groupby(reviews["month"].dt.year)["n_reviews"].sum()
    n = n[(n.index >= 2012) & (n.index <= 2025)]          # 2026 is incomplete
    change = n.diff()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    axes[0].bar(n.index, n.values / 1000, color=BLUE, width=0.7)
    axes[0].set(title="Reviews per year (SUM(n_reviews) ... GROUP BY year)", ylabel="thousand reviews")
    colors = [AQUA if v >= 0 else ORANGE for v in change.fillna(0)]
    axes[1].bar(change.index, change.fillna(0).values / 1000, color=colors, width=0.7)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set(title="Change vs previous year (n - LAG(n) OVER (ORDER BY year))", ylabel="thousand reviews")
    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(range(2012, 2026, 2))
    axes[0].text(2012, n.max() / 1000 * 0.92, f"running total 2012-2025: {n.sum():,}\n"
                 "(listings still online in June 2026 only)", fontsize=9, color=MUTED)
    save(fig, "reviews-per-year-window.png")


if __name__ == "__main__":
    join_types()
    reviews_per_year()
