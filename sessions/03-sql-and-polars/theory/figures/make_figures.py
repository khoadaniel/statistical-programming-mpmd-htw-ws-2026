"""Figures for the Session 3 theory pages.

Run from the repository root:
    uv run python sessions/03-sql-and-polars/theory/figures/make_figures.py
Data: Inside Airbnb Berlin (case-study/data/airbnb/). The benchmark numbers are measurements from
workbook 14 (see BENCHMARK below), not recomputed here, so that the figure is deterministic.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
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
    draw_table(ax, 0.0, -1.3, "reviews_monthly", ["listing_id", "month", "n_reviews"], reviews, w=1.1)
    inner = [["1", "Mitte", "2026-04", "3"], ["1", "Mitte", "2026-05", "2"], ["2", "Neukölln", "2026-05", "1"]]
    left = inner + [["3", "Mitte", "NULL", "NULL"]]
    hdr = ["id", "district", "month", "n_reviews"]
    draw_table(ax, 3.9, 0.3, "listings JOIN reviews_monthly: 3 rows", hdr, inner, w=1.15)
    draw_table(ax, 9.6, 0.3, "listings LEFT JOIN reviews_monthly: 4 rows", hdr, left, w=1.15,
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


# measured with workbook 14 on 2026-10-02: macOS, Apple silicon (18 cores, under load from other work),
# pandas 2.3.3, Polars 1.44.2, DuckDB 1.5.6; Inside Airbnb calendar.csv (4,692,075 rows, 159 MB) joined
# with listings.parquet; best of three runs; peak resident memory above a process that only imports the
# libraries. A second run gave the same order (pandas 2.75 s, DuckDB 0.20 s, Polars streaming 0.22 s).
BENCHMARK = pd.DataFrame({
    "tool": ["SQL (DuckDB)", "pandas", "Polars eager", "Polars lazy", "Polars streaming"],
    "seconds": [0.146, 2.184, 0.672, 0.599, 0.192],
    "peak_mb": [237, 1515, 1041, 1127, 534],
})


def benchmark():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    b = BENCHMARK.iloc[::-1]
    axes[0].barh(b["tool"], b["seconds"], color=BLUE, height=0.6)
    axes[0].set(title="Runtime, best of 3 (seconds)", xlabel="seconds")
    axes[1].barh(b["tool"], b["peak_mb"], color=ORANGE, height=0.6)
    axes[1].set(title="Peak resident memory above baseline (MB)", xlabel="MB")
    for ax, col, fmt in [(axes[0], "seconds", "{:.2f} s"), (axes[1], "peak_mb", "{:,.0f} MB")]:
        for y, v in zip(b["tool"], b[col]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, b[col].max() * 1.3)
    fig.suptitle("District-month aggregation of the availability calendar, 4.7 million rows in 159 MB of CSV "
                 "(one laptop; your numbers will differ)", fontsize=10, color=MUTED)
    save(fig, "pandas-polars-benchmark.png")


if __name__ == "__main__":
    join_types()
    reviews_per_year()
    benchmark()
