"""Figures for the Session 9 lecture notebooks.

Run from the repository root:
    uv run python sessions/09-feature-engineering/figures/make_figures.py

Deterministic.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def cyclical_month_encoding():
    months = np.arange(1, 13)
    angle = 2 * np.pi * months / 12
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRID, lw=2, zorder=1)
    ax.scatter(np.cos(angle), np.sin(angle), s=90, color=BLUE, edgecolor="white", lw=2, zorder=3)
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m, a, name in zip(months, angle, names):
        ax.annotate(name, (np.cos(a), np.sin(a)), xytext=(1.18 * np.cos(a), 1.18 * np.sin(a)),
                    ha="center", va="center", color=INK)
    for m in (12, 1):
        a = 2 * np.pi * m / 12
        ax.scatter([np.cos(a)], [np.sin(a)], s=130, color=ORANGE, edgecolor="white", lw=2, zorder=4)
    ax.set_xlabel("month_cos = cos(2π · month / 12)")
    ax.set_ylabel("month_sin = sin(2π · month / 12)")
    ax.set_title("Cyclical encoding: December and January are neighbours", fontsize=11, color=INK)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "cyclical_month_encoding.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cyclical_month_encoding()
    print("figures written to", OUT)
