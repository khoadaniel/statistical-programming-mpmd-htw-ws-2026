"""Make the course-drawn figure of the Session 7 block notebooks (cv_splits.png).

Run from the repository root:
    uv run python sessions/07-validation-and-tuning/figures/make_figures.py
The validation and learning curves are drawn inline in workbooks/02-lecture-regularisation-and-tuning.ipynb.
The other files in this folder come from the INRIA scikit-learn MOOC (see workbooks/extras/README.md).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def cv_splits():
    """k-fold (left) and time-series split (right) as coloured index bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    k = 5
    for ax, kind in zip(axes, ["k-fold (k = 5)", "Time-series split (5 folds)"]):
        for i in range(k):
            yy = k - 1 - i
            if kind.startswith("k-fold"):
                for j in range(k):
                    c = ORANGE if j == i else BLUE
                    ax.add_patch(Rectangle((j / k + 0.003, yy + 0.1), 1 / k - 0.006, 0.8, color=c))
            else:
                n = k + 1
                for j in range(n):
                    if j <= i:
                        c = BLUE
                    elif j == i + 1:
                        c = ORANGE
                    else:
                        c = GRID
                    ax.add_patch(Rectangle((j / n + 0.003, yy + 0.1), 1 / n - 0.006, 0.8, color=c))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, k)
        ax.set_yticks([k - 0.5 - i for i in range(k)], [f"split {i + 1}" for i in range(k)])
        ax.set_xticks([])
        ax.grid(False)
        ax.set_title(kind, color=INK, loc="left")
        ax.set_xlabel("rows (left to right: in time order for the time-series split)", color=MUTED)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRID)]
    fig.legend(handles, ["training", "validation", "not used"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, "cv_splits.png")


if __name__ == "__main__":
    cv_splits()
