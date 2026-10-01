"""Figures for the Session 1 theory pages.

Run from the repository root (needs the case-study data):

    uv run python sessions/01-careers-and-python/theory/figures/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")  # write files only, no window

OUT = Path(__file__).resolve().parent
LANGUAGES = ["de", "fr", "en", "nl", "pl"]  # the five most frequent languages
COLOUR = "#2d7d74"


def description_length_by_language() -> None:
    """Histogram of description length (log scale) per language, with mean and median."""
    decisions = pd.read_parquet("case-study/data/train_sample.parquet")
    lengths = decisions["description"].str.len().clip(lower=1)
    bins = np.logspace(1, np.log10(lengths.max() + 1), 50)

    fig, axes = plt.subplots(len(LANGUAGES), 1, figsize=(8, 8), sharex=True, dpi=130)
    for ax, language in zip(axes, LANGUAGES, strict=True):
        values = lengths[decisions["language"] == language]
        ax.hist(values, bins=bins, color=COLOUR, alpha=0.85)
        ax.axvline(values.median(), color="black", lw=1.5, label=f"median {values.median():.0f}")
        ax.axvline(values.mean(), color="black", lw=1.5, ls="--", label=f"mean {values.mean():.0f}")
        ax.set_ylabel(f"{language}\n(count)")
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("Length of the description of goods in characters (log scale)")
    fig.suptitle("Description length by language: long texts pull the mean above the median")
    fig.tight_layout()
    fig.savefig(OUT / "description_length_by_language.png")
    plt.close(fig)


if __name__ == "__main__":
    description_length_by_language()
