"""Figures for the Session 16 theory pages. Run from the repository root:

    uv run python sessions/16-deployment-and-monitoring/theory/figures/make_figures.py

Uses the EBTI case-study data: the training sample, the inputs of the test decisions, and the 2024 labels
that are released to the students in Session 16 (case-study/data/feedback_2024.csv, or the path in the
environment variable FEEDBACK_PATH; written by workbooks/make_feedback_2024.py). Deterministic.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data")
FEEDBACK = Path(os.environ.get("FEEDBACK_PATH", DATA / "feedback_2024.csv"))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})

train = pd.read_parquet(DATA / "train_sample.parquet", columns=["issuing_country", "start_date", "chapter"])
test = pd.read_parquet(DATA / "test.parquet", columns=["id", "issuing_country", "start_date"])
t2024 = test[test["start_date"].dt.year == 2024]
t2526 = test[test["start_date"].dt.year >= 2025]
feedback = pd.read_csv(FEEDBACK, dtype=str)

# ------------------------------------------------------------ figure 1: country drift + chapter label shift
countries = ["DE", "FR", "NL", "GB", "PL", "CZ", "ES"]
shares = pd.DataFrame({"training 2017-2023": train["issuing_country"].value_counts(normalize=True),
                       "2024": t2024["issuing_country"].value_counts(normalize=True),
                       "2025-2026": t2526["issuing_country"].value_counts(normalize=True)}).fillna(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=130, gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(countries))
for k, (col, color) in enumerate(zip(shares.columns, [BLUE, ORANGE, AQUA])):
    ax1.bar(x + (k - 1) * 0.27, shares.loc[countries, col] * 100, 0.26, color=color, label=col)
ax1.set_xticks(x, countries)
ax1.set(ylabel="share of decisions (%)", yscale="log", ylim=(0.5, 100))
ax1.annotate("GB: 3.6 % -> 0 %\n(Brexit)", (3, 4), xytext=(3.4, 20), fontsize=9, color=INK2,
             arrowprops={"arrowstyle": "->", "color": INK2})
ax1.set_title("Input drift: issuing countries", loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9)
ax1.grid(axis="y", color=GRID, lw=0.8)

y2023 = train[train["start_date"].dt.year == 2023]["chapter"].value_counts(normalize=True)
fb = t2024.merge(feedback, on="id")
y2024 = fb["heading"].str[:2].value_counts(normalize=True)
change = (y2024.reindex(y2023.index.union(y2024.index), fill_value=0)
          - y2023.reindex(y2023.index.union(y2024.index), fill_value=0))
top = change.reindex(change.abs().sort_values(ascending=False).index[:6])
ax2.barh(range(len(top))[::-1], top.to_numpy() * 100, color=[ORANGE if v > 0 else BLUE for v in top])
ax2.set_yticks(range(len(top))[::-1], [f"chapter {c}" for c in top.index])
ax2.axvline(0, color=INK2, lw=1)
ax2.set(xlabel="share 2024 minus 2023 (points)")
ax2.set_title("Label shift: chapters", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "drift-histogram.png")

# ------------------------------------------------------------ figure 2: PSI explained, per country
cats = shares.index.tolist()
p = np.clip(shares["training 2017-2023"].to_numpy(), 1e-4, None)
q = np.clip(shares["2024"].to_numpy(), 1e-4, None)
contrib = pd.Series((q - p) * np.log(q / p), index=cats).sort_values(ascending=False)
show = contrib.index[:8]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), dpi=130, sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
x = np.arange(len(show))
ax1.bar(x - 0.2, shares.loc[show, "training 2017-2023"] * 100, 0.38, color=BLUE, label="reference p (2017-2023)")
ax1.bar(x + 0.2, shares.loc[show, "2024"] * 100, 0.38, color=ORANGE, label="current q (2024)")
ax1.set(ylabel="share (%)", yscale="log", ylim=(0.5, 100))
ax1.set_title(f"PSI of the issuing country: {contrib.sum():.3f}, of which GB {contrib['GB']:.3f}",
              loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
ax2.bar(x, contrib[show], 0.6, color=INK2)
ax2.set(ylabel="(q - p)·ln(q/p)", xlabel="issuing country (the eight largest contributions)")
ax2.set_xticks(x, show)
for a in (ax1, ax2):
    a.grid(axis="y", color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(OUT / "psi-explained.png")
print("country PSI", round(contrib.sum(), 3), "GB", round(contrib["GB"], 3), "| chapter changes", top.round(4).to_dict())
