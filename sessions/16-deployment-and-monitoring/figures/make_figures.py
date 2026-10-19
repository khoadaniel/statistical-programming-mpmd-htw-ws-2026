"""Figures for the Session 16 lecture notebooks. Run from the repository root:

    uv run python sessions/16-deployment-and-monitoring/figures/make_figures.py

Uses the EBTI case-study data (case-study/data/ebti/), with the validation design and the simulated release
date of block notebook 02: the training decisions (decisions.parquet without the validation fold)
that started before 2026 are the reference, the training decisions of 2026 the current data (the requests
after a simulated release at the start of 2026) and their labels.
Deterministic.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data/ebti")
RELEASE = pd.Timestamp("2026-01-01")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})

decisions = pd.read_parquet(DATA / "decisions.parquet", columns=["issuing_country", "language", "start_date", "heading", "chapter"])
key = decisions["heading"] + "|" + decisions["language"]
train_rows, valid_rows = next(StratifiedKFold(n_splits=7, shuffle=True, random_state=0).split(decisions, key))
train, val = decisions.iloc[train_rows], decisions.iloc[valid_rows]
old = train[train["start_date"] < RELEASE]
new = train[train["start_date"] >= RELEASE]
labels_2026 = new["heading"]

# ------------------------------------------------------------ figure 1: country drift + chapter label shift
countries = ["DE", "FR", "NL", "CZ", "ES", "PL", "IT"]
shares = pd.DataFrame({"training 2023-2025": old["issuing_country"].value_counts(normalize=True),
                       "requests 2026": new["issuing_country"].value_counts(normalize=True)}).fillna(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=130, gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(countries))
for k, (col, color) in enumerate(zip(shares.columns, [BLUE, ORANGE])):
    ax1.bar(x + (k - 0.5) * 0.38, shares.loc[countries, col] * 100, 0.37, color=color, label=col)
ax1.set_xticks(x, countries)
ax1.set(ylabel="share of decisions (%)", yscale="log", ylim=(0.5, 100))
fr = shares.loc["FR"] * 100
ax1.annotate(f"FR: {fr.iloc[0]:.0f} % -> {fr.iloc[1]:.0f} %", (1.2, fr.iloc[1]), xytext=(2.2, 40), fontsize=9, color=INK2,
             arrowprops={"arrowstyle": "->", "color": INK2})
ax1.set_title("Input drift: issuing countries", loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9)
ax1.grid(axis="y", color=GRID, lw=0.8)

y2025 = old[old["start_date"].dt.year == 2025]["chapter"].value_counts(normalize=True)
y2026 = labels_2026.str[:2].value_counts(normalize=True)
idx = y2025.index.union(y2026.index)
change = y2026.reindex(idx, fill_value=0) - y2025.reindex(idx, fill_value=0)
top = change.reindex(change.abs().sort_values(ascending=False).index[:6])
ax2.barh(range(len(top))[::-1], top.to_numpy() * 100, color=[ORANGE if v > 0 else BLUE for v in top])
ax2.set_yticks(range(len(top))[::-1], [f"chapter {c}" for c in top.index])
ax2.axvline(0, color=INK2, lw=1)
ax2.set(xlabel="share 2026 minus 2025 (points)")
ax2.set_title("Label shift: chapters", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "drift-histogram.png")

# ------------------------------------------------------------ figure 2: PSI explained, per country
cats = shares.index.tolist()
p = np.clip(shares["training 2023-2025"].to_numpy(), 1e-4, None)
q = np.clip(shares["requests 2026"].to_numpy(), 1e-4, None)
contrib = pd.Series((q - p) * np.log(q / p), index=cats).sort_values(ascending=False)
show = contrib.index[:8]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), dpi=130, sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
x = np.arange(len(show))
ax1.bar(x - 0.2, shares.loc[show, "training 2023-2025"] * 100, 0.38, color=BLUE, label="reference p (2023-2025)")
ax1.bar(x + 0.2, shares.loc[show, "requests 2026"] * 100, 0.38, color=ORANGE, label="current q (2026)")
ax1.set(ylabel="share (%)", yscale="log", ylim=(0.01, 100))
ax1.set_title(f"PSI of the issuing country: {contrib.sum():.3f}, of which FR {contrib['FR']:.3f}",
              loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
ax2.bar(x, contrib[show], 0.6, color=INK2)
ax2.set(ylabel="(q - p)·ln(q/p)", xlabel="issuing country (the eight largest contributions)")
ax2.set_xticks(x, show)
for a in (ax1, ax2):
    a.grid(axis="y", color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(OUT / "psi-explained.png")
print("country PSI", round(contrib.sum(), 3), "FR", round(contrib["FR"], 3), "| chapter changes", top.round(4).to_dict())
