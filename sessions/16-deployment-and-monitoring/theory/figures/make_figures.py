"""Figures for the Session 16 theory pages. Run from the repository root:

    uv run --with pandas --with pyarrow --with matplotlib --with numpy \
        python sessions/16-deployment-and-monitoring/theory/figures/make_figures.py

Uses the case-study data (training sample and test inputs; the label shares of 2022 are the published
values of case-study/README.md, so no hidden labels are needed). Deterministic.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})


def text_of(df):
    return np.where(df["title"].fillna("") != "", df["title"].fillna("") + ". " + df["text"].fillna(""),
                    df["text"].fillna(""))


train = pd.read_parquet(DATA / "train_sample.parquet", columns=["title", "text", "label"])
test = pd.read_parquet(DATA / "test.parquet", columns=["title", "text", "date"])
ref = np.log1p(pd.Series(text_of(train)).str.len().to_numpy())
cur = np.log1p(pd.Series(text_of(test[test["date"].dt.year == 2022])).str.len().to_numpy())

# ------------------------------------------------------------ figure 1: drift histogram + label shares
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=140, gridspec_kw={"width_ratios": [2.2, 1]})
bins = np.linspace(1, 9, 41)
ax1.hist(ref, bins=bins, density=True, histtype="step", lw=2, color=BLUE, label="training data (to 2021)")
ax1.hist(cur, bins=bins, density=True, histtype="step", lw=2, color=ORANGE, label="2022 requests")
ticks = [10, 30, 100, 300, 1000, 3000]
ax1.set_xticks(np.log1p(ticks), [str(t) for t in ticks])
ax1.set(xlabel="review length in characters (log scale)", ylabel="density")
ax1.set_title("Input drift: reviews got longer", loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9)
ax1.grid(axis="y", color=GRID, lw=0.8)

shares_ref = train["label"].value_counts(normalize=True).reindex(["neg", "neu", "pos"]).to_numpy()
shares_cur = np.array([0.265, 0.079, 0.656])        # 2022, published in case-study/README.md
x = np.arange(3)
ax2.bar(x - 0.2, shares_ref * 100, 0.38, color=BLUE, label="to 2021")
ax2.bar(x + 0.2, shares_cur * 100, 0.38, color=ORANGE, label="2022")
for i in range(3):
    ax2.text(i - 0.2, shares_ref[i] * 100 + 1, f"{shares_ref[i]:.0%}", ha="center", fontsize=9, color=INK2)
    ax2.text(i + 0.2, shares_cur[i] * 100 + 1, f"{shares_cur[i]:.0%}", ha="center", fontsize=9, color=INK2)
ax2.set_xticks(x, ["neg", "neu", "pos"])
ax2.set(ylabel="share of reviews (%)", ylim=(0, 85))
ax2.set_title("Label shift", loc="left", fontsize=12)
ax2.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "drift-histogram.png")

# ------------------------------------------------------------ figure 2: PSI explained
edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11))[1:-1])
p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=10) / len(ref)
q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=10) / len(cur)
contrib = (q - p) * np.log(q / p)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), dpi=140, sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
x = np.arange(10)
ax1.bar(x - 0.2, p * 100, 0.38, color=BLUE, label="reference p (to 2021)")
ax1.bar(x + 0.2, q * 100, 0.38, color=ORANGE, label="current q (2022)")
ax1.axhline(10, color=INK2, lw=1, ls=":")
ax1.set(ylabel="share in bin (%)")
ax1.set_title(f"PSI: compare shares per reference decile of review length (PSI = {contrib.sum():.3f})",
              loc="left", fontsize=12)
ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
ax2.bar(x, contrib, 0.6, color=INK2)
ax2.set(ylabel="(q − p)·ln(q/p)", xlabel="bin = decile of the reference (1 = shortest reviews)")
ax2.set_xticks(x, [str(i + 1) for i in x])
for a in (ax1, ax2):
    a.grid(axis="y", color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(OUT / "psi-explained.png")
print("PSI", round(contrib.sum(), 3), "p", p.round(3).tolist(), "q", q.round(3).tolist())
