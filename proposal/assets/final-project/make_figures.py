"""Figures for the appendix of CURRICULUM.md (the final project).

Needs case-study/data/ebti/decisions.parquet (ebti-v1.zip from the course drive, see
case-study/README.md):

    uv run python proposal/assets/final-project/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "case-study" / "data" / "ebti"
OUT = Path(__file__).resolve().parent
BLUE, TEAL, PURPLE, GREY, RED = "#3b5b8c", "#2d7d74", "#6a5a8c", "#9aa3ae", "#8c4a5e"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


decisions = pd.read_parquet(DATA / "decisions.parquet", columns=["heading", "language", "start_date"])

# 1. the decisions students receive, by start year (the held-back evaluation set is not in their data)
by_year = pd.to_datetime(decisions["start_date"]).dt.year.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(by_year.index, by_year / 1000, color=TEAL, width=0.6)
for year, n in by_year.items():
    ax.text(year, n / 1000 + 0.6, f"{n:,.0f}", ha="center", fontsize=10)
ax.set_xticks(by_year.index, [f"{y}" + (" (from Oct)" if y == by_year.index.min() else
                                         " (to Oct)" if y == by_year.index.max() else "") for y in by_year.index])
ax.set_ylabel("decisions (thousands)")
ax.set_ylim(0, by_year.max() / 1000 * 1.15)
ax.set_title(f"The {len(decisions):,} decisions students receive, by start year")
save(fig, "decisions-per-year.png")

# 2. languages
share = decisions["language"].value_counts(normalize=True)
NAMES = {"de": "German", "fr": "French", "en": "English", "nl": "Dutch", "pl": "Polish", "cs": "Czech",
         "es": "Spanish", "sv": "Swedish", "it": "Italian"}
share.index = [NAMES.get(c, c) for c in share.index]
top = pd.concat([share.head(8), pd.Series({f"{len(share) - 8} others": share.iloc[8:].sum()})])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(top.index[::-1], top.values[::-1] * 100, color=[GREY] + [BLUE] * 8)
for i, v in enumerate(top.values[::-1] * 100):
    ax.text(v + 0.8, i, f"{v:.0f} %" if v >= 1 else f"{v:.1f} %", va="center", fontsize=10)
ax.set_xlabel("share of decisions (%)")
ax.set_title(f"Descriptions are written in {len(share)} languages")
save(fig, "languages.png")

# 3. the long tail of headings
counts = decisions["heading"].value_counts()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(range(1, len(counts) + 1), counts.values, width=1.0, color=BLUE)
ax.set_yscale("log")
ax.set_xlabel("headings, from the most to the least frequent")
ax.set_ylabel("decisions (log scale)")
ax.set_title(f"{len(counts):,} headings: a few frequent ones and a long tail")
rare = (counts < 10).sum()
ax.axhline(10, color=RED, lw=1, ls="--")
ax.annotate(f"{rare} headings have\nfewer than 10 decisions", xy=(len(counts) - rare / 2, 10), xytext=(560, 400),
            color=RED, fontsize=10, arrowprops={"arrowstyle": "->", "color": RED})
ax.text(30, counts.iloc[0] * 0.75, f"most frequent: {counts.index[0]}, other articles of plastics "
        f"({counts.iloc[0] / len(decisions):.0%})", fontsize=10)
save(fig, "long-tail.png")

# 4. reference accuracy from one possible validation design (values from case-study/README.md:
#    one fold of a stratified 7-fold split by heading and language, 87,805 decisions for training,
#    14,635 for validation),
#    followed by approaches that teams may try and whose results are open
ladder = pd.Series({
    "always the most frequent heading": 0.042,
    "word TF-IDF, sample of 50,000 decisions": 0.856,
    "word TF-IDF, all 87,805 training decisions": 0.877,
    "character TF-IDF, all 87,805 training decisions": 0.885,
})
open_ideas = [
    "fine-tuned multilingual encoder (BERT, XLM-R)",
    "TF-IDF combined with embeddings",
    "embeddings + nearest past decisions",
    "chapter first, then heading (hierarchical)",
    "language model choosing among candidates",
    "retrieval of similar decisions (RAG)",
    "ensemble of several models",
    "other designs",
]
labels = list(ladder.index) + open_ideas
fig, ax = plt.subplots(figsize=(8.5, 6.2))
ys = list(range(len(labels)))[::-1]
known_colors = [GREY, BLUE, TEAL, TEAL]
for y, v, c in zip(ys[:len(ladder)], ladder.values * 100, known_colors):
    ax.barh(y, v, color=c)
    ax.text(v + 1, y, f"{v:.1f} %", va="center", fontsize=10)
for y in ys[len(ladder):]:
    ax.barh(y, 100, color="none", edgecolor=PURPLE, hatch="///", linewidth=0, alpha=0.25)
    ax.barh(y, 100, color="none", edgecolor=PURPLE, linestyle="--", linewidth=1)
    ax.text(50, y, "?", va="center", ha="center", fontsize=13, fontweight="bold", color=PURPLE)
sep = ys[len(ladder)] + 0.5
ax.axhline(sep, color="black", lw=0.6)
ax.text(101, ys[0], "measured", va="center", fontsize=9, color="black")
ax.text(101, ys[len(ladder)], "open for\nthe teams", va="top", fontsize=9, color=PURPLE)
ax.set_yticks(ys, labels)
for tick in ax.get_yticklabels()[len(ladder):]:
    tick.set_color(PURPLE)
ax.set_xlim(0, 100)
ax.set_xlabel("accuracy on 14,635 validation decisions (one possible split)")
ax.set_title("Reference results, and approaches the teams may try")
save(fig, "reference-accuracy.png")

print("figures written to", OUT)
