"""Figures for the Session 1 lecture notebooks.

Run from the repository root. The listing and language figures need the case-study data: download `airbnb.zip` and
`ebti.zip` from the course drive and unzip them into `case-study/data/` (see case-study/README.md). The skills figure
reads job_ad_skills.csv, written by job_ad_skills.py:

    uv run python sessions/01-careers-and-python/figures/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")  # write files only, no window

OUT = Path(__file__).resolve().parent
DATA = Path("case-study/data/airbnb")
EBTI = Path("case-study/data/ebti")
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
    left.set_xlabel(f"Listings (all {len(listings):,})")
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


# Colours of the role figures: one blue ramp (light = little, dark = much), grey ink for text.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

STAGES = ["Collect and\nstore data", "Clean and\nmodel data", "Analyse and\nreport", "Build and\nevaluate models",
          "Deploy and\noperate", "Build AI\napplications"]
# 2 = core of the role, 1 = often part of it, 0 = rarely; the course team's simplified reading of typical ads
ROLE_MAP = {
    "Data analyst": [0, 1, 2, 1, 0, 0],
    "Data scientist": [0, 1, 1, 2, 1, 0],
    "Data engineer": [2, 2, 0, 0, 1, 0],
    "ML engineer": [0, 1, 0, 1, 2, 1],
    "AI engineer": [0, 0, 0, 1, 1, 2],
}


def role_map() -> None:
    """Where each role works along the path from raw data to a running product."""
    from matplotlib.patches import FancyBboxPatch

    roles = list(ROLE_MAP)
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=130)
    fill = {2: BLUE[4], 1: BLUE[1]}
    for y, role in enumerate(roles):
        for x, level in enumerate(ROLE_MAP[role]):
            if level:
                ax.add_patch(FancyBboxPatch((x + 0.06, y + 0.12), 0.88, 0.76, boxstyle="round,pad=0,rounding_size=0.08",
                                            facecolor=fill[level], edgecolor="none"))
            else:
                ax.plot(x + 0.5, y + 0.5, "o", ms=3, color=GRID)
    ax.set_xlim(0, len(STAGES))
    ax.set_ylim(len(roles), -1.1)
    ax.set_xticks([x + 0.5 for x in range(len(STAGES))], STAGES, fontsize=8.5, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks([y + 0.5 for y in range(len(roles))], roles, fontsize=10, color=INK)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.annotate("", xy=(len(STAGES) - 0.05, -0.55), xytext=(0.05, -0.55),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1}, annotation_clip=False)
    ax.text(0.05, -0.68, "from raw data ...", fontsize=8, color=MUTED, va="bottom")
    ax.text(len(STAGES) - 0.15, -0.68, "... to a product people use", fontsize=8, color=MUTED, va="bottom", ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE[4]), plt.Rectangle((0, 0), 1, 1, color=BLUE[1]),
               plt.Line2D([], [], marker="o", ls="", ms=3, color=GRID)]
    ax.legend(handles, ["core of the role", "often part of it", "rarely"], loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("Five data roles along the path from data to product", x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.005, "Simplified view by the course team. Real jobs mix these columns; see the skills heat map from Berlin job ads.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "role_map.png")
    plt.close(fig)


def skills_by_role() -> None:
    """Share of Berlin job ads per role that mention each skill group (from job_ad_skills.csv)."""
    from matplotlib.colors import LinearSegmentedColormap

    table = pd.read_csv(OUT / "job_ad_skills.csv")
    roles = list(dict.fromkeys(table["role"]))
    skills = list(dict.fromkeys(table["skill"]))
    shares = table.pivot(index="role", columns="skill", values="share").loc[roles, skills].to_numpy()
    n_ads = table.groupby("role")["n_ads"].first().loc[roles]
    retrieved = pd.to_datetime(table["retrieved"].iloc[0]).strftime("%-d %B %Y")

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", *BLUE])
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=130)
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for y in range(len(roles)):
        for x in range(len(skills)):
            value = shares[y, x]
            ax.text(x, y, f"{value:.0%}", ha="center", va="center", fontsize=8.5, color="white" if value >= 0.55 else INK)
    import textwrap

    ax.set_xticks(range(len(skills)), [textwrap.fill(s, 13, break_long_words=False) for s in skills], fontsize=8.5, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(roles)), [f"{r}  (n = {n})" for r, n in n_ads.items()], fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(-0.5, len(skills)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(roles)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    fig.suptitle("What Berlin job ads ask for: share of ads per role that name each skill group", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.02, 0.01, f"Data Berlin job board, ads open on {retrieved}, matched to a role by job title (no leads or managers); "
             "a skill counts if a keyword appears in the ad. See job_ad_skills.py.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "skills_by_role.png")
    plt.close(fig)


LANGUAGE_NAMES = {"de": "German", "fr": "French", "nl": "Dutch", "cs": "Czech", "es": "Spanish", "pl": "Polish",
                  "it": "Italian", "sv": "Swedish", "en": "English"}


def ebti_languages() -> None:
    """Share of training decisions per language of the description (EBTI, final project)."""
    language = pd.read_parquet(EBTI / "train.parquet", columns=["language"])["language"]
    shares = language.value_counts(normalize=True) * 100
    top = shares.iloc[:8]
    others = shares.iloc[8:]
    labels = [LANGUAGE_NAMES.get(code, code) for code in top.index] + [f"{len(others)} others"]
    values = [*top.to_numpy(), others.sum()]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=130)
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color=[BLUE[5]] * len(top) + [MUTED], height=0.75)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.8, yi, f"{value:.0f} %" if value >= 1 else f"{value:.1f} %", va="center", fontsize=9,
                color=INK)
    ax.set_yticks(y, labels, fontsize=10, color=INK)
    ax.set_xlabel(f"share of the {len(language):,} training decisions (%)", color=INK_2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Descriptions are written in {language.nunique()} languages", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ebti_languages.png")
    plt.close(fig)


if __name__ == "__main__":
    role_map()
    skills_by_role()
    if DATA.exists():
        price_by_room_type()
        listings_by_district()
    if EBTI.exists():
        ebti_languages()
