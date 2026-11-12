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


# ---------------------------------------------------------------- ER diagrams of the star-vs-snowflake playground
FACT, DIM, SPLIT, REPEAT = "#eb6834", "#2a78d6", "#1baf7a", "#fde3d8"


def entity(ax, x, y, name, columns, head=DIM, w=3.1, rh=0.36):
    """An ER box with its top-left corner at (x, y): a coloured title bar, then one row per column.
    columns: (key, name, note), key "PK", "FK", "PK, FK" or ""; a note "repeated" fills the row in light orange.
    Returns the y of each column's row centre, and the box's left and right edge."""
    ax.add_patch(plt.Rectangle((x, y - rh), w, rh, facecolor=head, edgecolor=head, lw=1))
    ax.text(x + w / 2, y - rh / 2, name, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    rows = {}
    for i, (key, col, note) in enumerate(columns):
        top = y - (i + 1) * rh
        face = REPEAT if note == "repeated" else "white"
        ax.add_patch(plt.Rectangle((x, top - rh), w, rh, facecolor=face, edgecolor=MUTED, lw=0.8))
        ax.text(x + 0.1, top - rh / 2, key, ha="left", va="center", fontsize=8, fontweight="bold", color=MUTED)
        ax.text(x + 0.85, top - rh / 2, col, ha="left", va="center", fontsize=9.5, family="monospace", color=INK)
        rows[col] = top - rh / 2
    ax.add_patch(plt.Rectangle((x, y - (len(columns) + 1) * rh), w, (len(columns) + 1) * rh, fill=False, edgecolor=head, lw=1.6))
    return {"rows": rows, "left": x, "right": x + w}


def relation(ax, one, many, y_one, y_many, label=""):
    """A one-to-many line in crow's-foot notation from the box `one` (two bars) to the box `many` (crow's foot).
    The boxes stand side by side; the line runs between their facing edges."""
    if one["right"] < many["left"]:
        x0, x1, d = one["right"], many["left"], 1
    else:
        x0, x1, d = one["left"], many["right"], -1
    xm = (x0 + x1) / 2
    ax.plot([x0, xm, xm, x1], [y_one, y_one, y_many, y_many], color=INK, lw=1.2, solid_capstyle="butt")
    for k in (0.12, 0.2):                                                           # "exactly one": two bars
        ax.plot([x0 + d * k] * 2, [y_one - 0.1, y_one + 0.1], color=INK, lw=1.2)
    for dy in (-0.13, 0, 0.13):                                                      # "many": the crow's foot
        ax.plot([x1 - d * 0.22, x1], [y_many, y_many + dy], color=INK, lw=1.2)
    ax.text(x0 + d * 0.3, y_one - 0.08, "1", ha="center", va="top", fontsize=9, color=MUTED)
    ax.text(x1 - d * 0.32, y_many - 0.08, "N", ha="center", va="top", fontsize=9, color=MUTED)
    if label:
        ax.text(xm, max(y_one, y_many) + 0.16, label, ha="center", va="bottom", fontsize=8.5, style="italic", color=MUTED)


STUDENT_COLS = [("PK", "student_id", ""), ("", "name", "")]
SCORE_COLS = [("PK, FK", "student_id", ""), ("PK, FK", "course_id", ""), ("", "exam_date", ""), ("", "score", "")]


def draw_star(ax, y=0.0, x=0.0):
    students = entity(ax, x + 0.0, y, "star_students", STUDENT_COLS + [("", "programme", "repeated"), ("", "faculty", "repeated"), ("", "start_year", "")])
    scores = entity(ax, x + 4.8, y, "star_scores  (fact)", SCORE_COLS, head=FACT)
    courses = entity(ax, x + 9.6, y, "star_courses", [("PK", "course_id", ""), ("", "title", ""), ("", "credits", ""), ("", "department", "repeated")])
    relation(ax, students, scores, students["rows"]["student_id"], scores["rows"]["student_id"], "takes exams")
    relation(ax, courses, scores, courses["rows"]["course_id"], scores["rows"]["course_id"], "is examined in")
    return 12.7


def draw_snowflake(ax, y=0.0):
    faculties = entity(ax, 0.0, y, "snowflake_faculties", [("PK", "faculty_id", ""), ("", "name", "")], head=SPLIT)
    programmes = entity(ax, 4.6, y, "snowflake_programmes", [("PK", "programme_id", ""), ("", "name", ""), ("FK", "faculty_id", "")], head=SPLIT)
    students = entity(ax, 9.2, y, "snowflake_students", STUDENT_COLS + [("FK", "programme_id", ""), ("", "start_year", "")])
    scores = entity(ax, 13.8, y, "snowflake_scores  (fact)", SCORE_COLS, head=FACT)
    courses = entity(ax, 18.4, y, "snowflake_courses", [("PK", "course_id", ""), ("", "title", ""), ("", "credits", ""), ("FK", "department_id", "")])
    departments = entity(ax, 23.0, y, "snowflake_departments", [("PK", "department_id", ""), ("", "name", "")], head=SPLIT)
    relation(ax, faculties, programmes, faculties["rows"]["faculty_id"], programmes["rows"]["faculty_id"], "has")
    relation(ax, programmes, students, programmes["rows"]["programme_id"], students["rows"]["programme_id"], "enrols")
    relation(ax, students, scores, students["rows"]["student_id"], scores["rows"]["student_id"], "takes exams")
    relation(ax, courses, scores, courses["rows"]["course_id"], scores["rows"]["course_id"], "is examined in")
    relation(ax, departments, courses, departments["rows"]["department_id"], courses["rows"]["department_id"], "teaches")
    return 26.1


def legend(ax, x, y, items):
    for i, (face, edge, text) in enumerate(items):
        ax.add_patch(plt.Rectangle((x + i * 4.2, y - 0.14), 0.4, 0.28, facecolor=face, edgecolor=edge, lw=1.2))
        ax.text(x + i * 4.2 + 0.55, y, text, va="center", fontsize=9, color=INK)


def er_star():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    width = draw_star(ax)
    legend(ax, 0.0, -2.75, [(FACT, FACT, "fact table"), (DIM, DIM, "dimension"), (REPEAT, MUTED, "text repeated in every row")])
    ax.set_xlim(-0.2, width + 0.2); ax.set_ylim(-3.0, 0.5); ax.axis("off")
    save(fig, "er-star.png")


def er_snowflake():
    fig, ax = plt.subplots(figsize=(24, 3.0))
    width = draw_snowflake(ax)
    legend(ax, 0.0, -2.0, [(FACT, FACT, "fact table"), (DIM, DIM, "dimension"), (SPLIT, SPLIT, "level split out of a dimension")])
    ax.set_xlim(-0.2, width + 0.2); ax.set_ylim(-2.3, 0.5); ax.axis("off")
    save(fig, "er-snowflake.png")


def er_star_vs_snowflake():
    """Both schemas on one figure, the fact tables aligned, so that the split dimensions stand out."""
    fig, ax = plt.subplots(figsize=(24, 6.6))
    ax.text(-0.1, 0.75, "Star schema: 3 tables, the names of programme, faculty and department repeated in the dimensions",
            fontsize=12, fontweight="bold", color=INK)
    draw_star(ax, y=0.0, x=9.0)                                    # star_scores (x 13.8) right above snowflake_scores
    ax.text(-0.1, -3.25, "Snowflake schema: 6 tables, each name stored once in its own table; more joins from the fact table",
            fontsize=12, fontweight="bold", color=INK)
    draw_snowflake(ax, y=-3.75)
    legend(ax, 0.0, -6.55, [(FACT, FACT, "fact table"), (DIM, DIM, "dimension"), (REPEAT, MUTED, "repeated text"),
                            (SPLIT, SPLIT, "level split out")])
    ax.set_xlim(-0.2, 26.3); ax.set_ylim(-6.85, 1.2); ax.axis("off")
    save(fig, "er-star-vs-snowflake.png")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["er"]:          # only the ER diagrams; they need no data
        er_star(); er_snowflake(); er_star_vs_snowflake()
    else:
        join_types()
        reviews_per_year()
        er_star(); er_snowflake(); er_star_vs_snowflake()
