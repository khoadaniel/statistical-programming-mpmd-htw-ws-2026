"""One-page review dashboard for a product manager (Session 5, block 3).

Start from the repository root:
    uv run streamlit run sessions/05-eda-and-statistics/workbooks/dashboard_app.py

The app opens at http://localhost:8501. Every widget change reruns this script from top to
bottom; the data are read once thanks to @st.cache_data.

Exercises
1. Add a filter for the sentiment label (st.sidebar.multiselect).
2. Add a chart of the median number of words per year.
3. Write the four-part finding (finding, number, meaning, limitation) in the text box at the end.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]          # repository root
DATA = ROOT / "case-study" / "data" / "train_sample.parquet"
COLOURS = {"neg": "#D55E00", "neu": "#999999", "pos": "#0072B2"}   # Okabe-Ito, colour-blind safe


@st.cache_data
def load() -> pd.DataFrame:
    """Read the 50,000-review sample once and add derived columns."""
    df = pd.read_parquet(DATA)
    df["year"] = df["date"].dt.year
    df["n_words"] = df["text"].str.split().str.len()
    return df


st.set_page_config(page_title="Review dashboard", layout="wide")
reviews = load()

st.title("Amazon health product reviews")
st.caption(f"Sample of {len(reviews):,} reviews, {reviews['date'].min():%Y}–"
           f"{reviews['date'].max():%Y}. Source: Amazon Reviews 2023 (McAuley Lab).")

# ---- filters -----------------------------------------------------------------------------
years = st.sidebar.slider("Years", 2010, 2021, (2017, 2021))
verified_only = st.sidebar.checkbox("Verified purchases only", value=True)

sel = reviews[reviews["year"].between(*years)]
if verified_only:
    sel = sel[sel["verified_purchase"]]

# ---- headline numbers --------------------------------------------------------------------
rho = stats.spearmanr(sel["n_words"], sel["helpful_vote"]).statistic if len(sel) > 2 else np.nan
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(sel):,}")
c2.metric("Share of 1–2 stars", f"{(sel['rating'] <= 2).mean():.1%}")
c3.metric("Median words per review", f"{sel['n_words'].median():.0f}")
c4.metric("Length vs helpful votes (Spearman)", f"{rho:.2f}")

# ---- charts ------------------------------------------------------------------------------
left, right = st.columns(2)
ratings = sel["rating"].value_counts(normalize=True).sort_index().rename("share").reset_index()
fig = px.bar(ratings, x="rating", y="share", text_auto=".0%",
             labels={"rating": "stars", "share": "share of reviews"},
             title="Rating distribution")
fig.update_traces(marker_color="#555555")
left.plotly_chart(fig, width="stretch")

yearly = (sel.groupby("year")["label"].value_counts(normalize=True)
          .rename("share").reset_index())
fig = px.line(yearly, x="year", y="share", color="label", markers=True,
              color_discrete_map=COLOURS, category_orders={"label": ["neg", "neu", "pos"]},
              labels={"share": "share of reviews", "year": ""},
              title="Sentiment label per year")
fig.update_yaxes(tickformat=".0%", rangemode="tozero")
right.plotly_chart(fig, width="stretch")

monthly = sel.set_index("date").resample("MS").size().rename("reviews").reset_index()
fig = px.line(monthly, x="date", y="reviews", labels={"date": "", "reviews": "reviews per month"},
              title="Number of reviews per month")
left.plotly_chart(fig, width="stretch")

bins = pd.cut(sel["n_words"], [-1, 5, 10, 20, 40, 60, 100, 5000],
              labels=["0–5", "6–10", "11–20", "21–40", "41–60", "61–100", ">100"])
votes = sel.groupby(bins, observed=True)["helpful_vote"].mean().rename("votes").reset_index()
fig = px.bar(votes, x="n_words", y="votes", text_auto=".1f",
             labels={"n_words": "words per review", "votes": "mean helpful votes"},
             title="Longer reviews receive more helpful votes")
fig.update_traces(marker_color="#0072B2")
right.plotly_chart(fig, width="stretch")

# ---- written finding ---------------------------------------------------------------------
st.subheader("Finding for the product manager")
st.text_area("Finding, number, meaning, limitation",
             "Longer reviews receive more helpful votes (Spearman ρ ≈ 0.3). "
             "Prompting reviewers for detail may make reviews more useful. "
             "Limitation: observational data; older reviews had more time to collect votes.")
