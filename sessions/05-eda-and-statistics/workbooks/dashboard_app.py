"""One-page dashboard of Binding Tariff Information decisions (Session 5, block 3).

Start from the repository root:
    uv run streamlit run sessions/05-eda-and-statistics/workbooks/dashboard_app.py

The app opens at http://localhost:8501. Every widget change reruns this script from top to
bottom; the data are read once thanks to @st.cache_data.

Data: case-study/data/monthly_counts.parquet (decisions per month, issuing country and HS
chapter, by start of validity) and nomenclature.parquet (English chapter names).

Exercises
1. Add a selectbox for the section of the nomenclature and filter the chapters by it.
2. Add a chart of the share of the selected chapter among all decisions of each country.
3. Write the four-part finding (finding, number, meaning, limitation) in the text box at the end.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]          # repository root
DATA = ROOT / "case-study" / "data"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.Series]:
    """Read the monthly counts and the English chapter names once."""
    counts = pd.read_parquet(DATA / "monthly_counts.parquet")
    # a few start dates are typing errors far in the future (e.g. 2200): keep 2004-2026
    counts = counts[counts["month"].dt.year.between(2004, 2026)].copy()
    nomenclature = pd.read_parquet(DATA / "nomenclature.parquet")
    chapters = nomenclature.drop_duplicates("chapter").set_index("chapter")["chapter_description"]
    return counts, chapters


st.set_page_config(page_title="EBTI decisions", layout="wide")
counts, chapter_names = load()

st.title("Binding Tariff Information decisions per month")
st.caption(f"{counts['n_decisions'].sum():,} decisions, {counts['month'].min():%Y}–"
           f"{counts['month'].max():%Y}, by start of validity. Source: European Commission, "
           "EBTI database. The last year is incomplete.")

# ---- filters -----------------------------------------------------------------------------
years = st.sidebar.slider("Years", 2004, 2026, (2015, 2025))
largest = counts.groupby("issuing_country")["n_decisions"].sum().nlargest(12).index.tolist()
countries = st.sidebar.multiselect("Issuing countries", largest, default=["DE", "FR", "NL", "GB"])
chapter_options = ["all"] + sorted(counts["chapter"].dropna().unique().tolist())
chapter = st.sidebar.selectbox(
    "HS chapter", chapter_options,
    format_func=lambda c: c if c == "all" else f"{c} {str(chapter_names.get(c, ''))[:40]}")

sel = counts[counts["month"].dt.year.between(*years) & counts["issuing_country"].isin(countries)]
if chapter != "all":
    sel = sel[sel["chapter"] == chapter]

# ---- headline numbers --------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Decisions", f"{sel['n_decisions'].sum():,}")
c2.metric("Months", f"{sel['month'].nunique():,}")
per_country = sel.groupby("issuing_country")["n_decisions"].sum()
c3.metric("Largest issuing country", per_country.idxmax() if len(per_country) else "–")

# ---- charts ------------------------------------------------------------------------------
left, right = st.columns(2)
per_month = sel.groupby(["month", "issuing_country"], as_index=False)["n_decisions"].sum()
fig = px.line(per_month, x="month", y="n_decisions", color="issuing_country",
              color_discrete_sequence=OKABE_ITO,
              labels={"month": "", "n_decisions": "decisions per month", "issuing_country": "country"},
              title="Decisions per month by issuing country")
fig.update_yaxes(rangemode="tozero")
left.plotly_chart(fig, width="stretch")

top = (sel.groupby("chapter")["n_decisions"].sum().nlargest(10).rename("decisions").reset_index())
top["name"] = top["chapter"] + " " + top["chapter"].map(chapter_names).fillna("").str[:35]
fig = px.bar(top, x="decisions", y="name", orientation="h",
             labels={"name": "", "decisions": "decisions"},
             title="Ten largest chapters in the selection")
fig.update_traces(marker_color="#0072B2")
fig.update_yaxes(autorange="reversed")
right.plotly_chart(fig, width="stretch")

# ---- written finding ---------------------------------------------------------------------
st.subheader("Finding for a customs analyst")
st.text_area("Finding, number, meaning, limitation",
             "The United Kingdom issued about 3,000 decisions a year until 2020 and none after Brexit; "
             "English-language decisions almost disappeared. A heading classifier trained on older "
             "years sees more English text than it will meet in 2024. Limitation: counts by start "
             "of validity; the last year is incomplete.")
