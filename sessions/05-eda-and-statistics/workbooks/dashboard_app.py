"""One-page dashboard of short-stay Airbnb prices in Berlin (Session 5, block 3).

Start from the repository root:
    uv run streamlit run sessions/05-eda-and-statistics/workbooks/dashboard_app.py

The app opens at http://localhost:8501. Every widget change reruns this script from top to
bottom; the data are read once thanks to @st.cache_data.

Data: case-study/data/airbnb/listings.parquet (Inside Airbnb, Berlin snapshot of 26 June 2026,
CC BY 4.0; prepare with `uv run python case-study/prepare_airbnb.py`). Only listings with a price
and a minimum stay below 28 nights are shown: the price field of medium-term listings is not
comparable (Session 4). The data come from public listing pages; the app shows aggregates only.

Exercises
1. Add a selectbox for the district and show the median price per neighbourhood of that district,
   with the number of listings next to each bar (hide neighbourhoods with fewer than 20 listings).
2. Add a chart of the share of entire homes per district for the current filter.
3. Write the four-part finding (finding, number, meaning, limitation) in the text box at the end.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]          # repository root
DATA = ROOT / "case-study" / "data" / "airbnb"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


@st.cache_data
def load() -> pd.DataFrame:
    """Read the listings once and keep the comparable short-stay prices."""
    listings = pd.read_parquet(DATA / "listings.parquet")
    return listings[listings["price"].notna() & listings["minimum_nights"].lt(28)].copy()


st.set_page_config(page_title="Berlin short-stay prices", layout="wide")
short = load()

st.title("Short-stay listings in Berlin: prices by district")
st.caption(f"{len(short):,} listings with a price and a minimum stay below 28 nights. Inside Airbnb, "
           f"snapshot of {short['last_scraped'].min():%d %B %Y} (CC BY 4.0). Listed prices per night, "
           "not paid prices.")

# ---- filters -----------------------------------------------------------------------------
room_types = st.sidebar.multiselect("Room types", sorted(short["room_type"].unique()),
                                    default=["Entire home/apt"])
guests = st.sidebar.slider("Guests (accommodates)", 1, 16, (1, 4))
sel = short[short["room_type"].isin(room_types) & short["accommodates"].between(*guests)]

# ---- headline numbers --------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Listings", f"{len(sel):,}")
c2.metric("Median price per night", f"€{sel['price'].median():,.0f}" if len(sel) else "–")
c3.metric("Districts", f"{sel['district'].nunique()}")

# ---- charts ------------------------------------------------------------------------------
left, right = st.columns(2)
by_district = (sel.groupby("district")["price"].agg(median="median", listings="size")
               .reset_index().sort_values("median"))
fig = px.bar(by_district, x="median", y="district", orientation="h", text="listings",
             labels={"median": "median price per night (EUR)", "district": "", "listings": "listings"},
             title="Median price by district (bar label: number of listings)")
fig.update_traces(marker_color=OKABE_ITO[0], textposition="outside")
left.plotly_chart(fig, width="stretch")

fig = px.scatter(sel, x="longitude", y="latitude", color="price", range_color=(50, 400),
                 color_continuous_scale="viridis", opacity=0.6,
                 labels={"price": "EUR per night"}, title="Where the listings are (colour: price)")
fig.update_traces(marker_size=4)
fig.update_yaxes(scaleanchor="x", scaleratio=1.6)   # roughly true proportions at Berlin's latitude
right.plotly_chart(fig, width="stretch")

# ---- written finding ---------------------------------------------------------------------
st.subheader("Finding for a city housing analyst")
st.text_area("Finding, number, meaning, limitation",
             "A night in Mitte costs about a fifth more than a comparable night in Neukölln: median "
             "short-stay price 187 against 130 EUR, about 20 % at equal room type and number of guests. "
             "The raw gap overstates the location premium because Neukölln offers more private rooms. "
             "Limitation: listed prices from one snapshot; medium-term listings excluded.")
