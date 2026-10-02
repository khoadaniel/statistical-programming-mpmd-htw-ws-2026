# %% [markdown]
# # Step 0: the analysis as it looks in a notebook
#
# This file is the *starting point* of the workspace: notebook-style code in cells
# (`# %%` markers, which VS Code and JupyterLab with Jupytext run cell by cell).
# Everything lives in global variables, the path and the 28-night rule are hard-coded,
# and nothing can be imported or tested. The package in `src/listingtools/` is the same
# analysis turned into a module with a class. Compare the two files side by side.
#
# Run from the repository root:
#     uv run python sessions/01-careers-and-python/workspace/notebooks/explore_listings.py

# %%
import pandas as pd

df = pd.read_parquet("case-study/data/airbnb/listings.parquet")
print(df.shape, df["host_id"].nunique(), "hosts")

# %%
# Question 2: where are the listings?
print(df["district"].value_counts().head(5))

# %%
# Question 4: what does a night cost? Only short stays with a price are comparable.
short = df[df["price"].notna() & (df["minimum_nights"] < 28)]
print(short.groupby("room_type")["price"].median().round(1))

# %%
# The same lines, copied into the next notebook with a small change (say, < 30 instead
# of < 28), are how diverging versions of "the same" analysis appear. The module keeps
# the rule in one place: SHORT_STAY_MAX_NIGHTS.
