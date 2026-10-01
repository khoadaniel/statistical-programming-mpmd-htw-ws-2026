# %% [markdown]
# # Step 0: the analysis as it looks in a notebook
#
# This file is the *starting point* of the workspace: notebook-style code in cells
# (`# %%` markers, which VS Code and JupyterLab with Jupytext run cell by cell).
# Everything lives in global variables, the path is hard-coded, and nothing can be
# imported or tested. The package in `src/btitools/` is the same analysis turned
# into a module with a class. Compare the two files side by side.
#
# Run from the repository root:
#     uv run python sessions/01-careers-and-python/workspace/notebooks/explore_decisions.py

# %%
import pandas as pd

df = pd.read_parquet("case-study/data/train_sample.parquet")
print(df.shape)

# %%
# Question 2: share of each language
print(df["language"].value_counts(normalize=True).round(3).head(5))

# %%
# Question 3: the most frequent headings with their English names
names = pd.read_parquet("case-study/data/nomenclature.parquet")
counts = df["heading"].value_counts().head(5).rename("n").reset_index()
print(counts.merge(names[["heading", "heading_description"]], on="heading", how="left"))

# %%
# The same lines, copied into the next notebook with a small change, are how
# diverging versions of "the same" analysis appear. The module removes the copies.
