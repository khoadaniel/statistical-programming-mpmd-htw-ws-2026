# Session 5 · Exploratory analysis and statistics recap

> **Guiding question.** What is in the data, which differences and relationships are real, and how do we show them?

## 1 · Exploring data: distributions, charts and relationships

[Open notebook](workbooks/01-lecture-exploring-data.ipynb)

- Variable types: nominal, ordinal, count, continuous, date; codes that look like numbers.
- Distributions: histograms, bin width, density plots, ECDF; strip, box, violin and point plots by category.
- Centre and spread, robust summaries, frequency tables and validating a variable.
- Choosing a chart for the question; perception and proportional ink.
- matplotlib, seaborn and Plotly: labels, annotation, small multiples, interactive charts.
- Colour and accessibility: palettes, colour maps, greyscale, colour vision deficiency.
- Scatter plots, binned percentiles, Pearson and Spearman correlation, correlation matrices, testing a correlation.
- Confounding and Simpson's paradox; what a night in Berlin costs, and where.

## 2 · Testing differences and communicating findings

[Open notebook](workbooks/02-lecture-testing-and-communicating.ipynb)

- The logic of a test: null hypothesis, p-value, permutation test, confidence intervals, the central limit theorem.
- Welch t-test and Mann–Whitney test with confidence intervals and effect sizes; paired data.
- Contingency tables, chi-square test, Cramér's V, residuals, odds ratio, Fisher's exact test, McNemar.
- A/B tests: sample size, power by simulation, analysis, sample-ratio check and peeking.
- Choosing a test from a decision table; ANOVA and many tests.
- From correlation to the regression line; regression plots in seaborn.
- Communicating findings in a short report or a Streamlit dashboard.
- Do district price differences remain once room type and size are held fixed?

## Team project until the next session

Exploratory findings: decisions per heading, language, member state and year, and the long tail of rare headings (the many headings with only a few decisions each), presented in a short team review.

## Setup

- Airbnb data, once: download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)).
- Dashboard (block 2), in a terminal from the repository root: `uv run streamlit run sessions/05-eda-and-statistics/workbooks/dashboard_app.py`.
- Some extras need packages outside the course environment; see the extras page.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
