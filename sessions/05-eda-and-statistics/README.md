# Session 5 · Exploratory analysis and statistics recap

> **Guiding question.** What is in the data, which differences and relationships are real, and how do we show them?

## 1 · Exploring data: distributions, charts and relationships

[Open notebook](workbooks/01-lecture-exploring-data.ipynb)

- Variable types: nominal, ordinal, count, continuous, date; codes that look like numbers.
- Distributions: histograms, bin width, density plots, ECDF, box and violin plots.
- Centre and spread, robust summaries and frequency tables.
- Choosing a chart for the question; perception and proportional ink.
- matplotlib, seaborn and Plotly: labels, annotation, small multiples, interactive charts.
- Colour and accessibility: palettes, colour maps, greyscale, colour vision deficiency.
- Scatter plots, Pearson and Spearman correlation, correlation matrices.
- Confounding and Simpson's paradox; what a night in Berlin costs, and where.

## 2 · Testing differences and communicating findings

[Open notebook](workbooks/02-lecture-testing-and-communicating.ipynb)

- The logic of a test: null hypothesis, p-value, permutation test, confidence intervals.
- Welch t-test and Mann–Whitney test with confidence intervals and effect sizes; paired data.
- Contingency tables, chi-square test, Cramér's V, odds ratio, Fisher's exact test.
- A/B tests: sample size, analysis, sample-ratio check, power and peeking.
- Choosing a test from a decision table; ANOVA and many tests.
- From correlation to the regression line.
- Communicating findings in a short report or a Streamlit dashboard.
- Do district price differences survive once room type and size are held fixed?

## Team project until the next session

Exploratory findings: decisions per heading, language, member state and year, and the long tail of rare headings, presented in a short team review.

## Setup

- Airbnb data, once: `uv run python case-study/prepare_airbnb.py`.
- Dashboard (block 2), in a terminal from the repository root: `uv run streamlit run sessions/05-eda-and-statistics/workbooks/dashboard_app.py`.
- Some extras need packages outside the course environment; see the extras page.

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
