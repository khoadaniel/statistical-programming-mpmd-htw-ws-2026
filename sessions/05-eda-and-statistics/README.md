# Session 5 · Exploratory analysis and statistics recap: descriptive statistics, visualisation, tests, contingency tables and correlation

> [!NOTE]
> **Guiding question.** What is in the data, which differences and relationships are real, and how do we show them?

**Learning outcomes.** Students are able to

- describe data by variable type with descriptive statistics and well-designed charts
- apply statistical tests and contingency tables in Python and report confidence intervals and effect sizes
- compute and interpret correlations as the starting point for regression

## Session plan

### 0:00–0:45 · Describing data and designing charts

- Describing data by variable type: [variable types](theory/01-describing-data.md#variable-types), [distributions](theory/01-describing-data.md#distributions), [centre and spread](theory/01-describing-data.md#centre-and-spread), [robust summaries](theory/01-describing-data.md#robust-summaries), [frequency tables](theory/01-describing-data.md#frequency-tables)
- Choosing and designing charts with matplotlib, seaborn and Plotly: [choosing a chart](theory/02-chart-design.md#choosing-a-chart-for-the-question), [perception](theory/02-chart-design.md#perception-and-the-dataink-ratio), [matplotlib and annotation](theory/02-chart-design.md#matplotlib-figures-axes-labels-and-annotation), [colour and accessibility](theory/02-chart-design.md#colour-and-accessibility), [seaborn](theory/02-chart-design.md#seaborn-statistical-plots-and-small-multiples), [Plotly](theory/02-chart-design.md#interactive-charts-with-plotly-express)

*Practice:* What does a night in Berlin cost, and where? Price distribution by room type, median price and listings by district, reviews per month since 2015 (Inside Airbnb Berlin); [improve a poorly designed chart](theory/02-chart-design.md#critique-and-improve-a-chart).

### 1:00–1:45 · Comparing groups

- [Comparing groups: t-test and Mann–Whitney test with confidence intervals and effect sizes](theory/03-comparing-groups-and-tests.md#t-test-and-mannwhitney-test-with-confidence-intervals-and-effect-sizes)
- [Categorical data: contingency tables, chi-square test and Cramér's V](theory/03-comparing-groups-and-tests.md#categorical-data-contingency-tables-chi-square-and-cramérs-v)
- [A/B tests as an application](theory/03-comparing-groups-and-tests.md#ab-tests-as-an-application)
- [Choosing a test from a decision table](theory/03-comparing-groups-and-tests.md#choosing-a-test-from-a-decision-table)

*Practice:* How much more does an entire home cost than a private room, and Mitte than Neukölln (Welch, Mann–Whitney, CI, Cohen's d, CLES)? Does superhost status change the price (the two tests disagree)? Multi-listing host × licence entry and district × room type (chi-square, Cramér's V); a simulated A/B test of a listing page.

### 2:00–2:45 · Relationships and communication

- Relationships between numerical variables: [scatter plots](theory/04-correlation-and-communication.md#scatter-plots), [Pearson and Spearman correlation](theory/04-correlation-and-communication.md#pearson-and-spearman-correlation), [confounding and Simpson's paradox](theory/04-correlation-and-communication.md#confounding-and-simpsons-paradox)
- [From correlation to the regression line](theory/04-correlation-and-communication.md#from-correlation-to-the-regression-line) (the bridge to Session 6)
- [Communicating findings in a short report or a Streamlit dashboard](theory/04-correlation-and-communication.md#communicating-findings-a-short-report-or-a-streamlit-dashboard)

*Practice:* How strongly does the price rise with the number of guests (Pearson, Spearman, log scale)? Do district price differences survive once room type, size and the type of stay are held fixed (a €10 gap that vanishes)? A one-page finding or a Streamlit dashboard of prices by district for a city housing analyst.

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-describing-data.md](theory/01-describing-data.md) | Variable types, distributions, centre and spread, robust summaries, frequency tables | 1 | core |
| [theory/02-chart-design.md](theory/02-chart-design.md) | Chart choice, perception, matplotlib, colour and accessibility, seaborn, Plotly, chart critique | 1 | core |
| [theory/03-comparing-groups-and-tests.md](theory/03-comparing-groups-and-tests.md) | t-test, Mann–Whitney, CIs, effect sizes, chi-square, Cramér's V, A/B tests, test decision table | 2 | core |
| [theory/04-correlation-and-communication.md](theory/04-correlation-and-communication.md) | Scatter plots, Pearson/Spearman, confounding, Simpson's paradox, regression line, report and dashboard | 3 | core |
| [workbooks/01-exploratory-data-analysis.ipynb](workbooks/01-exploratory-data-analysis.ipynb) | Think Stats ch. 1: first look at a dataset | 1 | core |
| [workbooks/02-descriptive-statistics-and-t-test.ipynb](workbooks/02-descriptive-statistics-and-t-test.ipynb) | Descriptive statistics, CIs and a t-test on baseball players | 1–2 | core |
| [workbooks/03-matplotlib-quick-start.ipynb](workbooks/03-matplotlib-quick-start.ipynb) | Figures, Axes, labels, the object-oriented interface | 1 | core |
| [workbooks/04-seaborn-distributions.ipynb](workbooks/04-seaborn-distributions.ipynb) | Histograms, KDE, ECDF with seaborn | 1 | core |
| [workbooks/05-seaborn-categorical.ipynb](workbooks/05-seaborn-categorical.ipynb) | Box, violin, bar and point plots by category | 1 | optional |
| [workbooks/06-seaborn-color-palettes.ipynb](workbooks/06-seaborn-color-palettes.ipynb) | Qualitative, sequential and diverging palettes | 1 | core |
| [workbooks/07-matplotlib-colormaps.ipynb](workbooks/07-matplotlib-colormaps.ipynb) | Choosing colour maps; perceptual uniformity; greyscale | 1 | optional |
| [workbooks/08-plotly-express.ipynb](workbooks/08-plotly-express.ipynb) | Plotly Express gallery of interactive charts | 1 | core |
| [workbooks/09-plotly-express-styling.ipynb](workbooks/09-plotly-express-styling.ipynb) | Labels, colours, templates and hover text in Plotly | 1 | optional |
| [workbooks/10-hypothesis-testing.ipynb](workbooks/10-hypothesis-testing.ipynb) | Hypothesis testing and t-tests on NHANES data | 2 | core |
| [workbooks/11-hypothesis-testing-by-simulation.ipynb](workbooks/11-hypothesis-testing-by-simulation.ipynb) | Think Stats ch. 9: permutation tests, "there is only one test" | 2 | core |
| [workbooks/12-one-way-anova.ipynb](workbooks/12-one-way-anova.ipynb) | Levene test, one-way ANOVA, post-hoc tests (three or more groups) | 2 | optional |
| [workbooks/13-contingency-tables-and-chi-square.ipynb](workbooks/13-contingency-tables-and-chi-square.ipynb) | Chi-square, Fisher exact, McNemar, Cochran's Q | 2 | core |
| [workbooks/14-categorical-relationships.ipynb](workbooks/14-categorical-relationships.ipynb) | Chi-square tests on NHANES data | 2 | optional |
| [workbooks/15-correlation.ipynb](workbooks/15-correlation.ipynb) | Think Stats ch. 7: scatter plots, Pearson, Spearman, causation | 3 | core |
| [workbooks/16-correlation-and-simple-regression.ipynb](workbooks/16-correlation-and-simple-regression.ipynb) | Pearson, Spearman, Kendall; simple regression | 3 | core |
| [workbooks/17-seaborn-regression.ipynb](workbooks/17-seaborn-regression.ipynb) | `regplot`, `lmplot`, residual plots in seaborn | 3 | optional |
| [workbooks/18-case-study-airbnb-exploration.ipynb](workbooks/18-case-study-airbnb-exploration.ipynb) | **Case study (Inside Airbnb Berlin)** for all three practice tasks | 1–3 | core |
| [workbooks/dashboard_app.py](workbooks/dashboard_app.py) | Streamlit dashboard of short-stay prices and listings by district, with exercises | 3 | core |

Sources and licences of third-party notebooks: [source.md](source.md).

## Before and after the session

**Preparation.** Revise mean, median, standard deviation, confidence interval and p-value from your first-semester statistics module. Prepare the Airbnb data once with `uv run python case-study/prepare_airbnb.py` (downloads about 100 MB) and run the first two cells of the [case-study notebook](workbooks/18-case-study-airbnb-exploration.ipynb) to check that the data load. Optional: read chapter 1 of Healy, *Data Visualization* (link below).

**Team project until the next session.** Exploratory and statistical findings of the project, presented in a short team review.

**Further reading (free).**

- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. <https://clauswilke.com/dataviz/>
- Downey, A. B. (2025). *Think Stats* (3rd ed.). <https://allendowney.github.io/ThinkStats/>
- Poldrack, R. A. (2023). *Statistical Thinking for the 21st Century* (Python companion). <https://statsthinking21.github.io/statsthinking21-python/>
- Healy, K. (2018). *Data Visualization: A Practical Introduction*, chapter 1. <https://socviz.co/lookatdata.html>
- Streamlit documentation: *Get started*. <https://docs.streamlit.io/get-started/tutorials>

## Setup

The course environment (root `pyproject.toml`) contains everything needed: pandas, pyarrow, matplotlib, seaborn, plotly, scipy, statsmodels, streamlit. From the repository root:

```bash
uv run python case-study/prepare_airbnb.py   # Inside Airbnb Berlin, once (case study, figures, dashboard)
uv run jupyter lab                       # notebooks
uv run streamlit run sessions/05-eda-and-statistics/workbooks/dashboard_app.py   # dashboard
uv run python sessions/05-eda-and-statistics/theory/figures/make_figures.py     # regenerate figures
```

Additional packages for single third-party notebooks (the notebooks install some of them themselves with `%pip install`):

| Workbook | Extra packages |
|---|---|
| 01, 11, 15 (Think Stats) | `empiricaldist`, `statadict` (installed by the notebook); data downloaded on first use |
| 07 (colour maps) | `colorspacious` |
| 08 (Plotly Express), one image cell | `scikit-image` |
| 10, 14 (statsthinking21) | `nhanes` (installed by the notebook); data downloaded on first use |

Example: `uv run --with colorspacious --with scikit-image --with empiricaldist --with statadict --with nhanes jupyter lab`. The seaborn tutorials download their example datasets from the internet on first use.
