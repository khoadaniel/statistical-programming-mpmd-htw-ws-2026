# Session 12 · Time series forecasting

> [!NOTE]
> **Guiding question.** How much will happen next month, and how certain is the forecast?

**Learning outcomes.** Students are able to

- describe a time series and produce baseline forecasts
- fit an exponential smoothing model and a lag-feature model
- evaluate forecasts with backtesting and report prediction intervals

> [!TIP]
> The session is kept short. If time is needed elsewhere, it can be done as self-study: read the three theory pages in order and work through the case-study notebook, which covers all three practice tasks.

## Session plan

**0:00–0:45 · Time series and baselines** ([theory/01](theory/01-time-series-and-baselines.md))

- [Time series: trend, seasonality, autocorrelation](theory/01-time-series-and-baselines.md#time-series-trend-seasonality-and-autocorrelation)
- [Aggregating to regular time intervals with pandas](theory/01-time-series-and-baselines.md#aggregating-to-regular-time-intervals-with-pandas)
- [Baselines (naive, seasonal naive, moving average)](theory/01-time-series-and-baselines.md#baselines-naive-seasonal-naive-and-moving-average)
- *Practice:* how busy is Airbnb in Berlin month by month? Build the monthly number of guest reviews (a proxy for stays), check its quality (partial last month, missing listings, the pandemic break) and compute the baseline forecasts → Part 1 of [workbooks/10-case-study-airbnb-review-forecast.ipynb](workbooks/10-case-study-airbnb-review-forecast.ipynb)

**1:00–1:45 · Exponential smoothing and ARIMA** ([theory/02](theory/02-exponential-smoothing-and-arima.md))

- [Exponential smoothing](theory/02-exponential-smoothing-and-arima.md#exponential-smoothing) [with prediction intervals](theory/02-exponential-smoothing-and-arima.md#prediction-intervals)
- [ARIMA as an outlook](theory/02-exponential-smoothing-and-arima.md#arima-as-an-outlook)
- *Practice:* fit exponential smoothing to the review series, decide how to treat the pandemic months (keep, cut, or mark as missing) and compare with the baselines → Part 2 of [workbooks/10-case-study-airbnb-review-forecast.ipynb](workbooks/10-case-study-airbnb-review-forecast.ipynb)

**2:00–2:45 · Lag features, backtesting and metrics** ([theory/03](theory/03-lag-features-and-backtesting.md))

- [Machine learning with lag features (using the tree-based models of Session 10)](theory/03-lag-features-and-backtesting.md#machine-learning-with-lag-features)
- [Rolling-origin backtesting](theory/03-lag-features-and-backtesting.md#rolling-origin-backtesting)
- [Forecast metrics: MAE and MASE](theory/03-lag-features-and-backtesting.md#forecast-metrics-mae-and-mase)
- *Practice:* case study: how many Airbnb stays should Berlin expect next year? Backtest the models from six forecast origins, test whether the Berlin weather from Session 2 adds anything, and recommend one with its 12-month forecast and prediction interval → Part 3 of [workbooks/10-case-study-airbnb-review-forecast.ipynb](workbooks/10-case-study-airbnb-review-forecast.ipynb)

## Materials

| File | Content | Block | Status |
|---|---|---|---|
| [theory/01-time-series-and-baselines.md](theory/01-time-series-and-baselines.md) | Components, STL, ACF, resampling, data quality of the review series, baselines, MAE (Airbnb reviews) | 1 | core |
| [theory/02-exponential-smoothing-and-arima.md](theory/02-exponential-smoothing-and-arima.md) | SES by hand, Holt–Winters, ETS, choosing the history after a break, prediction intervals, seasonal ARIMA with missing months | 2 | core |
| [theory/03-lag-features-and-backtesting.md](theory/03-lag-features-and-backtesting.md) | Lag features, rolling-origin backtest, MASE, coverage, the Berlin weather as a regressor, recommendation | 3 | core |
| [workbooks/01-stl-decomposition.ipynb](workbooks/01-stl-decomposition.ipynb) | STL decomposition in statsmodels | 1 | optional |
| [workbooks/02-statsforecast-quickstart.ipynb](workbooks/02-statsforecast-quickstart.ipynb) | StatsForecast quick start: AutoARIMA with intervals | 2 | optional |
| [workbooks/03-exponential-smoothing.ipynb](workbooks/03-exponential-smoothing.ipynb) | SES, Holt and Holt–Winters in statsmodels | 2 | core |
| [workbooks/04-ets-models.ipynb](workbooks/04-ets-models.ipynb) | ETS models and prediction intervals in statsmodels | 2 | core |
| [workbooks/05-arima.ipynb](workbooks/05-arima.ipynb) | ARIMA models in statsmodels | 2 | optional |
| [workbooks/06-lagged-features-gradient-boosting.ipynb](workbooks/06-lagged-features-gradient-boosting.ipynb) | Lag features and gradient boosting with quantile losses (scikit-learn) | 3 | core |
| [workbooks/07-time-related-feature-engineering.ipynb](workbooks/07-time-related-feature-engineering.ipynb) | Calendar features: one-hot, cyclical, splines (scikit-learn) | 3 | optional |
| [workbooks/08-statsforecast-cross-validation.ipynb](workbooks/08-statsforecast-cross-validation.ipynb) | Rolling-origin cross-validation for many series (StatsForecast) | 3 | optional |
| [workbooks/09-mlforecast-walkthrough.ipynb](workbooks/09-mlforecast-walkthrough.ipynb) | Lag features, LightGBM and backtesting for many series (MLForecast) | 3 | optional |
| [workbooks/10-case-study-airbnb-review-forecast.ipynb](workbooks/10-case-study-airbnb-review-forecast.ipynb) | **Practice 1–3:** monthly Airbnb reviews in Berlin, baselines, ETS, airline model, lag model, backtest, does the weather help, recommendation, forecast of the next 12 months, district series (own) | 1–3 | core |

Sources and licences: [source.md](source.md).

## Before and after the session

**Preparation.** Skim Chapter 2 (time series graphics) of *Forecasting: Principles and Practice, the Pythonic Way* and run Part 1 of the [case-study notebook](workbooks/10-case-study-airbnb-review-forecast.ipynb). It needs the Airbnb data and the Berlin weather in `case-study/data/airbnb/` (see [case-study/README.md](../../case-study/README.md)).

**Team project until the next session.** Open points from the interim review.

**Further reading (optional).**

- Hyndman et al. (2026). [*Forecasting: Principles and Practice, the Pythonic Way*](https://otexts.com/fpppy/). OTexts. Chapters 2–3, 5, 8–9.
- Hyndman and Koehler (2006). [Another look at measures of forecast accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001). *International Journal of Forecasting*.
- Nixtla. [StatsForecast cross-validation tutorial](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html) and [MLForecast end-to-end walkthrough](https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/end_to_end_walkthrough.html).
- statsmodels. [Time series analysis (`tsa`)](https://www.statsmodels.org/stable/tsa.html).
- Inside Airbnb. [Data assumptions](https://insideairbnb.com/data-assumptions/): how listings, calendars and reviews are scraped, and what the data cannot show.

## Setup

Prepare the Airbnb data once from the repository root (downloads the Berlin snapshot of Inside Airbnb, about 100 MB, and the daily Berlin weather from Open-Meteo):

```bash
uv run python case-study/prepare_airbnb.py
```

The theory pages, the case-study notebook 10 and workbooks 01, 03–07 run in the course environment (`uv run jupyter lab` in the repository root: pandas, statsmodels, scikit-learn, polars).

The Nixtla workbooks 02, 08 and 09 need extra packages:

```bash
uv run --with statsforecast --with mlforecast --with utilsforecast --with datasetsforecast --with lightgbm jupyter lab
```

On macOS, LightGBM needs the OpenMP runtime: `brew install libomp`.

Several workbooks download their example data (CO₂, air passengers, M4, bike sharing from OpenML): an internet connection is needed. To regenerate the theory figures: `uv run python sessions/12-time-series/theory/figures/make_figures.py` from the repository root.
