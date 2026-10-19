# Session 12 · Time series forecasting

> **Guiding question.** How much will happen next month, and how certain is the forecast?

## 1 · Time series, baselines and exponential smoothing

[Open notebook](workbooks/01-lecture-baselines-and-exponential-smoothing.ipynb)

- Trend, seasonality and autocorrelation of the monthly Airbnb reviews in Berlin; STL and the ACF.
- Aggregating to regular time intervals with pandas: partial and empty periods, survivorship, the date and time toolkit.
- Baselines: naive, seasonal naive, moving average, seasonal naive × growth, drift and equivalent dates.
- Exponential smoothing: SES, Holt, Holt–Winters and ETS with prediction intervals and coverage.
- ARIMA as an outlook: stationarity, ACF and PACF, the airline model with the pandemic months marked as missing, `STLForecast` and the long data format.
- Worked example: how to treat the pandemic months.

## 2 · Machine learning forecasts and backtesting

[Open notebook](workbooks/02-lecture-ml-forecasts-and-backtesting.ipynb)

- Machine learning with lag features: the growth target, gradient boosting, direct against recursive forecasts.
- Lag features, quantile regression and calendar encodings on hourly bike demand.
- Rolling-origin backtesting: six forecast origins, refit strategies, `TimeSeriesSplit`.
- Forecast metrics: MAE, MASE, interval coverage and the pinball loss.
- Worked example: does the Berlin weather help, and the recommended 12-month forecast with its interval.

## Team project until the next session

Open points from the interim review.

## Setup

- The block notebooks need the Airbnb data and the Berlin weather: download `airbnb-v1.zip` from the course drive and unzip it into `case-study/data/` (see [case-study/README.md](../../case-study/README.md)). Block 2 also downloads the Bike Sharing Demand data from OpenML on first run.
- Extras 10–13 use LightGBM, which needs the OpenMP runtime installed in [Session 10](../10-tree-based-models/README.md#setup).
- Extras 03, 07 and 10–13 use StatsForecast, MLForecast or skforecast, which do not support pandas 3 yet. Open them in a temporary environment, from the repository root:

  ```bash
  uv run --no-project --python 3.14 --with jupyterlab --with statsforecast --with mlforecast \
      --with datasetsforecast --with lightgbm --with window-ops --with skforecast \
      jupyter lab sessions/12-time-series/workbooks/extras
  ```

[Extras and downloaded notebook provenance](workbooks/extras/README.md)
