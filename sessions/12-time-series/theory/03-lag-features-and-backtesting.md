# Lag features, backtesting and forecast metrics

The last block brings forecasting back to the machine-learning tools of Sessions 8–10. A regression model can forecast once the series is turned into a table of **lag features**. To decide between the baselines, exponential smoothing and the machine-learning model, one hold-out year is not enough: **rolling-origin backtesting** repeats the test over several years. The errors are summarised with **MAE** and the scaled **MASE**, and the coverage of the prediction intervals is checked. The page ends with a model recommendation and a forecast of the next twelve months of BTI decisions.

The code blocks build on each other; run them in order from the repository root.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

pd.set_option("display.width", 120)
counts = pd.read_parquet("case-study/data/monthly_counts.parquet")
y = (counts[counts["issuing_country"] != "GB"].groupby("month")["n_decisions"].sum()
     ["2010":"2026-09"].rename("decisions"))
y.index.freq = "MS"
log_y = np.log(y)
train, test = y[:"2025-09"], y["2025-10":]
```

## Machine learning with lag features

### Concept

Any regression model forecasts if we turn the series into a table. Each row is a period t; the target is y_t; the features are **lags** (earlier values y_{t−k}) and **calendar features** such as the month. The rule: a feature may only use information that is available at the moment the forecast is made.

For a forecast of the next 12 months there are two strategies:

- **Direct**: use only lags of 12 or more. Then every feature is known for all 12 forecast months, and one model predicts all of them.
- **Recursive**: predict one month ahead with lags 1, 2, …, then feed the prediction back in as a lag for the next month. Errors can accumulate.

Small example: to predict March 2026 directly from a forecast origin at the end of September 2025, the features are lag 12 (March 2025), lag 13 (February 2025) and lag 24 (March 2024), all known in September 2025. Lag 1 (February 2026) is not known in September 2025 and must not be used.

Tree ensembles (Session 10) cannot predict values outside the range of the training targets: a forest trained on months with at most 4,900 decisions never forecasts 5,500. On trending series this is a real limitation; transforming the target (logs, differences, growth rates) helps.

```mermaid
flowchart LR
    S["Series y"] --> L["Shift: lag 12, 13, 24"]
    S --> C["Calendar: month"]
    L --> T["Table: one row per month<br/>target = log y"]
    C --> T
    T --> M["Gradient boosting<br/>fit on past rows"]
    M --> F["Predict the next 12 rows<br/>(all features known)"]
```

### Why it matters

The table view lets forecasting use everything from Sessions 8–10: many series in one model, external features (prices, promotions, holidays, weather) and gradient boosting. Many top solutions of the M5 competition (Walmart sales, 42,840 series) were gradient-boosted trees on lag features (Makridakis, Spiliotis and Assimakopoulos, 2022). For a single short series, however, there is little for a flexible model to learn.

### How it works in Python

```python
frame = pd.DataFrame({"target": log_y})
for lag in (12, 13, 24):                       # only values known 12 months before the target month
    frame[f"lag{lag}"] = log_y.shift(lag)
frame["month"] = frame.index.month
frame = frame.dropna()                         # the first 24 months have no lag 24
print(frame.loc["2026-03-01"].round(2).to_dict())
# {'target': 8.24, 'lag12': 8.26, 'lag13': 8.23, 'lag24': 8.26, 'month': 3.0}

X_train, y_train = frame.loc[:"2025-09"].drop(columns="target"), frame.loc[:"2025-09", "target"]
X_test = frame.loc["2025-10":].drop(columns="target")
hgb = HistGradientBoostingRegressor(min_samples_leaf=5, random_state=0).fit(X_train, y_train)
lag_fc = pd.Series(np.exp(hgb.predict(X_test)), index=test.index)
print(len(X_train), f"training rows, MAE {np.mean(np.abs(test - lag_fc)):.0f}")
# 165 training rows, MAE 339
```

165 training rows remain. In the hold-out year the lag model (MAE 339) is about as good as the naive forecast (349) and worse than seasonal naive (258) and ETS (282). `min_samples_leaf=5` is needed because the default of 20 leaves almost no splits on so few rows. Workbook 06 (scikit-learn) shows lag features with more data, and workbook 09 (MLForecast) builds them automatically for many series.

### In practice

- **Retail.** Top M5 solutions used LightGBM on lags, rolling means, prices and calendar events across thousands of products and stores, so that one model learns from many series.
- **Energy.** Load forecasts combine lagged load with temperature forecasts and calendar features (weekday, holidays); the scikit-learn example on bike-sharing demand (workbook 07) shows the same idea for hourly demand.
- **Hospitals.** Emergency department arrivals are forecast from lags, weekday, holidays and weather.

> [!WARNING]
> **Lag 1 in a 12-month-ahead forecast is leakage.** In the backtest the true value of last month is available, in reality it is not. Check for every feature: would I know this value on the day I make the forecast?

> [!CAUTION]
> **Rolling means must also be shifted.** A 3-month rolling mean that includes the target month uses the target to predict itself. Compute it on the shifted series: `y.shift(12).rolling(3).mean()`.

## Rolling-origin backtesting

### Concept

One hold-out year is one sample of forecast performance. **Rolling-origin backtesting** (also called time series cross-validation) repeats the evaluation: fit on all data up to a **forecast origin**, forecast the next h periods, move the origin forward, repeat. Each fold uses only the past to predict the future. With an **expanding window** the training data grow from fold to fold; with a **sliding window** they keep a fixed length.

```mermaid
flowchart LR
    subgraph F1["Fold 1"]
        A1["Train 2010 to Sep 2021"] --> B1["Test Oct 2021 to Sep 2022"]
    end
    subgraph F2["Fold 2"]
        A2["Train 2010 to Sep 2022"] --> B2["Test Oct 2022 to Sep 2023"]
    end
    subgraph F3["Fold 3"]
        A3["Train 2010 to Sep 2023"] --> B3["Test Oct 2023 to Sep 2024"]
    end
    subgraph F4["Fold 4"]
        A4["Train 2010 to Sep 2024"] --> B4["Test Oct 2024 to Sep 2025"]
    end
    subgraph F5["Fold 5"]
        A5["Train 2010 to Sep 2025"] --> B5["Test Oct 2025 to Sep 2026"]
    end
    F1 --> F2 --> F3 --> F4 --> F5
```

scikit-learn's `TimeSeriesSplit(n_splits=5, test_size=12)` produces exactly these five folds on the 201 months. StatsForecast and MLForecast have `cross_validation` methods that do the same for many series (workbooks 08 and 09).

### Why it matters

Forecast accuracy varies strongly from year to year. Backtesting shows how often a model beats the baseline and how large its worst errors are, instead of crowning the model that happened to fit one particular year. It is the forecasting version of the cross-validation of Session 7, with the order of time respected.

### How it works in Python

The function below returns the forecasts of four models for one fold. The lag model reuses the table `frame` built above; ETS also returns its 95 % interval.

```python
def forecasts(y_tr, test_index):
    ets = ETSModel(np.log(y_tr), error="add", trend="add", damped_trend=True,
                   seasonal="add", seasonal_periods=12).fit(disp=False)
    pred = np.exp(ets.get_prediction(start=test_index[0], end=test_index[-1]).summary_frame(alpha=0.05))
    rows = frame.loc[:y_tr.index[-1]]
    hgb = HistGradientBoostingRegressor(min_samples_leaf=5, random_state=0).fit(
        rows.drop(columns="target"), rows["target"])
    point = {
        "naive": np.repeat(y_tr.iloc[-1], len(test_index)),
        "seasonal naive": y_tr.iloc[-12:].to_numpy(),
        "ETS": pred["mean"].to_numpy(),
        "lag model": np.exp(hgb.predict(frame.loc[test_index].drop(columns="target"))),
    }
    return point, pred[["pi_lower", "pi_upper"]]
```

## Forecast metrics: MAE and MASE

### Concept

- **MAE** = mean |y_t − ŷ_t|, in the units of the series. Easy to explain, but not comparable across series of different size (an error of 100 is large for a product with 50 sales a month and tiny for one with 50,000).
- **MASE** (mean absolute scaled error; Hyndman and Koehler, 2006) divides the MAE by the in-sample MAE of the seasonal naive forecast on the **training** data: scale = mean |y_t − y_{t−m}| over the training period. MASE < 1 means the forecast errs less than the seasonal naive forecast did, on average, in the past. MASE is comparable across series and well defined when values are close to zero.
- **Coverage** of a prediction interval is the share of actual values inside it; a 95 % interval should have coverage close to 0.95 over many forecasts.

Worked example: the training series 10, 12, 14, 11, 13, 15 with m = 3. The seasonal differences are |11 − 10|, |13 − 12|, |15 − 14| = 1, 1, 1, so the scale is 1. A forecast with MAE 1.5 on the test period has MASE 1.5: it errs 50 % more than seasonal naive did in-sample.

### Why it matters

Scaled errors make results comparable across series and across folds with different levels, which is why forecasting competitions (M4) used MASE. Reporting MAE alongside keeps the result understandable ("about 280 decisions per month off").

### How it works in Python

```python
def mae(actual, forecast):
    return np.mean(np.abs(np.asarray(actual) - np.asarray(forecast)))


def mase(actual, forecast, y_tr, m=12):
    history = np.asarray(y_tr)
    scale = np.mean(np.abs(history[m:] - history[:-m]))     # in-sample seasonal naive MAE
    return mae(actual, forecast) / scale


results = []
for train_idx, test_idx in TimeSeriesSplit(n_splits=5, test_size=12).split(y):
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    point, interval = forecasts(y_tr, y_te.index)
    covered = ((y_te >= interval["pi_lower"]) & (y_te <= interval["pi_upper"])).mean()
    for name, fc in point.items():
        results.append({"test from": y_te.index[0].strftime("%Y-%m"), "model": name, "MAE": mae(y_te, fc),
                        "MASE": mase(y_te, fc, y_tr), "ETS coverage": covered})
bt = pd.DataFrame(results)
print(bt.pivot(index="test from", columns="model", values="MASE").round(2))
# model       ETS  lag model  naive  seasonal naive
# test from
# 2021-10    0.50       0.87   0.90            0.74
# 2022-10    0.81       0.69   0.80            1.03
# 2023-10    0.68       0.87   0.91            1.01
# 2024-10    0.53       0.53   1.15            0.78
# 2025-10    0.65       0.78   0.80            0.59
print(bt.groupby("model")[["MAE", "MASE"]].mean().round(2))
#                    MAE  MASE
# model
# ETS             280.04  0.63
# lag model       330.42  0.75
# naive           402.97  0.91
# seasonal naive  366.68  0.83
print(bt.groupby("test from")["ETS coverage"].first().round(2).to_list())   # [0.92, 1.0, 0.92, 1.0, 1.0]
```

### Reading the backtest and recommending a model

- **ETS is best on average and never bad.** Its mean MASE is 0.63; it is the best or joint best model in three of the five folds, and its worst fold (0.81) is better than the worst fold of every other model. The seasonal naive forecast won the hold-out year of pages 1 and 2, but over five folds it is only third (MASE 0.83): one year would have led to the wrong choice.
- **The lag model beats the baselines but not ETS** (MASE 0.75). With one short series there is little for a tree ensemble to learn that the seasonal pattern and the level do not already say.
- **The ETS intervals are close to their promise.** They covered 58 of 60 test months (97 %) for a nominal 95 %: about right, slightly conservative.

A defensible recommendation for the monthly decision forecast: **damped ETS on log counts**, reported with its 95 % interval, because it has the lowest error in the backtest, models the December dip explicitly and gives intervals whose coverage has been checked. Report the seasonal naive forecast as a benchmark and refit every month. The forecast for the next twelve months, fitted on all data up to September 2026:

```python
final = ETSModel(log_y, error="add", trend="add", damped_trend=True,
                 seasonal="add", seasonal_periods=12).fit(disp=False)
outlook = np.exp(final.get_prediction(start="2026-10-01", end="2027-09-01").summary_frame(alpha=0.05))
print(outlook.loc[["2026-12-01", "2027-03-01"], ["mean", "pi_lower", "pi_upper"]].round(0))
#               mean  pi_lower  pi_upper
# 2026-12-01  3293.0    2713.0    3997.0
# 2027-03-01  4128.0    3370.0    5057.0
print(round(outlook["mean"].sum()))   # 45085 decisions expected from October 2026 to September 2027
```

The intervals are for single months. The interval for the twelve-month total is not the sum of the monthly limits (errors partly cancel); it would need a simulation from the model.

```mermaid
flowchart TD
    B["Backtest results<br/>(MASE per fold)"] --> Q1{"Clearly better than<br/>the best baseline?"}
    Q1 -->|no| Q2{"Gives intervals or<br/>other needed outputs?"}
    Q1 -->|yes| Q3{"Stable across folds,<br/>no very bad year?"}
    Q2 -->|yes| R1["Use it with the baseline<br/>as a benchmark"]
    Q2 -->|no| R2["Use the baseline"]
    Q3 -->|yes| R3["Use the model"]
    Q3 -->|no| R1
```

### In practice

- **Forecasting competitions.** The M4 competition ranked methods by a combination of MASE and the symmetric MAPE; the M5 accuracy track used a scaled squared error (RMSSE) for the same reason (Makridakis, Spiliotis and Assimakopoulos, 2020; 2022).
- **Energy trading and grid operation.** Load and price models are backtested on many past days, with coverage of the forecast quantiles checked, before they are used for bidding.
- **Central banks** evaluate forecasting models in real-time out-of-sample exercises that reproduce, at each origin, only the data known at that date.

> [!WARNING]
> **Never use shuffled k-fold cross-validation on a time series.** It puts later months into the training folds, and the error looks better than it will be in use.

> [!CAUTION]
> **Tuning on the backtest makes it optimistic.** If you choose hyperparameters (or the model) on the same folds you report, hold out a final period or report the result as a model-selection result, not as an estimate of future accuracy.

*Practice (block 3):* case study: backtest the models and recommend one with its prediction interval: Part 3 of [10-case-study-decision-forecast.ipynb](../workbooks/10-case-study-decision-forecast.ipynb).

## Check your understanding

1. For a forecast made on 30 September 2026 for March 2027, which of these features may be used: lag 1, lag 6, lag 12, the month, a 3-month rolling mean of the shifted series `y.shift(12)`?
2. Why can a gradient-boosting model trained on months with at most 4,900 decisions not forecast 5,500? How does a log transform or a growth target change this?
3. Compute the MASE scale for the training series 5, 7, 6, 8, 9, 7 with m = 2.
4. Seasonal naive won the hold-out year, ETS won the five-fold backtest. Which result do you trust more, and why?
5. A 95 % interval covers 58 of 60 test months. Is it well calibrated? What would 60 of 60 or 45 of 60 suggest?

## Further reading

- Hyndman, R. J., Athanasopoulos, G., Garza, A., Challu, C., Mergenthaler, M. and Olivares, K. G. (2026). *Forecasting: Principles and Practice, the Pythonic Way*. OTexts. Chapter 5 (the forecaster's toolbox: evaluating accuracy, time series cross-validation). https://otexts.com/fpppy/
- Hyndman, R. J. and Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2022). M5 accuracy competition: results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
- scikit-learn developers (2026). *Time-related feature engineering* (example). https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
