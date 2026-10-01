"""Make the figures of the Session 12 theory pages.

Run from the repository root (needs case-study/data/train.parquet):
    uv run --with pandas --with pyarrow --with matplotlib --with statsmodels \
        python sessions/12-time-series/theory/figures/make_figures.py

The series is the number of reviews per calendar month, January 2012 to December 2021.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
})

reviews = pd.read_parquet("case-study/data/train.parquet", columns=["date"])
y = reviews.set_index("date").resample("MS").size()["2012":"2021"].rename("reviews")
y.index.freq = "MS"
log_y = np.log(y)
train, test = y[:"2020"], y["2021"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=110)
    plt.close(fig)
    print("wrote", name)


def decomposition():
    stl = STL(log_y, period=12, robust=True).fit()
    parts = [("observed: log(reviews per month)", log_y), ("trend", stl.trend),
             ("seasonal", stl.seasonal), ("remainder", stl.resid)]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for ax, (title, s), colour in zip(axes, parts, [INK, SERIES[0], SERIES[1], SERIES[2]]):
        if title == "remainder":
            ax.bar(s.index, s, width=20, color=colour)
        else:
            ax.plot(s.index, s, color=colour, lw=1.8)
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle("STL decomposition of monthly review counts (log scale, period 12)", color=INK)
    save(fig, "decomposition")


def acf_figure():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    plot_acf(y, lags=24, ax=a1, color=SERIES[0], vlines_kwargs={"colors": SERIES[0]})
    a1.set(title="ACF of the counts: the trend dominates", xlabel="lag (months)")
    plot_acf(log_y.diff().dropna(), lags=24, ax=a2, color=SERIES[1], vlines_kwargs={"colors": SERIES[1]})
    a2.set(title="ACF of monthly growth (diff of log): peak at lag 12", xlabel="lag (months)")
    save(fig, "acf")


def baselines():
    naive = pd.Series(train.iloc[-1], index=test.index)
    snaive = pd.Series(train.iloc[-12:].to_numpy(), index=test.index)
    moving = pd.Series(train.iloc[-3:].mean(), index=test.index)
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.plot(y["2018":].index, y["2018":], color=INK, lw=2, label="actual")
    for (name, f), colour in zip({"naive": naive, "seasonal naive": snaive, "moving average (3)": moving}.items(), SERIES):
        ax.plot(f.index, f, color=colour, lw=2, ls="--", label=name)
    ax.axvline(test.index[0], color=MUTED, lw=1, ls=":")
    ax.text(test.index[0], ax.get_ylim()[1] * 0.98, " hold-out 2021", va="top", color=INK, fontsize=9)
    ax.set(title="Baseline forecasts for 2021 against the actual counts", ylabel="reviews per month")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "baselines")


def interval():
    fit = ETSModel(np.log(train), error="add", trend="add", damped_trend=True,
                   seasonal="add", seasonal_periods=12).fit(disp=False)
    frame = np.exp(fit.get_prediction(start=test.index[0], end=test.index[-1]).summary_frame(alpha=0.05))
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.plot(y["2018":].index, y["2018":], color=INK, lw=2, label="actual")
    ax.fill_between(frame.index, frame["pi_lower"], frame["pi_upper"], color=SERIES[0], alpha=0.18,
                    label="95 % prediction interval")
    ax.plot(frame.index, frame["mean"], color=SERIES[0], lw=2, ls="--", label="ETS point forecast")
    ax.set(title="Damped exponential smoothing (ETS on log counts): the interval widens with the horizon",
           ylabel="reviews per month")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "ets_interval")


if __name__ == "__main__":
    decomposition()
    acf_figure()
    baselines()
    interval()
