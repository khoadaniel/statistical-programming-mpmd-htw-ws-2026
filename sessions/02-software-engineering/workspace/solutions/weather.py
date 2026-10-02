"""Daily Berlin weather from the Open-Meteo API, parsed into validated records.

Reference solution, for self-checking only: compare with your own version after
you have tried the exercise. Copy it over src/listingtools/weather.py to check it.

Why: does the weather explain how busy Berlin's Airbnb market is? Weather changes every day
and is not shipped as a file with the listings; Open-Meteo offers it through a web API. The
archive endpoint returns past days, the forecast endpoint the next days, both as JSON:

    {"daily": {"time": ["2024-07-01", ...], "temperature_2m_mean": [17.6, ...],
               "precipitation_sum": [7.9, ...], "sunshine_duration": [33378.58, ...]}, ...}

No key is needed for non-commercial use. The free API allows fewer than 600 calls per minute,
5,000 per hour and 10,000 per day; a request for more than two weeks of data counts as several
calls (one year with three variables is about 26). Data: CC BY 4.0, https://open-meteo.com.
Session 12 uses the same data, prepared once by case-study/prepare_airbnb.py as
weather_daily.parquet.

Try it (needs internet access):

    uv run berlin-weather 2024-07-01 2024-07-07
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx

from listingtools.errors import APIError, RateLimitError, ValidationError

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BERLIN = (52.52, 13.41)  # latitude, longitude of the city centre
DAILY_VARIABLES = ("temperature_2m_mean", "precipitation_sum", "sunshine_duration")
USER_AGENT = "htw-spp-course-listingtools/0.2 (teaching example)"


def _number(value: Any, field: str, low: float, high: float) -> float:
    """A finite number between ``low`` and ``high``; ``None`` (a missing value) is rejected."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(field, f"expected a number, got {value!r}")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValidationError(field, f"{value} lies outside {low} to {high}")
    return float(value)


@dataclass
class DailyWeather:
    """The weather of one day in Berlin, checked for plausible values."""

    day: date
    temperature: float  # daily mean temperature 2 m above ground, in °C
    precipitation: float  # rain and snow, in mm
    sunshine_hours: float  # hours of sunshine

    def __post_init__(self) -> None:
        if isinstance(self.day, str):
            try:
                self.day = date.fromisoformat(self.day)
            except ValueError as err:
                raise ValidationError("day", f"expected YYYY-MM-DD, got {self.day!r}") from err
        if not isinstance(self.day, date):
            raise ValidationError("day", f"expected a date, got {type(self.day).__name__}")
        self.temperature = _number(self.temperature, "temperature", -40, 45)
        self.precipitation = _number(self.precipitation, "precipitation", 0, 300)
        self.sunshine_hours = _number(self.sunshine_hours, "sunshine_hours", 0, 24)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.day.isoformat(),
            "temperature": self.temperature,
            "precipitation": self.precipitation,
            "sunshine_hours": round(self.sunshine_hours, 2),
        }


@dataclass
class DailySeries:
    """The result of one or more requests: valid days, and the number of days skipped."""

    records: list[DailyWeather] = field(default_factory=list)
    skipped: int = 0  # days with missing or implausible values

    def __len__(self) -> int:
        return len(self.records)

    def extend(self, other: DailySeries) -> None:
        self.records.extend(other.records)
        self.skipped += other.skipped


def parse_daily(payload: Mapping[str, Any]) -> DailySeries:
    """Turn the ``daily`` part of an Open-Meteo response into ``DailyWeather`` records.

    Days with a missing value (``null`` in JSON, for example the most recent days of the
    archive) or an implausible one are skipped and counted, so that one bad day does not stop
    a ten-year download.

    Raises:
        APIError: if the payload lacks ``daily`` or its arrays have different lengths.
    """
    daily = payload.get("daily")
    if not isinstance(daily, Mapping):
        raise APIError("the response has no 'daily' block")
    columns = ["time", *DAILY_VARIABLES]
    missing = [c for c in columns if c not in daily]
    if missing:
        raise APIError(f"the response lacks the daily variables {missing}")
    lengths = {len(daily[c]) for c in columns}
    if len(lengths) != 1:
        raise APIError(f"the daily arrays have different lengths {sorted(lengths)}")

    series = DailySeries()
    for day, temperature, rain, sunshine in zip(*(daily[c] for c in columns), strict=True):
        try:
            hours = sunshine / 3600 if isinstance(sunshine, int | float) else sunshine
            series.records.append(DailyWeather(day, temperature, rain, hours))
        except ValidationError:
            series.skipped += 1
    return series


def yearly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split the period from ``start`` to ``end`` (both included) into calendar years.

    Example: 2023-11-01 to 2025-02-28 gives three chunks, 2023-11-01 to 2023-12-31,
    2024-01-01 to 2024-12-31 and 2025-01-01 to 2025-02-28.
    """
    if end < start:
        raise ValueError(f"{end} lies before {start}")
    return [
        (max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


class WeatherClient:
    """Fetch daily Berlin weather from Open-Meteo and return validated records.

    The client *has* an ``httpx.Client`` (composition): it delegates the HTTP work to it and
    adds what is specific to Open-Meteo (parameters, error messages, retries, parsing).

    Args:
        latitude, longitude: the location; the default is the centre of Berlin.
        transport: replaces the network, e.g. ``httpx.MockTransport`` in tests.
        timeout: seconds to wait for an answer before giving up on one attempt.
        max_retries: how often to retry after HTTP 429, a server error (5xx) or a timeout.
        sleep: function used to wait between retries (replaced in tests).
    """

    def __init__(
        self,
        *,
        latitude: float = BERLIN[0],
        longitude: float = BERLIN[1],
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    # context manager: `with WeatherClient() as client:` closes the connection at the end
    def __enter__(self) -> WeatherClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _params(self, **extra: Any) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "Europe/Berlin",
            **extra,
        }

    def _get(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """GET ``url`` with retries and return the JSON payload as a dictionary."""
        for attempt in range(self.max_retries + 1):
            wait = 2.0**attempt  # exponential backoff: 1, 2, 4, ... seconds
            try:
                response = self._http.get(url, params=params)
            except httpx.TimeoutException as err:
                if attempt == self.max_retries:
                    raise APIError(f"no answer from {url} after {attempt + 1} attempts") from err
                self._sleep(wait)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("Retry-After", wait))
                if attempt == self.max_retries:
                    if response.status_code == 429:
                        raise RateLimitError("too many requests; giving up", retry_after)
                    raise APIError(f"server error {response.status_code}", response.status_code)
                self._sleep(retry_after)
                continue
            break

        if response.is_error:  # 4xx: the request itself is wrong; retrying does not help
            try:
                reason = response.json().get("reason", response.text)
            except ValueError:
                reason = response.text
            raise APIError(f"HTTP {response.status_code}: {reason}", response.status_code)
        try:
            return response.json()
        except ValueError as err:
            raise APIError(f"{url} did not return JSON") from err

    def daily(self, start: date, end: date) -> DailySeries:
        """Past daily weather from the archive, from ``start`` to ``end`` (both included)."""
        if end < start:
            raise ValidationError("end", f"{end} lies before start {start}")
        params = self._params(start_date=start.isoformat(), end_date=end.isoformat())
        return parse_daily(self._get(ARCHIVE_URL, params))

    def forecast(self, days: int = 7) -> DailySeries:
        """The daily forecast for today and the next ``days - 1`` days (at most 16)."""
        if not 1 <= days <= 16:
            raise ValidationError("days", f"must be between 1 and 16, got {days}")
        return parse_daily(self._get(FORECAST_URL, self._params(forecast_days=days)))

    def history(self, start: date, end: date) -> DailySeries:
        """Past daily weather for a long period, requested one calendar year at a time.

        Small requests keep each answer small, show progress and, if one request fails,
        lose one year instead of ten. This is the same idea as pagination.
        """
        series = DailySeries()
        for chunk_start, chunk_end in yearly_chunks(start, end):
            series.extend(self.daily(chunk_start, chunk_end))
        return series


def main(argv: list[str] | None = None) -> None:
    """Command line: ``berlin-weather START END`` prints the daily weather of a period."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: berlin-weather YYYY-MM-DD YYYY-MM-DD")
    start, end = (date.fromisoformat(a) for a in args)
    with WeatherClient() as client:
        series = client.daily(start, end)
    print(f"{len(series)} days, {series.skipped} skipped (Open-Meteo, CC BY 4.0)")
    for day in series.records:
        print(
            f"{day.day}  {day.temperature:5.1f} °C  {day.precipitation:5.1f} mm  "
            f"{day.sunshine_hours:4.1f} h sun"
        )


if __name__ == "__main__":
    main()
