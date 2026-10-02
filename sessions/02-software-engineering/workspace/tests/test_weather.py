"""Tests of the weather client. No network: httpx.MockTransport answers instead."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
from conftest import open_meteo

from listingtools.errors import APIError, RateLimitError, ValidationError
from listingtools.weather import DailyWeather, WeatherClient, parse_daily, yearly_chunks


def make_client(handler, **kwargs) -> WeatherClient:
    """A client whose requests go to ``handler`` instead of the internet."""
    return WeatherClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, **kwargs)


# --- parsing --------------------------------------------------------------------------


def test_daily_weather_converts_the_date():
    day = DailyWeather("2024-07-01", 17.6, 7.9, 9.27)
    assert day.day == date(2024, 7, 1)
    assert day.to_dict()["date"] == "2024-07-01"


@pytest.mark.parametrize(
    "changes",
    [
        {"day": "01.07.2024"},
        {"temperature": 61.0},
        {"temperature": None},
        {"precipitation": -0.1},
        {"sunshine_hours": 25.0},
        {"temperature": "warm"},
    ],
)
def test_daily_weather_rejects_implausible_values(changes):
    values = {"day": "2024-07-01", "temperature": 17.6, "precipitation": 7.9, "sunshine_hours": 9.3}
    with pytest.raises(ValidationError):
        DailyWeather(**{**values, **changes})


def test_parse_daily_converts_sunshine_seconds_to_hours():
    payload = open_meteo(["2024-07-01", "2024-07-02"], sunshine_seconds=[36000.0, 0.0])
    series = parse_daily(payload)
    assert [d.sunshine_hours for d in series.records] == [10.0, 0.0]
    assert series.skipped == 0


def test_parse_daily_skips_days_with_missing_values():
    payload = open_meteo(["2026-06-25", "2026-06-26"], temperature=[28.6, None])
    series = parse_daily(payload)
    assert len(series) == 1
    assert series.skipped == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "something else"},
        {"daily": {"time": ["2024-07-01"]}},
        {"daily": {**open_meteo(["2024-07-01"])["daily"], "precipitation_sum": []}},
    ],
)
def test_parse_daily_rejects_broken_payloads(payload):
    with pytest.raises(APIError):
        parse_daily(payload)


# --- requests -------------------------------------------------------------------------


def test_daily_sends_location_period_and_variables():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=open_meteo(["2024-07-01", "2024-07-02"]))

    with make_client(handler) as client:
        series = client.daily(date(2024, 7, 1), date(2024, 7, 2))

    assert seen["host"] == "archive-api.open-meteo.com"
    assert seen["params"] == {
        "latitude": "52.52",
        "longitude": "13.41",
        "daily": "temperature_2m_mean,precipitation_sum,sunshine_duration",
        "timezone": "Europe/Berlin",
        "start_date": "2024-07-01",
        "end_date": "2024-07-02",
    }
    assert len(series) == 2


def test_daily_rejects_a_reversed_period_without_a_request():
    def handler(request):
        raise AssertionError("no request expected")

    with make_client(handler) as client, pytest.raises(ValidationError, match="end"):
        client.daily(date(2024, 7, 2), date(2024, 7, 1))


def test_forecast_uses_the_forecast_endpoint():
    def handler(request):
        assert request.url.host == "api.open-meteo.com"
        assert request.url.params["forecast_days"] == "3"
        return httpx.Response(200, json=open_meteo(["2026-10-02", "2026-10-03", "2026-10-04"]))

    with make_client(handler) as client:
        assert len(client.forecast(days=3)) == 3


def test_bad_request_reports_the_reason_and_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        reason = "Latitude must be in range of -90 to 90°. Given: 152.52."
        return httpx.Response(400, json={"error": True, "reason": reason})

    with make_client(handler, latitude=152.52) as client:
        with pytest.raises(APIError, match="Latitude must be in range") as excinfo:
            client.daily(date(2024, 7, 1), date(2024, 7, 1))
    assert excinfo.value.status_code == 400
    assert len(calls) == 1


def test_rate_limit_is_retried_after_waiting():
    answers = [
        httpx.Response(429, headers={"Retry-After": "1.5"}),
        httpx.Response(200, json=open_meteo(["2024-07-01"])),
    ]
    waits: list[float] = []

    def handler(request):
        return answers.pop(0)

    client = WeatherClient(transport=httpx.MockTransport(handler), sleep=waits.append)
    with client:
        series = client.daily(date(2024, 7, 1), date(2024, 7, 1))
    assert waits == [1.5]
    assert len(series) == 1


def test_server_errors_and_timeouts_are_retried_with_growing_waits():
    events = ["timeout", 503, 200]
    waits: list[float] = []

    def handler(request):
        event = events.pop(0)
        if event == "timeout":
            raise httpx.ReadTimeout("too slow", request=request)
        if event == 503:
            return httpx.Response(503)
        return httpx.Response(200, json=open_meteo(["2024-07-01"]))

    client = WeatherClient(transport=httpx.MockTransport(handler), sleep=waits.append)
    with client:
        assert len(client.daily(date(2024, 7, 1), date(2024, 7, 1))) == 1
    assert waits == [1.0, 2.0]  # exponential backoff


def test_rate_limit_gives_up_after_max_retries():
    def handler(request):
        return httpx.Response(429)

    with make_client(handler, max_retries=2) as client, pytest.raises(RateLimitError):
        client.daily(date(2024, 7, 1), date(2024, 7, 1))


def test_html_instead_of_json_raises_api_error():
    def handler(request):
        return httpx.Response(200, text="<html>maintenance</html>")

    with make_client(handler) as client, pytest.raises(APIError, match="JSON"):
        client.daily(date(2024, 7, 1), date(2024, 7, 1))


# --- exercise -------------------------------------------------------------------------


@pytest.mark.skip(reason="exercise 4: delete this line when yearly_chunks is implemented")
def test_yearly_chunks():
    assert yearly_chunks(date(2023, 11, 1), date(2025, 2, 28)) == [
        (date(2023, 11, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 2, 28)),
    ]
    assert yearly_chunks(date(2024, 7, 1), date(2024, 7, 1)) == [(date(2024, 7, 1),) * 2]
    with pytest.raises(ValueError):
        yearly_chunks(date(2024, 7, 2), date(2024, 7, 1))


@pytest.mark.skip(reason="exercise 4: delete this line when history is implemented")
def test_history_requests_one_year_at_a_time():
    periods: list[tuple[str, str]] = []

    def handler(request):
        start, end = request.url.params["start_date"], request.url.params["end_date"]
        periods.append((start, end))
        days = [d.isoformat() for d in (date.fromisoformat(start), date.fromisoformat(end))]
        return httpx.Response(200, json=open_meteo(days))

    with make_client(handler) as client:
        series = client.history(date(2022, 12, 1), date(2024, 1, 31))
    assert periods == [
        ("2022-12-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-01-31"),
    ]
    assert len(series) == 6  # the fake server answers with the first and last day of each chunk
