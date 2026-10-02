"""Shared test data and helpers."""

from __future__ import annotations

from typing import Any

import pytest


def open_meteo(days: list[str], **columns: list[Any]) -> dict[str, Any]:
    """A minimal Open-Meteo response with a ``daily`` block for ``days``.

    Missing columns are filled with plausible values: 20 °C, no rain, 8 hours of sunshine.
    """
    n = len(days)
    daily = {
        "time": days,
        "temperature_2m_mean": columns.get("temperature", [20.0] * n),
        "precipitation_sum": columns.get("precipitation", [0.0] * n),
        "sunshine_duration": columns.get("sunshine_seconds", [8 * 3600.0] * n),
    }
    return {
        "latitude": 52.52,
        "longitude": 13.42,
        "timezone": "Europe/Berlin",
        "daily_units": {"time": "iso8601", "temperature_2m_mean": "°C"},
        "daily": daily,
    }


@pytest.fixture
def raw_listing() -> dict[str, Any]:
    """A messy but valid listing, shaped like a row of the raw Inside Airbnb file."""
    return {
        "id": "3176",
        "name": "  Fabulous Flat   in great Location ",
        "neighbourhood_group_cleansed": "Pankow",
        "latitude": "52.53574",
        "longitude": 13.41734,
        "room_type": "entire home/apt",
        "accommodates": 2,
        "price": "$1,160.50",
        "minimum_nights": "2",
        "maximum_nights": 730,
        "host_name": "ignored: not part of the record",
    }
