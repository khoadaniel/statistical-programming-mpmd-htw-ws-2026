"""Shared test data: six decisions small enough to check every answer by hand."""

import pandas as pd
import pytest


@pytest.fixture
def tiny_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bti_reference": ["DE1", "DE2", "FR1", "DE3", "PL1", "FR2"],
            "issuing_country": ["DE", "DE", "FR", "DE", "PL", "FR"],
            "language": ["de", "de", "fr", "de", "pl", "fr"],
            "start_date": pd.to_datetime(
                ["2021-01-04", "2021-03-01", "2021-06-15", "2023-02-01", "2023-05-10", "2023-11-30"]
            ),
            "end_date": pd.to_datetime(
                ["2024-01-03", "2024-02-29", "2024-06-14", "2026-01-31", "2026-05-09", "2026-11-29"]
            ),
            "status": ["INVALID", "INVALID", "INVALID", "VALID", "INVALID", "VALID"],
            "description": [
                "Plüschtier",
                "Spielzeugauto",
                "Sac",
                "Kunststoffbox",
                "Lalka",
                "Poupée",
            ],
            "heading": ["9503", "9503", "4202", "3926", "9503", "9503"],
        }
    )
