"""Shared test data and helpers."""

from __future__ import annotations

from typing import Any

import pytest


def ckan_package(name: str, title: str = "Straßenbäume", **extra: Any) -> dict[str, Any]:
    """A minimal CKAN package, shaped like one element of package_search results."""
    package = {
        "name": name,
        "title": title,
        "metadata_modified": "2026-01-22T08:12:27.224114",
        "num_resources": 2,
        "organization": {"name": "berlin-open-data", "title": "Berlin Open Data"},
        "tags": [{"name": "bäume"}, {"name": "umwelt"}],
    }
    package.update(extra)
    return package


def ckan_response(packages: list[dict[str, Any]], count: int | None = None) -> dict[str, Any]:
    """A CKAN package_search response wrapping ``packages``."""
    return {
        "help": "https://ckan.govdata.de/api/3/action/help_show?name=package_search",
        "success": True,
        "result": {"count": len(packages) if count is None else count, "results": packages},
    }


@pytest.fixture
def raw_decision() -> dict[str, Any]:
    """A messy but valid decision as it might arrive from the EBTI export or a web form."""
    return {
        "bti_reference": "  DE0001/23-1 ",
        "issuing_country": " de",
        "language": "DE",
        "start_date": "10/05/2023",
        "end_date": "09/05/2026",
        "description": "Plüschtier in Form eines Bären,\n  Höhe   30 cm ",
        "keywords": "TOYS, PLUSH",
        "heading": "9503",
        "classification_justification": "ignored: not part of the record",
    }
