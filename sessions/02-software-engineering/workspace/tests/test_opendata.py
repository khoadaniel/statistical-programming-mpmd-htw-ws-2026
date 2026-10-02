"""Tests of the open-data client. No network: httpx.MockTransport answers instead."""

from __future__ import annotations

import httpx
import pytest
from conftest import ckan_package, ckan_response

from btitools.errors import APIError, RateLimitError, ValidationError
from btitools.opendata import DatasetRecord, OpenDataClient


def make_client(handler, **kwargs) -> OpenDataClient:
    """A client whose requests go to ``handler`` instead of the internet."""
    return OpenDataClient(transport=httpx.MockTransport(handler), **kwargs)


# --- parsing --------------------------------------------------------------------------


def test_dataset_record_from_ckan():
    record = DatasetRecord.from_ckan(
        ckan_package("strassenbaeume", title="  Straßenbäume  Berlin ")
    )
    assert record.name == "strassenbaeume"
    assert record.title == "Straßenbäume Berlin"
    assert record.organization == "berlin-open-data"
    assert record.modified.year == 2026
    assert record.n_resources == 2
    assert record.tags == ["bäume", "umwelt"]


@pytest.mark.parametrize(
    "broken",
    [
        {"title": ""},
        {"metadata_modified": "yesterday"},
        {"num_resources": -1},
    ],
)
def test_dataset_record_rejects_broken_packages(broken):
    with pytest.raises(ValidationError):
        DatasetRecord.from_ckan(ckan_package("x", **broken))


# --- requests -------------------------------------------------------------------------


def test_search_sends_query_page_and_organisation():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=ckan_response([ckan_package("a")], count=57))

    with make_client(handler) as client:
        page = client.search("baum", rows=5, start=10)

    assert seen["path"] == "/api/3/action/package_search"
    assert seen["params"] == {
        "q": "baum",
        "rows": "5",
        "start": "10",
        "fq": "organization:berlin-open-data",
    }
    assert page.count == 57
    assert [r.name for r in page.records] == ["a"]


def test_search_skips_invalid_records():
    packages = [ckan_package("good"), ckan_package("bad", title="")]

    def handler(request):
        return httpx.Response(200, json=ckan_response(packages))

    with make_client(handler) as client:
        page = client.search()
    assert [r.name for r in page.records] == ["good"]
    assert page.skipped == 1


def test_api_key_is_sent_as_header():
    def handler(request):
        assert request.headers["Authorization"] == "secret-token"
        return httpx.Response(200, json=ckan_response([]))

    with make_client(handler, api_key="secret-token") as client:
        assert client.search().records == []


@pytest.mark.parametrize("status", [404, 500, 503])
def test_http_errors_raise_api_error(status):
    def handler(request):
        return httpx.Response(status, text="error page")

    with make_client(handler) as client, pytest.raises(APIError) as excinfo:
        client.search()
    assert excinfo.value.status_code == status


def test_unsuccessful_payload_raises_api_error():
    def handler(request):
        return httpx.Response(200, json={"success": False, "error": {"message": "bad query"}})

    with make_client(handler) as client, pytest.raises(APIError, match="bad query"):
        client.search()


def test_rate_limit_is_retried_after_waiting():
    answers = [
        httpx.Response(429, headers={"Retry-After": "1.5"}),
        httpx.Response(200, json=ckan_response([ckan_package("a")])),
    ]
    waits: list[float] = []

    def handler(request):
        return answers.pop(0)

    with make_client(handler, sleep=waits.append) as client:
        page = client.search()
    assert waits == [1.5]
    assert len(page.records) == 1


def test_rate_limit_gives_up_after_max_retries():
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "0"})

    with make_client(handler, max_retries=2, sleep=lambda s: None) as client:
        with pytest.raises(RateLimitError):
            client.search()


# --- exercise -------------------------------------------------------------------------


@pytest.mark.skip(reason="exercise 4: delete this line when iter_datasets is implemented")
def test_iter_datasets_follows_pages():
    all_packages = [ckan_package(f"d{i}") for i in range(7)]
    starts: list[int] = []

    def handler(request):
        start, rows = int(request.url.params["start"]), int(request.url.params["rows"])
        starts.append(start)
        return httpx.Response(200, json=ckan_response(all_packages[start : start + rows], count=7))

    with make_client(handler) as client:
        names = [r.name for r in client.iter_datasets(page_size=3, max_records=100)]
        assert names == [f"d{i}" for i in range(7)]
        assert starts == [0, 3, 6]

        starts.clear()
        assert len(list(client.iter_datasets(page_size=3, max_records=4))) == 4
        assert starts == [0, 3]
