"""A small client for CKAN open-data portals, with validated records.

Reference solution, for self-checking only: compare with your own version after
you have tried the exercise. Copy it over src/btitools/opendata.py to check it.

CKAN is the open-source software behind many open-data portals, including GovData
(Germany, https://www.govdata.de) and the Berlin open-data portal. Its search endpoint
``package_search`` returns JSON with the total number of matching datasets
(``count``) and one page of results; ``rows`` sets the page size and ``start`` the
offset (pagination).

By default the client asks GovData for datasets published by Berlin
(organisation ``berlin-open-data``). No API key is needed for public data.

Try it (needs internet access):

    uv run opendata-search baum
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from btitools.errors import APIError, RateLimitError, ValidationError

DEFAULT_BASE_URL = "https://ckan.govdata.de/api/3/action/"
BERLIN = "berlin-open-data"
USER_AGENT = "htw-spp-course-btitools/0.2 (teaching example)"


@dataclass
class DatasetRecord:
    """One dataset (a CKAN "package") of an open-data portal, checked and cleaned."""

    name: str
    title: str
    organization: str
    modified: datetime
    n_resources: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("name", "title"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(name, "must be a non-empty string")
            setattr(self, name, " ".join(value.split()))
        if not isinstance(self.n_resources, int) or self.n_resources < 0:
            raise ValidationError("n_resources", f"must be an integer >= 0, got {self.n_resources}")

    @classmethod
    def from_ckan(cls, package: Mapping[str, Any]) -> DatasetRecord:
        """Parse one element of ``result.results`` of a CKAN ``package_search`` response."""
        try:
            modified = datetime.fromisoformat(package["metadata_modified"])
        except KeyError as err:
            raise ValidationError("metadata_modified", "is missing") from err
        except (TypeError, ValueError) as err:
            raise ValidationError("metadata_modified", f"not an ISO date: {err}") from err
        organization = package.get("organization") or {}
        return cls(
            name=package.get("name", ""),
            title=package.get("title", ""),
            organization=organization.get("name", ""),
            modified=modified,
            n_resources=package.get("num_resources", len(package.get("resources", []))),
            tags=[tag["name"] for tag in package.get("tags", []) if "name" in tag],
        )


@dataclass
class SearchPage:
    """One page of search results."""

    count: int  # total number of matching datasets on the portal
    records: list[DatasetRecord]
    skipped: int = 0  # results that failed validation


class OpenDataClient:
    """Search a CKAN portal and return validated ``DatasetRecord`` objects.

    The client *has* an ``httpx.Client`` (composition): it delegates the HTTP work to it
    and adds what is specific to CKAN (URLs, parameters, error handling, parsing).

    Args:
        base_url: URL of the CKAN action API, ending in ``/api/3/action/``.
        organization: restrict searches to this publisher; ``None`` searches all.
        api_key: optional CKAN API token, sent in the ``Authorization`` header. Read it
            from an environment variable; never write it into the code.
        transport: replaces the network, e.g. ``httpx.MockTransport`` in tests.
        max_retries: how often to retry after HTTP 429 (too many requests).
        sleep: function used to wait between retries (replaced in tests).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        organization: str | None = BERLIN,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = api_key
        self.organization = organization
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    # context manager: `with OpenDataClient() as client:` closes the connection at the end
    def __enter__(self) -> OpenDataClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _get(self, action: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """GET ``action`` and return the ``result`` part of the CKAN response."""
        for attempt in range(self.max_retries + 1):
            response = self._http.get(action, params=params)
            if response.status_code != 429:
                break
            retry_after = float(response.headers.get("Retry-After", 2**attempt))
            if attempt == self.max_retries:
                raise RateLimitError("too many requests; giving up", retry_after=retry_after)
            self._sleep(retry_after)

        if response.is_error:
            raise APIError(
                f"{action} returned HTTP {response.status_code}", status_code=response.status_code
            )
        try:
            payload = response.json()
        except ValueError as err:
            raise APIError(f"{action} did not return JSON") from err
        if not payload.get("success", False):
            raise APIError(f"{action} failed: {payload.get('error')}")
        return payload["result"]

    def search(self, query: str = "", *, rows: int = 10, start: int = 0) -> SearchPage:
        """Return one page of datasets that match ``query``."""
        params: dict[str, Any] = {"q": query, "rows": rows, "start": start}
        if self.organization:
            params["fq"] = f"organization:{self.organization}"
        result = self._get("package_search", params)

        records, skipped = [], 0
        for package in result.get("results", []):
            try:
                records.append(DatasetRecord.from_ckan(package))
            except ValidationError:
                skipped += 1  # keep going; one bad record should not stop the search
        return SearchPage(count=int(result.get("count", 0)), records=records, skipped=skipped)

    def iter_datasets(
        self, query: str = "", *, page_size: int = 20, max_records: int = 100
    ) -> Iterator[DatasetRecord]:
        """Yield up to ``max_records`` datasets, requesting one page after the other."""
        start, yielded = 0, 0
        while yielded < max_records:
            page = self.search(query, rows=page_size, start=start)
            if not page.records and not page.skipped:
                return
            for record in page.records:
                yield record
                yielded += 1
                if yielded == max_records:
                    return
            start += page_size
            if start >= page.count:
                return


def main(argv: list[str] | None = None) -> None:
    """Command line: ``opendata-search [query]`` prints the first ten Berlin datasets."""
    args = sys.argv[1:] if argv is None else argv
    query = " ".join(args)
    with OpenDataClient(api_key=os.environ.get("CKAN_API_TOKEN")) as client:
        page = client.search(query, rows=10)
    print(f"{page.count} datasets match {query!r}; showing {len(page.records)}")
    for record in page.records:
        print(f"- {record.modified:%Y-%m-%d}  {record.title}  ({record.n_resources} files)")


if __name__ == "__main__":
    main()
