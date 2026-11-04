"""Shared test data, and how pytest reports a task that is not started yet.

pytest passes the fixtures below to every test that names them as a parameter.
"""

import pytest

from supermarket.checkout import Checkout
from supermarket.cli import load_catalogue
from supermarket.products import Product


@pytest.fixture
def catalogue() -> dict[str, Product]:
    return load_catalogue()


@pytest.fixture
def checkout(catalogue: dict[str, Product]) -> Checkout:
    return Checkout(catalogue)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Report a task test as "not started yet" while its code is missing.

    The code of a task raises NotImplementedError until you write it, and a class of a task that
    you have not written cannot be imported (ImportError). Any other result is a normal pass or fail.
    """
    outcome = yield
    report = outcome.get_result()
    task = item.get_closest_marker("task")
    if task and call.excinfo and call.excinfo.errisinstance((NotImplementedError, ImportError)):
        report.outcome = "skipped"
        # "Task 1", or "cannot import name 'PaymentError' from ...": keep it short, without the path
        reason = str(call.excinfo.value).split(" (")[0]
        report.longrepr = (str(item.path), item.location[1], f"not started yet: {reason}")
