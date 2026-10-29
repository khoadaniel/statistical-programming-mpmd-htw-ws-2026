"""Shared test data: pytest passes these fixtures to every test that names them as a parameter."""

import pytest

from supermarket.catalogue import build_sample_inventory
from supermarket.checkout import Checkout
from supermarket.inventory import Inventory


@pytest.fixture
def inventory() -> Inventory:
    return build_sample_inventory()


@pytest.fixture
def checkout(inventory: Inventory) -> Checkout:
    return Checkout(inventory)
