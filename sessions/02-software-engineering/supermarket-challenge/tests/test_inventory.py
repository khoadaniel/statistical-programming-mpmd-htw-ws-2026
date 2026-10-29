"""Tests of inventory.py: taking items off the shelves. Task 2 and Task 3 tests are skipped.

Each test is a function whose name starts with test_. pytest finds and runs them all. The
parameter `inventory` or `checkout` is a fixture from conftest.py: pytest builds a fresh sample
shop for each test. A line @pytest.mark.skip(...) above a test switches it off until you
start that task; delete the line to switch the test on.
"""

import pytest

from supermarket.errors import OutOfStockError, UnknownProductError
from supermarket.inventory import Inventory


def test_take_lowers_the_quantity(inventory: Inventory):
    inventory.take("Bread", 3)
    assert inventory.quantity_of("Bread") == 5


def test_take_refuses_more_than_is_in_stock(inventory: Inventory):
    with pytest.raises(OutOfStockError):
        inventory.take("Toothpaste", 3)
    assert inventory.quantity_of("Toothpaste") == 2  # nothing was taken


def test_an_unknown_product_is_reported(inventory: Inventory):
    with pytest.raises(UnknownProductError):
        inventory.find("Caviar")


def test_product_names_are_alphabetical(inventory: Inventory):
    assert inventory.product_names()[:3] == ["Apples 1 kg", "Bread", "Cheese"]


@pytest.mark.skip(reason="Task 2: delete this line when you start the task")
def test_low_stock_lists_products_below_the_threshold(inventory: Inventory):
    assert inventory.low_stock() == ["Cheese", "Cola", "Toothpaste"]
    assert inventory.low_stock(threshold=3) == ["Toothpaste"]


@pytest.mark.skip(reason="Task 2: delete this line when you start the task")
def test_low_stock_of_an_empty_inventory_is_an_empty_list():
    assert Inventory().low_stock() == []


@pytest.mark.skip(reason="Task 3: delete this line when you start the task")
def test_restock_raises_the_quantity(inventory: Inventory):
    inventory.restock("Toothpaste", 10)
    assert inventory.quantity_of("Toothpaste") == 12


@pytest.mark.skip(reason="Task 3: delete this line when you start the task")
@pytest.mark.parametrize("quantity", [0, -5])
def test_restock_refuses_a_quantity_below_one(inventory: Inventory, quantity):
    with pytest.raises(ValueError):
        inventory.restock("Toothpaste", quantity)


@pytest.mark.skip(reason="Task 3: delete this line when you start the task")
def test_restock_refuses_an_unknown_product(inventory: Inventory):
    with pytest.raises(UnknownProductError):
        inventory.restock("Caviar", 10)
