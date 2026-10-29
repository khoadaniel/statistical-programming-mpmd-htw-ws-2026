"""Tests of discounts.py, and of a Checkout with a discount. Task 6 tests are skipped.

Each test is a function whose name starts with test_. pytest finds and runs them all. The
parameter `inventory` or `checkout` is a fixture from conftest.py: pytest builds a fresh sample
shop for each test. A line @pytest.mark.skip(...) above a test switches it off until you
start that task; delete the line to switch the test on.
"""

import pytest

from supermarket.checkout import Checkout
from supermarket.discounts import NoDiscount, PercentageDiscount
from supermarket.inventory import Inventory


def test_no_discount_keeps_the_price():
    assert NoDiscount().apply(20.00) == 20.00


def test_ten_percent_off():
    assert PercentageDiscount(10).apply(20.00) == 18.00


@pytest.mark.parametrize("percent", [0, -5, 120])
def test_an_impossible_percentage_is_refused(percent):
    with pytest.raises(ValueError):
        PercentageDiscount(percent)


@pytest.mark.skip(reason="Task 6: delete this line when you start the task")
@pytest.mark.parametrize("subtotal, expected_total", [(30.00, 25.00), (29.99, 29.99), (3.00, 3.00)])
def test_threshold_discount(subtotal, expected_total):
    from supermarket.discounts import ThresholdDiscount

    assert ThresholdDiscount(minimum_subtotal=30, amount_off=5).apply(subtotal) == expected_total


@pytest.mark.skip(reason="Task 6: delete this line when you start the task")
def test_the_checkout_works_with_a_threshold_discount(inventory: Inventory):
    from supermarket.discounts import ThresholdDiscount

    checkout = Checkout(inventory, discount=ThresholdDiscount(minimum_subtotal=10, amount_off=2))
    checkout.scan("Cheese", quantity=4)  # 4 x 3.73 = 14.92
    assert checkout.total() == 12.92
