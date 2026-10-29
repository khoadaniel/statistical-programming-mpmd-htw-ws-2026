"""Tests of checkout.py: scanning, totals, the receipt. Task 5 and Task 8 tests are skipped.

Each test is a function whose name starts with test_. pytest finds and runs them all. The
parameter `inventory` or `checkout` is a fixture from conftest.py: pytest builds a fresh sample
shop for each test. A line @pytest.mark.skip(...) above a test switches it off until you
start that task; delete the line to switch the test on.
"""

import pytest

from supermarket.checkout import Checkout
from supermarket.discounts import PercentageDiscount
from supermarket.errors import OutOfStockError, SupermarketError
from supermarket.inventory import Inventory


def test_scan_moves_items_from_the_shelf_to_the_cart(checkout: Checkout):
    checkout.scan("Bread", quantity=2)
    assert checkout.cart.item_count() == 2
    assert checkout.inventory.quantity_of("Bread") == 6


def test_scan_more_than_is_in_stock(checkout: Checkout):
    with pytest.raises(OutOfStockError):
        checkout.scan("Cola", quantity=4)


def test_every_error_of_the_package_is_a_supermarket_error(checkout: Checkout):
    with pytest.raises(SupermarketError):
        checkout.scan("Caviar")


def test_total_after_a_percentage_discount(inventory: Inventory):
    checkout = Checkout(inventory, discount=PercentageDiscount(10))
    checkout.scan("Bread", quantity=2)  # 2 x 3.00
    assert checkout.total() == 5.40


def test_receipt_ends_with_the_total(checkout: Checkout):
    checkout.scan("Bread")
    assert checkout.receipt().splitlines()[-1].endswith("3.00")


@pytest.mark.skip(reason="Task 5: delete this line when you start the task")
@pytest.mark.parametrize(
    "has_member_card, bread_count, expected_points",
    [
        (False, 8, 0),  # no card: no points, whatever the total
        (True, 5, 15),  # total 15.00: 1 point per full euro
        (True, 8, 24),  # total 24.00: 1 point per full euro
        (True, 1, 3),  # total 3.00: 1 point per full euro
    ],
)
def test_loyalty_points(inventory: Inventory, has_member_card, bread_count, expected_points):
    checkout = Checkout(inventory, has_member_card=has_member_card)
    checkout.scan("Bread", quantity=bread_count)
    assert checkout.loyalty_points() == expected_points


@pytest.mark.skip(reason="Task 5: delete this line when you start the task")
def test_loyalty_points_double_from_fifty_euros(inventory: Inventory):
    checkout = Checkout(inventory, has_member_card=True)
    checkout.scan("Cheese", quantity=4)  # 14.92
    checkout.scan("Bread", quantity=8)  # 24.00
    checkout.scan("Apples 1 kg", quantity=5)  # 12.25 -> total 51.17
    assert checkout.loyalty_points() == 102


@pytest.mark.skip(reason="Task 5: delete this line when you start the task")
def test_loyalty_points_count_the_total_after_the_discount(inventory: Inventory):
    checkout = Checkout(inventory, discount=PercentageDiscount(50), has_member_card=True)
    checkout.scan("Bread", quantity=5)  # 15.00 -> 7.50 after the discount
    assert checkout.loyalty_points() == 7


@pytest.mark.skip(reason="Task 8: delete this line when you start the task")
def test_pay_returns_the_change(checkout: Checkout):
    checkout.scan("Bread", quantity=2)  # 6.00
    assert checkout.pay(10.00) == 4.00
    assert checkout.pay(6.00) == 0.00


@pytest.mark.skip(reason="Task 8: delete this line when you start the task")
def test_pay_refuses_too_little_money(checkout: Checkout):
    from supermarket.errors import PaymentError

    checkout.scan("Bread", quantity=2)  # 6.00
    with pytest.raises(PaymentError):
        checkout.pay(5.00)
    assert issubclass(PaymentError, SupermarketError)
