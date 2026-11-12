"""Tests of checkout.py: scanning, the total, the receipt and the payment.

Tests marked @pytest.mark.task(N) check your work on Task N. Until you start the task, pytest
reports them as "not started yet".
"""

import pytest

from supermarket.checkout import Checkout, PaymentError, SupermarketError, UnknownProductError
from supermarket.products import Product


def test_scan_puts_the_items_in_the_cart(checkout: Checkout):
    checkout.scan("Bread", quantity=2)
    assert checkout.cart.quantities() == {"Bread": 2}


def test_an_unknown_product_is_refused(checkout: Checkout):
    with pytest.raises(UnknownProductError):
        checkout.scan("Caviar")


def test_total_adds_up_the_shelf_prices(checkout: Checkout):
    checkout.scan("Bread", quantity=2)  # 2 x 3.00
    checkout.scan("Mineral water")  # 0.83
    assert checkout.total() == 6.83


def test_receipt_ends_with_the_total(checkout: Checkout):
    checkout.scan("Bread")
    assert checkout.receipt().splitlines()[-1].endswith("3.00")


@pytest.mark.task(2)
def test_a_sandwich_past_its_use_by_date_is_refused(catalogue: dict[str, Product]):
    from supermarket.checkout import ExpiredProductError
    from supermarket.products import FreshProduct

    catalogue["Egg sandwich"] = FreshProduct("Egg sandwich", 2.60, days_until_use_by=-1)
    checkout = Checkout(catalogue)
    with pytest.raises(ExpiredProductError, match="Egg sandwich"):
        checkout.scan("Egg sandwich")
    assert checkout.cart.quantities() == {}  # the sandwich did not reach the cart


@pytest.mark.task(2)
def test_a_sandwich_on_its_use_by_day_is_sold(catalogue: dict[str, Product]):
    from supermarket.products import FreshProduct

    catalogue["Cheese sandwich"] = FreshProduct("Cheese sandwich", 3.00, days_until_use_by=0)
    checkout = Checkout(catalogue)
    checkout.scan("Cheese sandwich", quantity=2)
    assert checkout.cart.quantities() == {"Cheese sandwich": 2}


@pytest.mark.task(2)
def test_an_expired_product_error_is_a_supermarket_error():
    from supermarket.checkout import ExpiredProductError

    assert issubclass(ExpiredProductError, SupermarketError)


@pytest.mark.task(3)
def test_pay_returns_the_change(checkout: Checkout):
    checkout.scan("Bread", quantity=2)  # 6.00
    assert checkout.pay(10.00) == 4.00
    assert checkout.pay(6.00) == 0.00


@pytest.mark.task(3)
def test_too_little_money_is_refused(checkout: Checkout):
    checkout.scan("Bread", quantity=2)  # 6.00
    with pytest.raises(PaymentError, match="6.00"):
        checkout.pay(5.00)
