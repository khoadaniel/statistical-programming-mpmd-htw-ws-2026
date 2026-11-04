"""Tests of checkout.py: scanning, totals, the receipt and the payment.

Tests marked @pytest.mark.task(N) check your work on Task N. Until you start the task, pytest
reports them as "not started yet".
"""

import pytest

from supermarket.checkout import Checkout, UnknownProductError
from supermarket.discounts import PercentageDiscount
from supermarket.products import Product


def test_scan_puts_the_items_in_the_cart(checkout: Checkout):
    checkout.scan("Bread", quantity=2)
    assert checkout.cart.quantities() == {"Bread": 2}


def test_an_unknown_product_is_refused(checkout: Checkout):
    with pytest.raises(UnknownProductError):
        checkout.scan("Caviar")


def test_total_after_a_percentage_discount(catalogue: dict[str, Product]):
    checkout = Checkout(catalogue, discount=PercentageDiscount(10))
    checkout.scan("Bread", quantity=2)  # 2 x 3.00
    assert checkout.total() == 5.40


def test_receipt_ends_with_the_total(checkout: Checkout):
    checkout.scan("Bread")
    assert checkout.receipt().splitlines()[-1].endswith("3.00")


@pytest.mark.task(3)
def test_pay_returns_the_change(checkout: Checkout):
    checkout.scan("Bread", quantity=2)  # 6.00
    assert checkout.pay(10.00) == 4.00
    assert checkout.pay(6.00) == 0.00


@pytest.mark.task(3)
def test_too_little_money_is_refused(checkout: Checkout):
    from supermarket.checkout import PaymentError

    checkout.scan("Bread", quantity=2)
    with pytest.raises(PaymentError, match="6.00"):
        checkout.pay(5.00)


@pytest.mark.task(3)
def test_a_payment_error_is_a_supermarket_error():
    from supermarket.checkout import PaymentError, SupermarketError

    assert issubclass(PaymentError, SupermarketError)
