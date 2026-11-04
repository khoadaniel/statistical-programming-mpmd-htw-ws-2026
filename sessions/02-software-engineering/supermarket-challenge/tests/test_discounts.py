"""Tests of discounts.py, and of a Checkout with a discount.

Tests marked @pytest.mark.task(N) check your work on Task N. Until you start the task, pytest
reports them as "not started yet".
"""

import pytest

from supermarket.checkout import Checkout
from supermarket.discounts import NoDiscount, PercentageDiscount
from supermarket.products import Product


def test_no_discount_keeps_the_price():
    assert NoDiscount().apply(20.00) == 20.00


def test_ten_percent_off():
    assert PercentageDiscount(10).apply(20.00) == 18.00


@pytest.mark.parametrize("percent", [0, -5, 120])
def test_an_impossible_percentage_is_refused(percent):
    with pytest.raises(ValueError):
        PercentageDiscount(percent)


@pytest.mark.task(2)
@pytest.mark.parametrize("subtotal, expected_total", [(30.00, 25.00), (29.99, 29.99), (3.00, 3.00)])
def test_threshold_discount(subtotal, expected_total):
    from supermarket.discounts import ThresholdDiscount

    assert ThresholdDiscount(minimum_subtotal=30, amount_off=5).apply(subtotal) == expected_total


@pytest.mark.task(2)
def test_the_checkout_works_with_a_threshold_discount(catalogue: dict[str, Product]):
    from supermarket.discounts import ThresholdDiscount

    checkout = Checkout(catalogue, discount=ThresholdDiscount(minimum_subtotal=10, amount_off=2))
    checkout.scan("Cheese", quantity=4)  # 4 x 3.73 = 14.92
    assert checkout.total() == 12.92
