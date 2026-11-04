"""Tests of products.py: VAT, deposit and the price rule.

Tests marked @pytest.mark.task(N) check your work on Task N. Until you start the task, pytest
reports them as "not started yet".
"""

import pytest

from supermarket.products import BottledDrink, FoodProduct, NonFoodProduct, Product


def test_food_is_taxed_at_the_reduced_rate():
    assert FoodProduct("Bread", 2.80).gross_price() == 3.00


def test_non_food_is_taxed_at_the_standard_rate():
    assert NonFoodProduct("Dish soap", 1.95).gross_price() == 2.32


def test_a_bottled_drink_adds_the_deposit():
    assert BottledDrink("Mineral water", 0.49).gross_price() == 0.83


def test_each_kind_of_product_has_its_own_category():
    categories = [
        FoodProduct("Bread", 2.80).category,
        NonFoodProduct("Dish soap", 1.95).category,
        BottledDrink("Cola", 1.09).category,
    ]
    assert categories == ["food", "non-food", "drinks"]


def test_a_plain_product_cannot_be_created():
    with pytest.raises(TypeError):
        Product("Mystery", 1.00)


def test_a_negative_price_is_refused_when_the_product_is_created():
    with pytest.raises(ValueError):
        FoodProduct("Bread", -1.00)


def test_a_negative_price_is_refused_after_creation_too():
    bread = FoodProduct("Bread", 2.80)
    with pytest.raises(ValueError):
        bread.net_price = -1.00


@pytest.mark.task(1)
def test_fresh_product_is_food():
    from supermarket.products import FreshProduct

    sandwich = FreshProduct("Cheese sandwich", 3.00, days_until_use_by=5)
    assert isinstance(sandwich, FoodProduct)
    assert sandwich.category == "food"
    assert sandwich.gross_price() == 3.21


@pytest.mark.task(1)
@pytest.mark.parametrize("days_until_use_by", [1, 0])
def test_fresh_product_costs_less_on_its_last_day(days_until_use_by):
    from supermarket.products import FreshProduct

    sandwich = FreshProduct("Cheese sandwich", 3.00, days_until_use_by)
    assert sandwich.gross_price() == 2.25


@pytest.mark.task(1)
def test_fresh_product_past_its_use_by_date_cannot_be_sold():
    from supermarket.products import FreshProduct

    sandwich = FreshProduct("Cheese sandwich", 3.00, days_until_use_by=-1)
    assert not sandwich.is_sellable()
    assert FreshProduct("Cheese sandwich", 3.00, days_until_use_by=0).is_sellable()
