"""Tests of products.py: VAT, deposit, categories and the price rule. Task 4 tests are skipped.

Each test is a function whose name starts with test_. pytest finds and runs them all. The
parameter `inventory` or `checkout` is a fixture from conftest.py: pytest builds a fresh sample
shop for each test. A line @pytest.mark.skip(...) above a test switches it off until you
start that task; delete the line to switch the test on.
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


@pytest.mark.skip(reason="Task 4: delete this line when you start the task")
def test_fresh_product_is_food():
    from supermarket.products import FreshProduct

    strawberries = FreshProduct("Strawberries", 3.00, days_until_best_before=5)
    assert isinstance(strawberries, FoodProduct)
    assert strawberries.category == "food"
    assert strawberries.gross_price() == 3.21


@pytest.mark.skip(reason="Task 4: delete this line when you start the task")
@pytest.mark.parametrize("days_until_best_before", [1, 0])
def test_fresh_product_costs_less_on_its_last_day(days_until_best_before):
    from supermarket.products import FreshProduct

    strawberries = FreshProduct("Strawberries", 3.00, days_until_best_before)
    assert strawberries.gross_price() == 2.25


@pytest.mark.skip(reason="Task 4: delete this line when you start the task")
def test_fresh_product_past_its_best_before_date_cannot_be_sold():
    from supermarket.products import FreshProduct

    strawberries = FreshProduct("Strawberries", 3.00, days_until_best_before=-1)
    assert not strawberries.is_sellable()
    assert FreshProduct("Strawberries", 3.00, days_until_best_before=0).is_sellable()
