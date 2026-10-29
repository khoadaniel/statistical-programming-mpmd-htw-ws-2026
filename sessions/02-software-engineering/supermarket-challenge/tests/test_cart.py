"""Tests of cart.py: adding items, counting them, the subtotal. Task 7 tests are skipped.

Each test is a function whose name starts with test_. pytest finds and runs them all. The
parameter `inventory` or `checkout` is a fixture from conftest.py: pytest builds a fresh sample
shop for each test. A line @pytest.mark.skip(...) above a test switches it off until you
start that task; delete the line to switch the test on.
"""

import pytest

from supermarket.cart import ShoppingCart
from supermarket.products import BottledDrink, FoodProduct, NonFoodProduct


def test_each_item_is_one_entry_in_the_cart():
    cart = ShoppingCart()
    bread = FoodProduct("Bread", 2.80)
    cart.add(bread)
    cart.add(bread, quantity=2)
    assert cart.item_count() == 3
    assert cart.items() == [bread, bread, bread]


def test_quantities_count_the_items_of_each_product():
    cart = ShoppingCart()
    cart.add(FoodProduct("Bread", 2.80))
    cart.add(FoodProduct("Milk 1 l", 1.09), quantity=2)
    assert cart.quantities() == {"Bread": 1, "Milk 1 l": 2}


def test_subtotal_adds_all_items():
    cart = ShoppingCart()
    cart.add(FoodProduct("Bread", 2.80))  # 3.00
    cart.add(BottledDrink("Mineral water", 0.49), quantity=6)  # 6 x 0.83
    assert cart.subtotal() == 7.98


def test_adding_zero_items_is_refused():
    with pytest.raises(ValueError):
        ShoppingCart().add(FoodProduct("Bread", 2.80), quantity=0)


@pytest.mark.skip(reason="Task 7: delete this line when you start the task")
def test_total_by_category():
    cart = ShoppingCart()
    cart.add(FoodProduct("Bread", 2.80))  # 3.00
    cart.add(FoodProduct("Milk 1 l", 1.09), quantity=2)  # 2.34
    cart.add(NonFoodProduct("Dish soap", 1.95))  # 2.32
    assert cart.total_by_category() == {"food": 5.34, "non-food": 2.32}


@pytest.mark.skip(reason="Task 7: delete this line when you start the task")
def test_total_by_category_of_an_empty_cart():
    assert ShoppingCart().total_by_category() == {}
