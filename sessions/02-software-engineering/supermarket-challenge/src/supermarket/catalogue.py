"""The shelves of HTW Markt, used by the commands and the tests: eight products and their stock.

Change the products here to try your own shop. The tests expect these exact products and
quantities, so change them back before you run the tests.
"""

from supermarket.inventory import Inventory
from supermarket.products import BottledDrink, FoodProduct, NonFoodProduct


def build_sample_inventory() -> Inventory:
    """A new Inventory with the sample products.

    Every call builds a new, full shop. The tests use this, so that one test that takes items off
    the shelves cannot change the stock that another test sees.
    """
    inventory = Inventory()
    # Net prices in EUR (before VAT). The shelf prices are computed by gross_price().
    inventory.add_product(FoodProduct("Apples 1 kg", 2.29), quantity=20)
    inventory.add_product(FoodProduct("Bread", 2.80), quantity=8)
    inventory.add_product(FoodProduct("Milk 1 l", 1.09), quantity=30)
    inventory.add_product(FoodProduct("Cheese", 3.49), quantity=4)
    inventory.add_product(BottledDrink("Mineral water", 0.49), quantity=48)
    inventory.add_product(BottledDrink("Cola", 1.09), quantity=3)
    inventory.add_product(NonFoodProduct("Dish soap", 1.95), quantity=12)
    inventory.add_product(NonFoodProduct("Toothpaste", 1.49), quantity=2)
    return inventory
