"""supermarket: products, stock, a shopping cart and a checkout, written as plain Python classes.

This file makes the folder `supermarket` a package. It runs once, when the package is first
imported, and re-exports the main classes, so that users can write the short form
`from supermarket import Checkout` instead of `from supermarket.checkout import Checkout`.
"""

from supermarket.cart import ShoppingCart
from supermarket.checkout import Checkout
from supermarket.discounts import Discount, NoDiscount, PercentageDiscount
from supermarket.errors import OutOfStockError, SupermarketError, UnknownProductError
from supermarket.inventory import Inventory
from supermarket.products import BottledDrink, FoodProduct, NonFoodProduct, Product

__all__ = [
    "BottledDrink",
    "Checkout",
    "Discount",
    "FoodProduct",
    "Inventory",
    "NoDiscount",
    "NonFoodProduct",
    "OutOfStockError",
    "PercentageDiscount",
    "Product",
    "ShoppingCart",
    "SupermarketError",
    "UnknownProductError",
]
