"""The checkout counter, where one customer pays.

A Checkout brings the other parts together (composition): it has the catalogue to find products
in, and a ShoppingCart to put them in.
"""

from supermarket.cart import ShoppingCart
from supermarket.products import Product


class SupermarketError(Exception):
    """Base class of every error that the shop raises on purpose: `except SupermarketError` catches all."""


class UnknownProductError(SupermarketError):
    """Raised when the shop does not sell a product with the name that was asked for."""


class PaymentError(SupermarketError):
    """Raised when a customer hands over less money than they must pay."""


# Task 2, step 1: write the class ExpiredProductError here, as a subclass of SupermarketError.


class Checkout:
    """One customer at the checkout counter.

    Attributes:
        catalogue: The products the shop sells, by name: {"Bread": FoodProduct("Bread", 2.8), ...}.
        cart: The customer's ShoppingCart; a new Checkout starts with an empty one.
    """

    def __init__(self, catalogue: dict[str, Product]) -> None:
        self.catalogue = catalogue
        self.cart = ShoppingCart()

    def scan(self, product_name: str, quantity: int = 1) -> None:
        """Scan a product: find it in the catalogue and put it in the cart.

        Raises:
            UnknownProductError: If the shop does not sell this product.
            ExpiredProductError: If the product cannot be sold today (Task 2).
        """
        if product_name not in self.catalogue:
            raise UnknownProductError(f"the shop does not sell {product_name!r}")
        product = self.catalogue[product_name]
        # Task 2, step 3: if the product is not sellable, raise ExpiredProductError here,
        # with a message that names the product. The product must not reach the cart.
        self.cart.add(product, quantity)

    def total(self) -> float:
        """What the customer pays, in EUR: the sum of the shelf prices of everything in the cart."""
        return self.cart.subtotal()

    def pay(self, amount_given: float) -> float:
        """Task 3: the customer pays; return the change.

        Args:
            amount_given: The money the customer hands over, in EUR.

        Returns:
            The change in EUR, rounded to cents: amount_given minus total().

        Raises:
            PaymentError: If amount_given is less than total().
        """
        raise NotImplementedError("Task 3")

    def receipt(self) -> str:
        """The receipt as text: one row per product, then the total.

        Example of one row: "  2 x Cheese sandwich       2.25     4.50"
        """
        rows = [f"{'Qty':>3}   {'Product':<20}{'Price':>6}{'Amount':>9}"]
        for product_name, quantity in self.cart.quantities().items():
            shelf_price = self.catalogue[product_name].gross_price()
            # :>3 right-aligns in 3 characters, :<20 left-aligns in 20, :>9.2f gives 2 decimals.
            rows.append(
                f"{quantity:>3} x {product_name:<20}{shelf_price:>6.2f}{shelf_price * quantity:>9.2f}"
            )
        rows.append("-" * 41)
        rows.append(f"{'Total EUR':<32}{self.total():>9.2f}")
        return "\n".join(rows)  # join the rows with line breaks into one text
