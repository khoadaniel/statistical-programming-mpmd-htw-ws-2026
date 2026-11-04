"""The checkout counter, where one customer pays.

A Checkout brings the other parts together (composition): it has the catalogue to find products
in, a ShoppingCart to put them in, and a Discount to lower the total.
"""

from supermarket.cart import ShoppingCart
from supermarket.discounts import Discount, NoDiscount
from supermarket.products import Product


class SupermarketError(Exception):
    """Base class of every error that the shop raises on purpose: `except SupermarketError` catches all."""


class UnknownProductError(SupermarketError):
    """Raised when the shop does not sell a product with the name that was asked for."""


# Task 3, step 1: write the class PaymentError here, as a subclass of SupermarketError.


class Checkout:
    """One customer at the checkout counter.

    Attributes:
        catalogue: The products the shop sells, by name: {"Bread": FoodProduct("Bread", 2.8), ...}.
        cart: The customer's ShoppingCart; a new Checkout starts with an empty one.
        discount: The customer's Discount; NoDiscount() if none is given.
    """

    def __init__(self, catalogue: dict[str, Product], discount: Discount | None = None) -> None:
        self.catalogue = catalogue
        self.cart = ShoppingCart()
        # No discount given: use NoDiscount(), so that total() works the same way for everyone.
        self.discount = discount if discount is not None else NoDiscount()

    def scan(self, product_name: str, quantity: int = 1) -> None:
        """Scan a product: find it in the catalogue and put it in the cart.

        Raises:
            UnknownProductError: If the shop does not sell this product.
        """
        if product_name not in self.catalogue:
            raise UnknownProductError(f"the shop does not sell {product_name!r}")
        self.cart.add(self.catalogue[product_name], quantity)

    def subtotal(self) -> float:
        """The price of everything in the cart, in EUR, before the discount."""
        return self.cart.subtotal()

    def total(self) -> float:
        """What the customer pays, in EUR: the subtotal after the discount.

        The checkout does not know which kind of discount it has. It calls apply(), and the
        discount object runs its own amount() (polymorphism).
        """
        return self.discount.apply(self.subtotal())

    def pay(self, amount_given: float) -> float:
        """Task 3, step 2: the customer pays; return the change.

        Args:
            amount_given: The money the customer hands over, in EUR.

        Returns:
            The change in EUR, rounded to cents: amount_given minus total().

        Raises:
            PaymentError: If amount_given is less than total().
        """
        raise NotImplementedError("Task 3")

    def receipt(self) -> str:
        """The receipt as text: one row per product, then subtotal, discount and total.

        Example of one row: "  2 x Milk 1 l              1.17     2.34"
        """
        rows = [f"{'Qty':>3}   {'Product':<20}{'Price':>6}{'Amount':>9}"]
        for product_name, quantity in self.cart.quantities().items():
            shelf_price = self.catalogue[product_name].gross_price()
            # :>3 right-aligns in 3 characters, :<20 left-aligns in 20, :>9.2f gives 2 decimals.
            rows.append(
                f"{quantity:>3} x {product_name:<20}{shelf_price:>6.2f}{shelf_price * quantity:>9.2f}"
            )
        rows.append("-" * 41)
        rows.append(f"{'Subtotal':<32}{self.subtotal():>9.2f}")
        rows.append(f"{'Discount':<32}{self.total() - self.subtotal():>9.2f}")
        rows.append(f"{'Total EUR':<32}{self.total():>9.2f}")
        return "\n".join(rows)  # join the rows with line breaks into one text
