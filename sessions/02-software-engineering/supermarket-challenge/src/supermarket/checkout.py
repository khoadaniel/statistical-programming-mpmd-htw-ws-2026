"""The checkout counter, where one customer pays.

A Checkout brings the other parts together (composition): it has an Inventory to take products
from, a ShoppingCart to put them in, and a Discount to lower the total.
"""

from supermarket.cart import ShoppingCart
from supermarket.discounts import Discount, NoDiscount
from supermarket.inventory import Inventory


class Checkout:
    """One customer at the checkout counter.

    Attributes:
        inventory: The shop's stock. scan() takes the items out of it.
        cart: The customer's ShoppingCart; a new Checkout starts with an empty one.
        discount: The customer's Discount; NoDiscount() if none is given.
        has_member_card: True if the customer has a member card and collects loyalty points.
    """

    def __init__(
        self,
        inventory: Inventory,
        discount: Discount | None = None,
        has_member_card: bool = False,
    ) -> None:
        self.inventory = inventory
        self.cart = ShoppingCart()
        # No discount given: use NoDiscount(), so that total() works the same way for everyone.
        self.discount = discount if discount is not None else NoDiscount()
        self.has_member_card = has_member_card

    def scan(self, product_name: str, quantity: int = 1) -> None:
        """Scan a product: take the items off the shelf and put them in the cart.

        Raises:
            UnknownProductError: If the shop does not sell this product.
            OutOfStockError: If fewer items are in stock. Nothing goes into the cart then.
        """
        product = self.inventory.take(product_name, quantity)
        self.cart.add(product, quantity)

    def subtotal(self) -> float:
        """The price of everything in the cart, in EUR, before the discount."""
        return self.cart.subtotal()

    def total(self) -> float:
        """What the customer pays, in EUR: the subtotal after the discount.

        The checkout does not know which kind of discount it has. It calls apply(), and the
        discount object runs its own amount() (polymorphism).
        """
        return self.discount.apply(self.subtotal())

    def loyalty_points(self) -> int:
        """Task 5: the points the customer collects on the member card.

        Rules (they use total(), the amount after the discount):
            no member card              0 points
            member card, below 50 EUR   1 point per full euro
            member card, 50 EUR or more 2 points per full euro

        Example:
            A member who pays 24.80 EUR gets 24 points; one who pays 51.17 EUR gets 102.

        Hint:
            Write an `if` inside an `if`: first ask about the card, then about the total.
            int(24.80) is 24, the number of full euros.
        """
        raise NotImplementedError("Task 5")

    def pay(self, amount_given: float) -> float:
        """Task 8: the customer pays; return the change.

        Args:
            amount_given: The money the customer hands over, in EUR.

        Returns:
            The change in EUR, rounded to cents: amount_given minus total().

        Raises:
            PaymentError: If amount_given is less than total(). Write this exception in errors.py
                and import it at the top of this file.
        """
        raise NotImplementedError("Task 8")

    def receipt(self) -> str:
        """The receipt as text: one row per product, then subtotal, discount and total.

        Each row shows the quantity, the product, the shelf price of one item (net price plus VAT,
        plus a deposit for drinks) and the amount: quantity times shelf price.

        Example of one row: "  2 x Milk 1 l              1.17     2.34"
        """
        receipt_rows = [f"{'Qty':>3}   {'Product':<20}{'Price':>6}{'Amount':>9}"]
        for product_name, quantity in self.cart.quantities().items():
            product = self.inventory.find(product_name)
            shelf_price = product.gross_price()
            row_amount = shelf_price * quantity
            # :>3 right-aligns in 3 characters, :<20 left-aligns in 20, :>9.2f gives 2 decimals.
            receipt_rows.append(
                f"{quantity:>3} x {product_name:<20}{shelf_price:>6.2f}{row_amount:>9.2f}"
            )
        discount_amount = self.subtotal() - self.total()
        receipt_rows.append("-" * 41)
        receipt_rows.append(f"{'Subtotal':<32}{self.subtotal():>9.2f}")
        receipt_rows.append(f"{'Discount':<32}{-discount_amount:>9.2f}")
        receipt_rows.append(f"{'Total EUR':<32}{self.total():>9.2f}")
        return "\n".join(receipt_rows)  # join the rows with line breaks into one text
