"""The shopping cart: the items a customer has chosen, in a list."""

from supermarket.products import Product


class ShoppingCart:
    """The items a customer has chosen, one entry per item.

    Three bottles of water are three entries, all three the same BottledDrink object.
    """

    def __init__(self) -> None:
        self._items: list[Product] = []  # a new cart is empty

    def add(self, product: Product, quantity: int = 1) -> None:
        """Put `quantity` items of a product in the cart.

        Raises:
            ValueError: If quantity is below 1.
        """
        if quantity < 1:
            raise ValueError("add at least one item")
        for _ in range(quantity):  # `_` is the name for a loop variable that is not used
            self._items.append(product)

    def subtotal(self) -> float:
        """The sum of the shelf prices of all items in EUR, before any discount."""
        return round(sum(product.gross_price() for product in self._items), 2)

    def quantities(self) -> dict[str, int]:
        """How many items of each product are in the cart, for example {"Bread": 1, "Milk 1 l": 2}."""
        quantity_by_name: dict[str, int] = {}
        for product in self._items:
            quantity_by_name[product.name] = quantity_by_name.get(product.name, 0) + 1
        return quantity_by_name
