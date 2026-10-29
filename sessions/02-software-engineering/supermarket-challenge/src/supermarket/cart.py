"""The shopping cart: the items a customer has chosen, in a list.

The cart holds one entry per item, as a real cart does: three bottles of water are three entries,
all three the same BottledDrink object. The Checkout fills the cart when the cashier scans products.
"""

from supermarket.products import Product


class ShoppingCart:
    """The items a customer has chosen, kept in the list `_items` (internal).

    Example after adding one bread and two milk:

        _items  [FoodProduct('Bread', 2.8),
                 FoodProduct('Milk 1 l', 1.09),
                 FoodProduct('Milk 1 l', 1.09)]
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

    def items(self) -> list[Product]:
        """All items in the cart, one entry per item.

        Returns a copy of the list: a caller who changes the copy does not change the cart.
        """
        return list(self._items)

    def item_count(self) -> int:
        """The number of items in the cart: 3 apples and 2 bottles are 5 items."""
        return len(self._items)

    def subtotal(self) -> float:
        """The sum of the shelf prices of all items in EUR, before any discount."""
        return round(sum(product.gross_price() for product in self._items), 2)

    def quantities(self) -> dict[str, int]:
        """How many items of each product are in the cart.

        Example:
            {"Bread": 1, "Milk 1 l": 2} for one bread and two milk.
        """
        quantity_by_name: dict[str, int] = {}
        for product in self._items:
            if product.name in quantity_by_name:
                quantity_by_name[product.name] += 1  # seen before: count one more
            else:
                quantity_by_name[product.name] = 1  # first item of this product
        return quantity_by_name

    def total_by_category(self) -> dict[str, float]:
        """Task 7: the money spent per product category, in EUR, rounded to cents.

        Returns:
            A dictionary from category to money, for example {"food": 5.34, "non-food": 2.32}.
            Only categories that are in the cart appear. An empty cart gives {}.

        Hint:
            quantities() above uses the same pattern: it counts items, this method adds prices.
            Each product knows its category: `product.category`.
        """
        raise NotImplementedError("Task 7")
