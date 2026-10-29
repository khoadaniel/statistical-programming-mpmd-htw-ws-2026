"""The stock of the shop: which products it sells and how many items of each are on the shelves.

The Checkout uses the inventory: `take()` removes items when the cashier scans them.
"""

from supermarket.errors import OutOfStockError, UnknownProductError
from supermarket.products import Product


class Inventory:
    """The products of the shop and their quantities in stock.

    The data is kept in two dictionaries with the same keys, the product names:

        _products   {"Bread": FoodProduct('Bread', 2.8), "Cola": BottledDrink('Cola', 1.09), ...}
        _quantities {"Bread": 8, "Cola": 3, ...}

    Both names start with an underscore: they are internal. Other code changes the stock only
    through the methods below, which check every change, so a quantity never becomes negative.
    """

    def __init__(self) -> None:
        # A new inventory is empty; add_product() fills it (see catalogue.py).
        self._products: dict[str, Product] = {}
        self._quantities: dict[str, int] = {}

    def add_product(self, product: Product, quantity: int) -> None:
        """Start selling a product.

        Args:
            product: The product object, for example FoodProduct("Bread", 2.80).
            quantity: How many items are put on the shelf at the start.

        Raises:
            ValueError: If quantity is negative.
        """
        if quantity < 0:
            raise ValueError("a quantity in stock cannot be negative")
        self._products[product.name] = product
        self._quantities[product.name] = quantity

    def find(self, product_name: str) -> Product:
        """Look up a product by its name.

        Raises:
            UnknownProductError: If the shop does not sell a product with this name.
        """
        if product_name not in self._products:
            raise UnknownProductError(f"the shop does not sell {product_name!r}")
        return self._products[product_name]

    def quantity_of(self, product_name: str) -> int:
        """How many items of this product are in stock.

        Raises:
            UnknownProductError: If the shop does not sell a product with this name.
        """
        self.find(product_name)  # only for its check: raises UnknownProductError for a wrong name
        return self._quantities[product_name]

    def product_names(self) -> list[str]:
        """The names of all products, in alphabetical order."""
        return sorted(self._products)  # sorting a dictionary sorts its keys

    def take(self, product_name: str, quantity: int) -> Product:
        """Take items off the shelf, for example when the cashier scans them.

        Returns:
            The product object, so that the checkout can put it in the cart.

        Raises:
            UnknownProductError: If the shop does not sell a product with this name.
            OutOfStockError: If fewer than `quantity` items are in stock. Nothing is taken then.
        """
        product = self.find(product_name)
        quantity_in_stock = self._quantities[product_name]
        if quantity > quantity_in_stock:
            raise OutOfStockError(
                f"{quantity} x {product_name!r} asked for, only {quantity_in_stock} in stock"
            )
        self._quantities[product_name] = quantity_in_stock - quantity
        return product

    def low_stock(self, threshold: int = 5) -> list[str]:
        """Task 2: the products that should be reordered soon.

        Args:
            threshold: A product is low when fewer than this many items are in stock.

        Returns:
            The names of the low products, in alphabetical order. An empty list if none is low.

        Example:
            In the sample shop, low_stock() is ["Cheese", "Cola", "Toothpaste"] (4, 3 and 2 items).

        Hint:
            Loop over `self._quantities.items()`, which gives each name with its quantity.
        """
        raise NotImplementedError("Task 2")

    def restock(self, product_name: str, quantity: int) -> None:
        """Task 3: put more items of a product on the shelf.

        Args:
            product_name: A product the shop already sells.
            quantity: How many items are added; at least 1.

        Raises:
            ValueError: If quantity is below 1.
            UnknownProductError: If the shop does not sell this product (reuse `self.find`).
        """
        raise NotImplementedError("Task 3")
