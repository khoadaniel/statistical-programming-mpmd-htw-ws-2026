"""The products a supermarket sells.

Product is an abstract base class: it holds what all products share (a name, a net price and how
the shelf price is computed). Each subclass adds what is special about its kind of product, such as
its VAT rate or a bottle deposit.

    Product (abstract)
    ├── FoodProduct       7 % VAT
    ├── NonFoodProduct   19 % VAT
    └── BottledDrink     19 % VAT, plus the deposit
"""

from abc import ABC, abstractmethod

# German VAT rates (Umsatzsteuergesetz, section 12): most food is taxed at the reduced rate.
REDUCED_VAT_RATE = 0.07
STANDARD_VAT_RATE = 0.19
BOTTLE_DEPOSIT = 0.25  # EUR per single-use bottle (Pfand)


class Product(ABC):
    """Something on the shelf, with a name and a net price (the price before VAT).

    Product is abstract (it inherits from ABC and has an @abstractmethod), so
    `Product("Tea", 1.00)` raises TypeError. Create an object of a subclass instead, for example
    `FoodProduct("Tea", 1.00)`.

    Attributes:
        name: The name on the shelf label, for example "Bread". The inventory finds products by it.
        net_price: The price in EUR before VAT. It can never be negative (see the setter below).
        category: A class attribute, the same for all products of one class, for example "food".
    """

    category = "uncategorised"  # class attribute: each subclass sets its own

    def __init__(self, name: str, net_price: float) -> None:
        self.name = name
        # This line does not store the price directly: it calls the setter `net_price` below,
        # so the rule "not negative" is checked when the product is created, too.
        self.net_price = net_price

    # Encapsulation: the price is stored in `_net_price` (leading underscore = internal).
    # Outside code reads and writes `product.net_price`; Python then runs these two methods.
    @property
    def net_price(self) -> float:
        """The price in EUR before VAT."""
        return self._net_price

    @net_price.setter
    def net_price(self, new_net_price: float) -> None:
        if new_net_price < 0:
            raise ValueError(f"the price of {self.name!r} cannot be negative")
        self._net_price = new_net_price

    @abstractmethod
    def vat_rate(self) -> float:
        """The VAT rate as a fraction, for example 0.19 for 19 %.

        Abstract: there is no body here. Every subclass must write this method, otherwise Python
        refuses to create objects of the subclass.
        """

    def gross_price(self) -> float:
        """The shelf price in EUR: the net price plus VAT, rounded to cents.

        Written once, here, for all products. It calls `self.vat_rate()`, so each subclass gets
        its own result without writing gross_price again (polymorphism).

        Example:
            FoodProduct("Bread", 2.80).gross_price() is 2.80 * 1.07 = 2.996, rounded 3.00.
        """
        return round(self.net_price * (1 + self.vat_rate()), 2)

    def __repr__(self) -> str:
        # What print() and the notebook show for a product, for example FoodProduct('Bread', 2.8).
        return f"{type(self).__name__}({self.name!r}, {self.net_price})"


class FoodProduct(Product):
    """Food, taxed at the reduced VAT rate of 7 %."""

    category = "food"

    def vat_rate(self) -> float:
        return REDUCED_VAT_RATE


class NonFoodProduct(Product):
    """Everything that is neither food nor a drink, such as soap or batteries: 19 % VAT."""

    category = "non-food"

    def vat_rate(self) -> float:
        return STANDARD_VAT_RATE


class BottledDrink(Product):
    """A drink in a deposit bottle. The deposit is added to the shelf price.

    Attributes:
        deposit: The deposit in EUR, paid back when the bottle is returned. Default 0.25.
    """

    category = "drinks"

    def __init__(self, name: str, net_price: float, deposit: float = BOTTLE_DEPOSIT) -> None:
        # Inheritance: the parent's constructor sets name and net_price (and checks the price);
        # this class adds only what is new, the deposit.
        super().__init__(name, net_price)
        self.deposit = deposit

    def vat_rate(self) -> float:
        return STANDARD_VAT_RATE

    def gross_price(self) -> float:
        """The parent's shelf price (net price plus VAT), plus the deposit.

        Example:
            BottledDrink("Mineral water", 0.49).gross_price() is 0.58 + 0.25 = 0.83.
        """
        return round(super().gross_price() + self.deposit, 2)


# Task 4: write the class FreshProduct here.
