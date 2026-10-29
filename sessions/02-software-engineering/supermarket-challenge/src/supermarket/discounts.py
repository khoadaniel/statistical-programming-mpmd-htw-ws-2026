"""Discounts: the rules that lower the price at the checkout.

Discount is an abstract base class. The Checkout only ever calls `discount.apply(subtotal)`, so it
works with every kind of discount, including kinds that are written later (Task 6).

    Discount (abstract)
    ├── NoDiscount            nothing comes off
    └── PercentageDiscount    a share of the subtotal comes off, for example 10 %
"""

from abc import ABC, abstractmethod


class Discount(ABC):
    """A rule that lowers the price at the checkout.

    The work is split in two (abstraction):
        apply()   written once, here: subtracts the amount and makes sure the total is not negative.
        amount()  abstract: each subclass decides how many euros come off.
    """

    @abstractmethod
    def amount(self, subtotal: float) -> float:
        """How many EUR come off this subtotal. Every subclass must write this method."""

    def apply(self, subtotal: float) -> float:
        """The subtotal after the discount, in EUR, never below zero, rounded to cents.

        Example:
            PercentageDiscount(10).apply(20.00) is 20.00 - 2.00 = 18.00.
        """
        return round(max(subtotal - self.amount(subtotal), 0.0), 2)

    def __str__(self) -> str:
        """The text that print() shows: the class name, unless a subclass writes a better one."""
        return type(self).__name__


class NoDiscount(Discount):
    """The discount for a customer without one: nothing comes off.

    The Checkout uses it when no discount is given. Thanks to it, the Checkout never needs to ask
    `if self.discount is None:`; every checkout has a discount object that answers apply().
    """

    def amount(self, subtotal: float) -> float:
        return 0.0

    def __str__(self) -> str:
        return "none"


class PercentageDiscount(Discount):
    """A share of the subtotal comes off, for example 10 % on everything.

    Attributes:
        percent: The share in percent, above 0 and at most 100.
    """

    def __init__(self, percent: float) -> None:
        if not 0 < percent <= 100:
            raise ValueError("percent must be above 0 and at most 100")
        self.percent = percent

    def amount(self, subtotal: float) -> float:
        return subtotal * self.percent / 100

    def __str__(self) -> str:
        return f"{self.percent:g} % off"  # :g writes 10.0 as 10


# Task 6: write the class ThresholdDiscount here.
