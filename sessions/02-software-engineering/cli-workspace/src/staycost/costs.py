"""The calculation itself: no printing, no arguments from the terminal, so it is easy to import and test."""

CLEANING_FEE = 25.0  # EUR, charged once per stay


def stay_cost(nightly_price: float, nights: int = 1, cleaning_fee: float = CLEANING_FEE) -> float:
    """Total price of a stay: nights times the nightly price, plus the cleaning fee.

    >>> stay_cost(80, 3)
    265.0
    """
    if nightly_price < 0 or cleaning_fee < 0:
        raise ValueError("prices cannot be negative")
    if nights < 1:
        raise ValueError("a stay has at least one night")
    return float(nightly_price * nights + cleaning_fee)
