"""A class for one rental listing: the file the collaborator changes in step 2."""


class Listing:
    """A place to rent, with its price per night in euros."""

    def __init__(self, name: str, price_per_night: int) -> None:
        self.name = name
        self.price_per_night = price_per_night

    def describe(self) -> str:
        """Return the listing as one line of text."""
        return f"{self.name}: {self.price_per_night} EUR per night"
