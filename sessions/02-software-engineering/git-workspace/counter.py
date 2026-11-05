"""A class that counts: the file the owner changes in step 2."""


class Counter:
    """Counts up from zero."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        """Add one to the count."""
        self.value += 1
