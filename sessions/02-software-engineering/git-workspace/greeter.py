"""A class that greets one person: the file both of you change in step 3."""


class Greeter:
    """Greets one person by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        """Return the greeting for this person."""
        return f"Hello, {self.name}!"
