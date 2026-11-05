"""Uses the three classes: run it with python3 main.py after every step."""
from counter import Counter
from greeter import Greeter
from listing import Listing


def main() -> None:
    """Print one line from each class."""
    print(Greeter("Ada").greet())
    clicks = Counter()
    clicks.increment()
    clicks.increment()
    print(f"Counter: {clicks.value}")
    print(Listing("Kreuzberg loft", 80).describe())


if __name__ == "__main__":
    main()
