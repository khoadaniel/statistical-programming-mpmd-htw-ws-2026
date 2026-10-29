"""The command line of HTW Markt: two commands that read YAML files and print the results.

    uv run htw-checkout ana                            # one customer: data/baskets/ana.yaml
    uv run htw-shelves                                 # the shelves: prices and stock
    uv run htw-shelves --delivery data/delivery.yaml   # the shelves after a delivery (Task 3)

This module only reads the input, calls the classes and prints. Every calculation happens in the
classes, so that the tests can check it without the command line.

Before a task is done, its method raises NotImplementedError. The commands catch it and print
"not available yet (Task N)", so you can run them at any time and watch your results appear.
"""

import argparse
from pathlib import Path

import yaml

from supermarket.catalogue import build_sample_inventory
from supermarket.checkout import Checkout
from supermarket.discounts import Discount, NoDiscount, PercentageDiscount
from supermarket.errors import SupermarketError

NOT_AVAILABLE = "not available yet (Task {task_number})"

# cli.py is in src/supermarket/, so three levels up (parents[2]) is the project folder.
BASKETS_FOLDER = Path(__file__).resolve().parents[2] / "data" / "baskets"


def load_yaml(yaml_path: Path) -> dict:
    """Read a YAML file into a dictionary: `Bread: 1` in the file becomes {"Bread": 1}."""
    with open(yaml_path, encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def find_basket(basket_name_or_path: str) -> Path | None:
    """The basket file for a name such as "ana", or for a path such as "my_baskets/eva.yaml".

    Returns:
        The path of the file, or None if there is no such file.
    """
    as_typed = Path(basket_name_or_path)
    if as_typed.is_file():  # a path to a file: use it as it is
        return as_typed
    by_name = BASKETS_FOLDER / f"{basket_name_or_path}.yaml"  # a name: look in data/baskets/
    if by_name.is_file():
        return by_name
    return None


def basket_names() -> list[str]:
    """The names of the baskets in data/baskets/, for example ["ana", "ben", "cem", "dana"]."""
    return sorted(basket_path.stem for basket_path in BASKETS_FOLDER.glob("*.yaml"))


def run_task(task_number: int, task_method, *arguments):
    """Call the method of a task with its arguments, or say that the task is not done yet.

    Example:
        run_task(5, checkout.loyalty_points) returns the points once Task 5 is done, and the
        text "not available yet (Task 5)" before. `*arguments` passes on any further arguments.
    """
    try:
        return task_method(*arguments)
    except NotImplementedError:
        return NOT_AVAILABLE.format(task_number=task_number)


def make_discount(discount_settings: dict | None) -> Discount:
    """Turn the `discount:` part of a basket file into a Discount object.

    This is the one place that maps the text in a file to a class. All other code only calls
    discount.apply() and never asks which kind of discount it has.
    """
    if discount_settings is None:
        return NoDiscount()
    kind = discount_settings["kind"]
    if kind == "percentage":
        return PercentageDiscount(discount_settings["percent"])
    if kind == "threshold":
        # Imported here, not at the top: ThresholdDiscount exists only after Task 6, and the
        # other baskets must work before that.
        from supermarket.discounts import ThresholdDiscount

        return ThresholdDiscount(
            discount_settings["minimum_subtotal"], discount_settings["amount_off"]
        )
    raise ValueError(f"unknown discount kind {kind!r}; use percentage or threshold")


def checkout_main(argv: list[str] | None = None) -> int:
    """The command `htw-checkout`: one customer's basket, from the YAML file to the receipt.

    Returns:
        The exit code: 0 if the customer could pay, 1 if the checkout counter refused the basket.
    """
    parser = argparse.ArgumentParser(description="One customer at the checkout counter.")
    parser.add_argument(
        "basket", help="a basket name such as ana (from data/baskets/), or the path of a YAML file"
    )
    arguments = parser.parse_args(argv)  # argv None: read the words typed in the terminal
    basket_path = find_basket(arguments.basket)
    if basket_path is None:
        # parser.error prints the usage line and the message, then stops with exit code 2.
        parser.error(
            f"no basket {arguments.basket!r}; choose one of {', '.join(basket_names())}, "
            "or give the path of a YAML file"
        )
    basket = load_yaml(basket_path)

    try:
        discount = make_discount(basket.get("discount"))
    except ImportError:
        print("This basket uses a threshold discount: " + NOT_AVAILABLE.format(task_number=6))
        return 1
    has_member_card = basket.get("has_member_card", False)
    checkout = Checkout(build_sample_inventory(), discount, has_member_card)

    card_text = "with member card" if has_member_card else "without member card"
    print("HTW Markt · checkout counter")
    print(f"Basket file : {basket_path.name}")
    print(f"Customer    : {basket['customer']}, {card_text}")
    print(f"Discount    : {discount}")
    print()

    try:
        for product_name, quantity in basket["items"].items():
            checkout.scan(product_name, quantity)
    except SupermarketError as error:
        print(f"The checkout counter refuses the basket: {error}")
        return 1
    print(checkout.receipt())
    print()

    print(f"Spent per category : {run_task(7, checkout.cart.total_by_category)}")
    print(f"Loyalty points     : {run_task(5, checkout.loyalty_points)}")
    if "paid" in basket:
        print(f"Paid               : {basket['paid']:.2f}")
        try:
            print(f"Change             : {run_task(8, checkout.pay, basket['paid'])}")
        except SupermarketError as error:
            print(f"Payment refused    : {error}")
            return 1
    return 0


def shelves_main(argv: list[str] | None = None) -> int:
    """The command `htw-shelves`: every product with its prices and stock, after a delivery.

    Returns:
        The exit code: 0 if everything worked, 1 if a delivery line was refused.
    """
    parser = argparse.ArgumentParser(description="The shelves of HTW Markt: prices and stock.")
    parser.add_argument("--delivery", type=Path, help="a delivery, e.g. data/delivery.yaml")
    parser.add_argument("--threshold", type=int, default=5, help="running low below (default 5)")
    arguments = parser.parse_args(argv)
    inventory = build_sample_inventory()

    print("HTW Markt · shelves")
    if arguments.delivery is not None:
        delivery = load_yaml(arguments.delivery)
        print(f"Delivery file: {arguments.delivery}")
        for product_name, quantity in delivery["items"].items():
            try:
                inventory.restock(product_name, quantity)
            except NotImplementedError:
                print("Delivery: " + NOT_AVAILABLE.format(task_number=3))
                break
            except (SupermarketError, ValueError) as error:
                print(f"Delivery line refused: {product_name}: {quantity}: {error}")
                return 1
    print()

    print(f"{'Product':<20}{'Kind':<16}{'Net':>6}{'VAT':>6}{'Shelf':>8}{'Stock':>7}")
    for product_name in inventory.product_names():
        product = inventory.find(product_name)
        kind = type(product).__name__  # the class name, for example FoodProduct
        vat_text = f"{product.vat_rate():.0%}"  # 0.07 becomes "7%"
        print(
            f"{product_name:<20}{kind:<16}{product.net_price:>6.2f}{vat_text:>6}"
            f"{product.gross_price():>8.2f}{inventory.quantity_of(product_name):>7}"
        )
    print()
    low_stock = run_task(2, inventory.low_stock, arguments.threshold)
    print(f"Running low (fewer than {arguments.threshold}): {low_stock}")
    return 0
