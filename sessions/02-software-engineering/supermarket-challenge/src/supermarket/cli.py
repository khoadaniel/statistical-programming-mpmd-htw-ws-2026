"""The command `htw-checkout`: one customer's basket, from the YAML files to the receipt.

    uv run htw-checkout ana      # reads data/catalogue.yaml and data/baskets/ana.yaml

This module only reads the files, calls the classes and prints. Every calculation happens in the
classes, so that the tests can check it without the command line. Before a task is done, its
method raises NotImplementedError, and the command prints "not available yet (Task N)" instead.
"""

import argparse
from pathlib import Path

import yaml

from supermarket.checkout import Checkout, SupermarketError
from supermarket.discounts import Discount, NoDiscount, PercentageDiscount, ThresholdDiscount
from supermarket.products import BottledDrink, FoodProduct, NonFoodProduct, Product

# cli.py is in src/supermarket/, so three levels up (parents[2]) is the project folder.
DATA_FOLDER = Path(__file__).resolve().parents[2] / "data"
NOT_AVAILABLE = "not available yet (Task {task_number})"

# From the word `kind` in catalogue.yaml to the class. Task 1, step 4: add "fresh": FreshProduct.
PRODUCT_CLASSES: dict[str, type[Product]] = {
    "food": FoodProduct,
    "non-food": NonFoodProduct,
    "drink": BottledDrink,
}


def load_yaml(yaml_path: Path) -> dict:
    """Read a YAML file into a dictionary: `Bread: 1` in the file becomes {"Bread": 1}."""
    with open(yaml_path, encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def load_catalogue(catalogue_path: Path = DATA_FOLDER / "catalogue.yaml") -> dict[str, Product]:
    """Every product of the catalogue as an object, by name.

    Raises:
        ValueError: If a line has a kind that is not in PRODUCT_CLASSES.
    """
    catalogue: dict[str, Product] = {}
    for line in load_yaml(catalogue_path)["products"]:
        settings = dict(line)  # a copy: pop must not change the line itself
        kind = settings.pop("kind")
        if kind not in PRODUCT_CLASSES:
            raise ValueError(f"unknown product kind {kind!r}; use one of {sorted(PRODUCT_CLASSES)}")
        product = PRODUCT_CLASSES[kind](**settings)  # ** passes the dictionary as arguments
        catalogue[product.name] = product
    return catalogue


def make_discount(settings: dict | None) -> Discount:
    """Turn the `discount:` part of a basket file into a Discount object."""
    if settings is None:
        return NoDiscount()
    if settings["kind"] == "percentage":
        return PercentageDiscount(settings["percent"])
    if settings["kind"] == "threshold":
        return ThresholdDiscount(settings["minimum_subtotal"], settings["amount_off"])
    raise ValueError(f"unknown discount kind {settings['kind']!r}")


def checkout_main(argv: list[str] | None = None) -> int:
    """Read a basket, scan it, print the receipt and the change.

    Returns:
        The exit code: 0 if the customer could pay, 1 if not.
    """
    parser = argparse.ArgumentParser(description="One customer at the checkout counter.")
    parser.add_argument("basket", help="a basket name such as ana, from data/baskets/")
    basket = load_yaml(DATA_FOLDER / "baskets" / f"{parser.parse_args(argv).basket}.yaml")

    try:
        discount = make_discount(basket.get("discount"))
    except NotImplementedError:
        print("This basket uses a threshold discount: " + NOT_AVAILABLE.format(task_number=2))
        return 1
    checkout = Checkout(load_catalogue(), discount)
    for product_name, quantity in basket["items"].items():
        checkout.scan(product_name, quantity)

    print(f"HTW Markt · {basket['customer']} · discount: {discount}")
    print()
    print(checkout.receipt())
    print()
    print(f"Paid      : {basket['paid']:.2f}")
    try:
        print(f"Change    : {checkout.pay(basket['paid']):.2f}")
    except NotImplementedError:
        print("Change    : " + NOT_AVAILABLE.format(task_number=3))
    except SupermarketError as error:  # your PaymentError is one, because it inherits from it
        print(f"Not enough money: {error}")
        return 1
    return 0
