"""The command `htw-checkout`: one customer at the checkout, from the YAML files to the screen.

    uv run htw-checkout lina      # reads data/catalogue.yaml and data/baskets/lina.yaml

This module only reads the files, calls the classes and prints. Every calculation happens in the
classes, so that the tests can check it without the command line. Before a task is done, its
method raises NotImplementedError, and the command prints "(Task N)" in place of the result.
"""

import argparse
from pathlib import Path

import yaml

from supermarket.checkout import Checkout, SupermarketError
from supermarket.products import BottledDrink, FoodProduct, FreshProduct, NonFoodProduct, Product

# cli.py is in src/supermarket/, so three levels up (parents[2]) is the project folder.
DATA_FOLDER = Path(__file__).resolve().parents[2] / "data"

# From the word `kind` in catalogue.yaml to the class.
PRODUCT_CLASSES: dict[str, type[Product]] = {
    "food": FoodProduct,
    "fresh": FreshProduct,
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


def use_by_check_is_written(catalogue: dict[str, Product]) -> bool:
    """True once every product answers is_sellable(), that is, once Task 2, step 2 is written."""
    try:
        for product in catalogue.values():
            product.is_sellable()
    except NotImplementedError:
        return False
    return True


def checkout_main(argv: list[str] | None = None) -> int:
    """Serve one customer: scan the basket, print what was not sold, the receipt and the change.

    Returns:
        The exit code: 0 if the customer was served, 1 if not.
    """
    parser = argparse.ArgumentParser(description="One customer at the checkout counter.")
    parser.add_argument("basket", help="a basket name such as lina, from data/baskets/")
    basket = load_yaml(DATA_FOLDER / "baskets" / f"{parser.parse_args(argv).basket}.yaml")

    print(f"HTW Markt · Friday, 18:00 · customer: {basket['customer']}")
    print()
    try:
        catalogue = load_catalogue()
    except NotImplementedError:
        print("The till cannot start: the shelf has sandwiches, and the class FreshProduct")
        print("cannot create them yet (Task 1).")
        return 1

    checkout = Checkout(catalogue)
    for product_name, quantity in basket["items"].items():
        try:
            checkout.scan(product_name, quantity)
        except SupermarketError as error:  # your ExpiredProductError is one: it inherits from it
            print(f"Not sold: {error}")
    if not use_by_check_is_written(catalogue):
        print("Use-by dates are not checked yet (Task 2).")
    print()
    print(checkout.receipt())
    print()
    print(f"Paid      : {basket['paid']:.2f}")
    try:
        print(f"Change    : {checkout.pay(basket['paid']):.2f}")
    except NotImplementedError:
        print("Change    : (Task 3)")
    except SupermarketError as error:  # PaymentError is one
        print(f"Not enough money: {error}")
        return 1
    return 0
