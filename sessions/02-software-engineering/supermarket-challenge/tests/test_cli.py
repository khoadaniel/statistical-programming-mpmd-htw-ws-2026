"""Tests of cli.py: the catalogue file and the command, called with a list of words.

capsys is a fixture of pytest itself: capsys.readouterr().out holds everything that was printed.
"""

import pytest

from supermarket.cli import checkout_main, load_catalogue
from supermarket.products import BottledDrink, FoodProduct


def test_each_line_of_the_catalogue_becomes_the_right_class():
    catalogue = load_catalogue()
    assert isinstance(catalogue["Bread"], FoodProduct)
    assert isinstance(catalogue["Cola"], BottledDrink)
    assert catalogue["Bread"].net_price == 2.80


def test_an_unknown_kind_is_refused(tmp_path):
    catalogue_path = tmp_path / "catalogue.yaml"
    catalogue_path.write_text("products:\n  - {name: X, kind: luxury, net_price: 1}\n")
    with pytest.raises(ValueError, match="unknown product kind"):
        load_catalogue(catalogue_path)


def test_checkout_prints_the_receipt(capsys):
    assert checkout_main(["ana"]) == 0
    printed = capsys.readouterr().out
    assert "discount: 10 % off" in printed
    assert "Total EUR                           11.38" in printed
