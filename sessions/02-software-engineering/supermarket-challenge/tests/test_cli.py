"""Tests of cli.py: the catalogue file and the command, called with a list of words.

capsys is a fixture of pytest itself: capsys.readouterr().out holds everything that was printed.
"""

import pytest

from supermarket.cli import checkout_main, load_catalogue
from supermarket.products import BottledDrink, FoodProduct


def test_an_unknown_kind_is_refused(tmp_path):
    catalogue_path = tmp_path / "catalogue.yaml"
    catalogue_path.write_text("products:\n  - {name: X, kind: luxury, net_price: 1}\n")
    with pytest.raises(ValueError, match="unknown product kind"):
        load_catalogue(catalogue_path)


@pytest.mark.task(1)
def test_each_line_of_the_catalogue_becomes_the_right_class():
    from supermarket.products import FreshProduct

    catalogue = load_catalogue()
    assert isinstance(catalogue["Bread"], FoodProduct)
    assert isinstance(catalogue["Cola"], BottledDrink)
    assert isinstance(catalogue["Cheese sandwich"], FreshProduct)
    assert catalogue["Cheese sandwich"].days_until_use_by == 0


@pytest.mark.task(3)
def test_lina_is_served(capsys):
    """The whole story: this test passes once all three tasks are done."""
    from supermarket.checkout import Checkout, ExpiredProductError  # noqa: F401 (Task 2)

    Checkout(load_catalogue()).pay(0.00)  # raises NotImplementedError until Tasks 1 and 3 are done
    assert checkout_main(["lina"]) == 0
    printed = capsys.readouterr().out
    assert "Not sold: Egg sandwich" in printed
    assert "Total EUR                            7.78" in printed
    assert "Change    : 2.22" in printed
