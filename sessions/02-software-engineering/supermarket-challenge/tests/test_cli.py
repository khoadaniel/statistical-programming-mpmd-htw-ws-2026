"""Tests of cli.py: the two commands, called with a list of words instead of the terminal.

capsys is a fixture of pytest itself: capsys.readouterr().out holds everything that was printed.
These tests check only what stays the same while you work on the tasks.
"""

from pathlib import Path

import pytest

from supermarket.cli import checkout_main, shelves_main

DATA_FOLDER = Path(__file__).parent.parent / "data"  # tests/../data, wherever pytest starts


def test_checkout_prints_the_receipt(capsys):
    exit_code = checkout_main(["ana"])
    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "Customer    : Ana, with member card" in printed
    assert "Discount    : 10 % off" in printed
    assert "Total EUR                           11.38" in printed


def test_checkout_refuses_more_than_is_in_stock(capsys):
    exit_code = checkout_main(["dana"])
    assert exit_code == 1
    assert "only 3 in stock" in capsys.readouterr().out


def test_a_basket_can_also_be_given_as_a_path(capsys):
    exit_code = checkout_main([str(DATA_FOLDER / "baskets" / "ana.yaml")])
    assert exit_code == 0
    assert "Basket file : ana.yaml" in capsys.readouterr().out


def test_an_unknown_basket_name_lists_the_baskets(capsys):
    with pytest.raises(SystemExit):  # parser.error stops the program
        checkout_main(["anna"])
    assert "choose one of ana, ben, cem, dana" in capsys.readouterr().err


def test_shelves_list_every_product_with_its_shelf_price(capsys):
    exit_code = shelves_main([])
    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "Mineral water       BottledDrink      0.49   19%    0.83     48" in printed
