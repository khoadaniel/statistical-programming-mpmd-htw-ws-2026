"""The CLI is tested by calling main() with a list of arguments, as if they had been typed."""

import pytest

from staycost.cli import main


def test_positional_and_option(capsys):
    main(["80", "--nights", "3"])
    assert capsys.readouterr().out.strip().endswith("= 265.00 EUR")


def test_options_may_stand_before_the_positional_argument(capsys):
    main(["-n", "3", "80"])
    assert "265.00" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["eighty"], ["80", "--nights", "0"]])
def test_bad_input_stops_with_exit_code_2(argv):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2
