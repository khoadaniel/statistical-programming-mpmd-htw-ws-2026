"""The command-line interface (CLI): the program's front door for people typing in a terminal.

It only reads what the user typed, checks it and hands it to `stay_cost`. The calculation lives in
`costs.py`, so that tests and notebooks can use it without a terminal.

Three ways to start it, all ending in `main()`:

    uv run staycost 80 --nights 3          # the installed command, see [project.scripts] in pyproject.toml
    uv run python -m staycost.cli 80 -n 3  # this module, found by its dotted name
    uv run python -m staycost 80 -n 3      # the package, through __main__.py
"""

import argparse

from staycost.costs import CLEANING_FEE, stay_cost


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and print the cost. `argv=None` reads the terminal; tests pass a list."""
    parser = argparse.ArgumentParser(prog="staycost", description="Cost of a stay in a Berlin listing.")
    # positional argument: no dashes, required, identified by its position
    parser.add_argument("price", type=float, help="price per night in EUR")
    # options: start with dashes, optional, identified by their name, may stand anywhere
    parser.add_argument("-n", "--nights", type=int, default=1, help="number of nights (default: 1)")
    parser.add_argument("--fee", type=float, default=CLEANING_FEE,
                        help=f"cleaning fee in EUR (default: {CLEANING_FEE:g})")
    args = parser.parse_args(argv)

    try:
        total = stay_cost(args.price, args.nights, args.fee)
    except ValueError as error:
        parser.error(str(error))  # prints the usage line and the message, exits with code 2
    print(f"{args.nights} night(s) at {args.price:.2f} EUR + {args.fee:.2f} EUR cleaning = {total:.2f} EUR")


if __name__ == "__main__":  # true for `python -m staycost.cli`, false when the module is imported
    main()
