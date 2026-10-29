"""Runs on `python -m supermarket`: the same as the command `htw-checkout`.

    uv run python -m supermarket ana
    uv run htw-checkout ana          # the same, through [project.scripts]

Python runs this file when it is asked to run the package itself. It passes the work on to cli.py.
"""

from supermarket.cli import checkout_main

# True only when this file is started (python -m supermarket), not when it is imported.
if __name__ == "__main__":
    raise SystemExit(checkout_main())  # the exit code: 0 for success, 1 for a refused basket
