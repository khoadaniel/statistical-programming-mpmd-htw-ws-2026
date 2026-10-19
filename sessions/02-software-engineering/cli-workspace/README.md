# CLI workspace · Session 2: from a script to a command

The smallest complete command-line program, to try out what block 2 explains in the section [Scripts, modules, packages and project structure](../workbooks/02-lecture-notebook-to-application-git-and-ci.ipynb): it computes the cost of a stay in a Berlin listing. It has no dependencies and no data, so every command below works right after cloning.

```
cli-workspace/
├── pyproject.toml          [project.scripts] staycost = "staycost.cli:main"  -> the terminal command `staycost`
├── src/staycost/           the package: a folder of modules with one import name
│   ├── __init__.py         runs on `import staycost`; re-exports stay_cost
│   ├── costs.py            the calculation, stay_cost(): no printing, easy to test
│   ├── cli.py              the command-line interface: argparse, then a call to stay_cost
│   ├── __main__.py         runs on `python -m staycost`
│   └── whoami.py           prints its own __name__: started or imported?
└── tests/                  tests of the calculation and of the CLI (main() called with a list)
```

## Try it

Run everything from this folder: `cd sessions/02-software-engineering/cli-workspace`. The first `uv run` creates `.venv` and installs the package into it.

**1 · `__name__` depends on how a file is loaded**

```bash
uv run python -m staycost.whoami             # started directly: __name__ is '__main__', the guarded line runs
uv run python -c "import staycost.whoami"    # imported:         __name__ is 'staycost.whoami', it does not
```

**2 · Three ways to start the same program**

```bash
uv run staycost 80 --nights 3                # the command declared in [project.scripts]
uv run python -m staycost.cli 80 --nights 3  # the module cli.py, found by its dotted name
uv run python -m staycost 80 --nights 3      # the package, through __main__.py
```

All three print the same line, because all three end in `main()` of `cli.py`. The command `staycost` is a small file that uv wrote into `.venv/bin/` when it installed the package; look at it with `cat .venv/bin/staycost`.

**3 · Positional arguments and options**

```bash
uv run staycost --help                       # usage text, written by argparse from the help= strings
uv run staycost 80                           # price is positional: required, no dashes
uv run staycost -n 3 80 --fee 0              # options have dashes, are optional and may stand anywhere
uv run staycost eighty                       # refused: price must be a number (exit code 2)
uv run staycost 80 --nights 0                # refused by stay_cost, reported by the CLI
```

**4 · Tests**

```bash
uv run pytest -q                             # 9 tests: the calculation and the CLI
```

## Exercises

1. Add an option `--guests` (default 1) and print the cost per guest as well. Which file changes, and which does not?
2. Give `--nights` a short form `-N` as well as `-n`. Then try `uv run staycost 80 -n3`: why does it work without a space?
3. Add a second command `staycost-week` to `[project.scripts]` that prints the cost of seven nights. Run `uv sync`, then `uv run staycost-week 80`. What did `uv sync` create in `.venv/bin/`?
