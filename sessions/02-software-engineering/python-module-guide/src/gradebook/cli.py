"""The command-line interface: reads what the user typed, loads the file and prints the report.

Started from the project folder (the folder with pyproject.toml):

    uv run gradebook                       # the installed command, see [project.scripts]
    uv run python -m gradebook.cli         # this module, found by its dotted name
    uv run python -m gradebook             # the package, through __main__.py
"""

import argparse

from gradebook.loading import load_students
from gradebook.reports.summary import subject_report


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and print the report. `argv=None` reads the terminal; tests pass a list."""
    parser = argparse.ArgumentParser(prog="gradebook", description="Average and best score per subject.")
    # a path relative to the current folder of the terminal, not to this file
    parser.add_argument("csv_path", nargs="?", default="data/students.csv",
                        help="CSV file with name, subject, score (default: data/students.csv)")
    args = parser.parse_args(argv)
    students = load_students(args.csv_path)
    print(subject_report(students))


if __name__ == "__main__":  # true when this file is started, false when it is imported
    main()
