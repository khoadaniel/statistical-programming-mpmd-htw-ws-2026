"""Reading the gradebook from a CSV file."""

import csv
from pathlib import Path

from gradebook.models import Student  # absolute import: starts with the package name


def load_students(path: str | Path) -> list[Student]:
    """Read a CSV file with the columns name, subject, score into a list of Student objects."""
    with open(path, newline="", encoding="utf-8") as file:
        return [Student(row["name"], row["subject"], float(row["score"])) for row in csv.DictReader(file)]
