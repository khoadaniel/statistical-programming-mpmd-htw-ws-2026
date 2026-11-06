"""A helper script, not an entry point of the app. It lives outside src/, so it is not installed with the package.

It uses the package exactly as a notebook or another project would, by its name. That works because uv
installed `gradebook` into .venv. Start it from the project folder:

    uv run python scripts/class_report.py
"""

from gradebook import average_score, load_students
from gradebook.reports.summary import subject_report

students = load_students("data/students.csv")
print(subject_report(students))
print(f"Overall average: {average_score(students):.1f}")
