"""The package `gradebook`. This file runs once, when the package is first imported.

It re-exports the main names, so that users can write `from gradebook import Student`
instead of `from gradebook.models import Student`.
"""

from gradebook.loading import load_students
from gradebook.models import Student
from gradebook.scores import average_score, best_student
