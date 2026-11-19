"""A text report per subject. One folder deeper, and the imports look exactly the same."""

from gradebook.models import Student
from gradebook.scores import average_score, best_student


def subject_report(students: list[Student]) -> str:
    """One line per subject: the average score and the best student."""
    subjects = sorted({student.subject for student in students})
    lines = []
    for subject in subjects:
        best = best_student(students, subject)
        lines.append(f"{subject}: average {average_score(students, subject):.1f}, best {best.name} ({best.score:g})")
    return "\n".join(lines)
