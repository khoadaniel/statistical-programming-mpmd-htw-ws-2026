"""Calculations on a list of students: no printing and no files, so the functions can be tested on small lists."""

from gradebook.models import Student  # absolute import: starts with the package name


def average_score(students: list[Student], subject: str | None = None) -> float:
    """Mean score of all students, or only of one subject.

    >>> average_score([Student("Ana", "Statistics", 80), Student("Ben", "Statistics", 90)])
    85.0
    """
    selected_scores = [student.score for student in students if subject is None or student.subject == subject]
    if not selected_scores:
        raise ValueError(f"no scores for subject {subject!r}")
    return sum(selected_scores) / len(selected_scores)


def best_student(students: list[Student], subject: str) -> Student:
    """The student with the highest score in one subject."""
    subject_students = [student for student in students if student.subject == subject]
    if not subject_students:
        raise ValueError(f"no scores for subject {subject!r}")
    return max(subject_students, key=lambda student: student.score)
