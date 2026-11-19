"""Tests import the package the same way as every other file: by its name."""

import pytest

from gradebook import Student, average_score, best_student
from gradebook.reports.summary import subject_report

STUDENTS = [
    Student("Ana", "Statistics", 80),
    Student("Ben", "Statistics", 90),
    Student("Ana", "Programming", 70),
]


def test_average_of_all_scores():
    assert average_score(STUDENTS) == 80.0


def test_average_of_one_subject():
    assert average_score(STUDENTS, "Statistics") == 85.0


def test_best_student():
    assert best_student(STUDENTS, "Statistics").name == "Ben"


def test_report_has_one_line_per_subject():
    assert subject_report(STUDENTS).splitlines() == [
        "Programming: average 70.0, best Ana (70)",
        "Statistics: average 85.0, best Ben (90)",
    ]


def test_unknown_subject_is_refused():
    with pytest.raises(ValueError):
        average_score(STUDENTS, "History")


def test_score_out_of_range_is_refused():
    with pytest.raises(ValueError):
        Student("Ana", "Statistics", 120)
