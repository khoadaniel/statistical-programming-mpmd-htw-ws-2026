"""The data: one Student object is one row of the gradebook. This module imports nothing from the package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Student:
    """One result: a student's score in one subject, from 0 to 100."""

    name: str
    subject: str
    score: float

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError(f"score must be between 0 and 100, got {self.score}")
