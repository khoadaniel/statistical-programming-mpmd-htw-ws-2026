"""Request and response models. FastAPI validates every request against them before our code runs."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
Code2 = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[a-zA-Z]{2}$")]


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"description": "Damenstiefel mit Oberteil aus Rindleder und Laufsohle aus Gummi",
                      "language": "de"}]})

    description: Description
    language: Code2 | None = None  # two-letter code; logged for monitoring, not used by the model


class HeadingScore(BaseModel):
    heading: str = Field(pattern=r"^\d{4}$")
    score: float  # decision score (margin) of the linear SVM, not a probability
    heading_description: str


class PredictionOut(BaseModel):
    heading: str = Field(pattern=r"^\d{4}$")  # the most likely heading
    top: list[HeadingScore]                  # the three most likely headings, best first
    model_version: str


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[DecisionIn] = Field(min_length=1, max_length=100)


class BatchOut(BaseModel):
    predictions: list[PredictionOut]


class Health(BaseModel):
    status: str
    model_version: str
