"""Request and response models. FastAPI validates every request against them before our code runs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000)]
Label = Literal["neg", "neu", "pos"]


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"title": "Stopped working", "text": "Broke after two days. Waste of money."}]})

    text: Text
    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] = ""


class PredictionOut(BaseModel):
    label: Label
    probabilities: dict[Label, float]
    model_version: str


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[ReviewIn] = Field(min_length=1, max_length=100)


class BatchOut(BaseModel):
    predictions: list[PredictionOut]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
