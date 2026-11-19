"""Pydantic models: the shape of what goes into and comes out of the API.

A pydantic model is a class whose attributes are type-annotated fields. Pydantic checks the types
and the rules (``Field(ge=1)``) when an object is created, and raises a ``ValidationError`` if the
data do not fit. FastAPI uses the same models to validate requests and to write the API docs.
"""

from pydantic import BaseModel, Field


class ListingFeatures(BaseModel):
    """The input of the model: what we know about one listing."""

    # Field(...) adds rules to the annotation: ge = "greater or equal", le = "less or equal".
    accommodates: int = Field(
        ge=1,
        le=16,
        description="Number of guests the listing sleeps.",
        examples=[4],
    )


class PricePrediction(BaseModel):
    """The output of the model: the predicted price for one listing."""

    accommodates: int = Field(description="The number of guests the prediction is for.")
    predicted_price_eur: float = Field(description="Predicted price per night in EUR.")


class ModelCoefficients(BaseModel):
    """The two numbers of the fitted line: price = slope * accommodates + intercept."""

    slope: float = Field(description="EUR added to the price for each extra guest.")
    intercept: float = Field(
        description="Price in EUR of a listing with zero guests (the line's start)."
    )
