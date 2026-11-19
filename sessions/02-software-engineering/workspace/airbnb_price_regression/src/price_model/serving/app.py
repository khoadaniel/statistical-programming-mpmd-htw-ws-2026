"""The FastAPI app: loads the checkpoint once and serves predictions over HTTP."""

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from price_model.config import DEFAULT_CHECKPOINT_PATH
from price_model.models import LinearPriceModel
from price_model.schemas import ListingFeatures, ModelCoefficients, PricePrediction


# An async generator function: FastAPI runs the code before `yield` at startup and the code after
# it at shutdown. (Older tutorials add @asynccontextmanager here; current FastAPI no longer needs it.)
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run code at startup (before the first request) and at shutdown (after the yield).

    Loading the model here means it is read from disk once, not on every request.
    """
    # os.environ.get reads an environment variable, so the checkpoint can be changed without code.
    checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH))
    app.state.model = LinearPriceModel.load(checkpoint_path)  # app.state keeps shared objects
    yield  # the app serves requests while we are paused here


def get_model(request: Request) -> LinearPriceModel:
    """Give an endpoint the model that was loaded at startup (a FastAPI dependency)."""
    return request.app.state.model


# Annotated[Type, Depends(function)] tells FastAPI: "call get_model and pass me the result".
ModelDependency = Annotated[LinearPriceModel, Depends(get_model)]

app = FastAPI(
    title="Airbnb price model",
    description="Predicts the nightly price of a Berlin listing from the number of guests.",
    lifespan=lifespan,
)


@app.get("/health")  # a GET request to /health runs the function below
def health() -> dict[str, str]:
    """Tell a load balancer or a human that the service is up."""
    return {"status": "ok"}


@app.get("/model")
def model_coefficients(model: ModelDependency) -> ModelCoefficients:
    """Show the fitted line: the slope and the intercept."""
    return model.coefficients  # the return type hint is also the response schema in the docs


@app.post("/predict")  # POST is used because the client sends data in the request body
def predict(features: ListingFeatures, model: ModelDependency) -> PricePrediction:
    """Predict the price per night for one listing.

    FastAPI reads the JSON body, builds a ``ListingFeatures`` from it (pydantic checks the rules),
    and answers with status 422 and an explanation if the body is invalid.
    """
    predicted_price = model.predict_price(features.accommodates)
    return PricePrediction(
        accommodates=features.accommodates,
        predicted_price_eur=round(predicted_price, 2),
    )
