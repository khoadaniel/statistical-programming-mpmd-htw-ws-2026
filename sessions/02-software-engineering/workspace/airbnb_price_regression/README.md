# Airbnb price model · Session 2: organise a machine learning project

The smallest complete machine learning project: a **linear regression** that predicts the price per night of a Berlin Airbnb listing from the **number of guests** it sleeps,

$$\text{price} = a \cdot \text{accommodates} + b$$

It is trained once, saved as a checkpoint, and served by a **FastAPI** web service that you call with `curl`. The model is deliberately plain (scikit-learn `LinearRegression`, no tuning, no regularisation) so that the project structure and the object-oriented code are the lesson, not the statistics.

## Project structure

```
airbnb_price_regression/
├── Makefile                    shortcuts: make train, make serve, make predict, ...
├── pyproject.toml              dependencies and the command `train-price-model`
├── uv.lock                     exact versions of every package
├── .python-version             Python version used by uv
├── checkpoints/                the saved model (written by `make train`, not kept in Git)
└── src/price_model/
    ├── config.py               file locations and column names, in one place
    ├── schemas.py              pydantic models: the shape of API input and output
    ├── models/
    │   ├── base.py             BasePriceModel: abstract class, save() and load()
    │   └── linear.py           LinearPriceModel: the straight line, inherits from the base
    ├── training/
    │   ├── trainer.py          PriceModelTrainer: load the data, fit, save the checkpoint
    │   └── train.py            the command `train-price-model` (argparse)
    └── serving/
        └── app.py              the FastAPI app: /health, /model, /predict
```

Data flows left to right: `listings.parquet` → **training** → `checkpoints/price_model.joblib` → **serving** → your request.

## Object-oriented ideas in the code

| Idea | Where to look |
|---|---|
| **Abstraction**: a class that says *what* must be possible, not *how* | `BasePriceModel` in `models/base.py` (`ABC`, `@abstractmethod`) |
| **Inheritance**: a subclass reuses what the parent already wrote | `LinearPriceModel(BasePriceModel)` gets `save()` and `load()` without writing them |
| **Encapsulation**: internals are hidden behind a small public surface | `_estimator` and `_is_fitted` in `LinearPriceModel`; outside code uses `fit()`, `predict_price()` and the properties |

Every code line has a comment, so read the files in this order: `config.py`, `schemas.py`, `models/base.py`, `models/linear.py`, `training/trainer.py`, `training/train.py`, `serving/app.py`.

## Setup

You need [uv](https://docs.astral.sh/uv/) and the case-study data (`git lfs pull`, see [case-study/README.md](../../../../case-study/README.md)). Run every command from this folder:

```bash
cd sessions/02-software-engineering/workspace/airbnb_price_regression
make install
```

## Use it

**1. Train** the model. It reads `case-study/airbnb/data/listings.parquet`, keeps the short-stay listings with a price of at most 1,000 EUR, fits the line and writes the checkpoint:

```bash
make train
# Trained on 5278 listings: price = 31.16 * accommodates + 76.33
```

**2. Serve** the model. The app loads the checkpoint once at startup and listens on port 8000 (`make serve PORT=8001` if that port is taken). Leave this terminal open:

```bash
make serve
```

**3. Call it** from a second terminal. `make predict` sends the first request below.

Predict the price of a listing for 4 guests:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"accommodates": 4}'
# {"accommodates":4,"predicted_price_eur":200.97}
```

Read the fitted line (the slope and the intercept):

```bash
curl http://127.0.0.1:8000/model
# {"slope":31.15899649907584,"intercept":76.33037990094532}
```

Check that the service is up:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

A request that breaks a rule is refused with status 422 and a reason, because pydantic checks it (`accommodates` must be between 1 and 16):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"accommodates": 0}'
# {"detail":[{"type":"greater_than_equal","loc":["body","accommodates"],"msg":"Input should be greater than or equal to 1", ...}]}
```

FastAPI also writes interactive documentation from the same pydantic models: open <http://127.0.0.1:8000/docs> in a browser and try the endpoints there.

## Make commands

| Command | What it does |
|---|---|
| `make install` | Creates `.venv` and installs the locked dependencies |
| `make train` | Trains the model and writes the checkpoint |
| `make serve` | Starts the API |
| `make predict` | Sends one example request to the running API |
| `make lint` / `make format` | Checks / fixes the code style with ruff |
| `make clean` | Deletes the checkpoint and the caches |

## Conventions

- **Docstrings** follow the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`, `Returns:`, `Raises:`), the same as the other Session 2 projects.
- **Type hints** on every parameter and return value; **pydantic** models (`BaseModel`) for data that crosses the API boundary; no dataclasses.
- The checkpoint is a joblib file, which can run code when it is opened: load only files you trained yourself.

## Ideas to extend

Add a second feature (`bedrooms`), compare the line with a tree model, or write a pytest test that calls `/predict` with FastAPI's `TestClient`.
