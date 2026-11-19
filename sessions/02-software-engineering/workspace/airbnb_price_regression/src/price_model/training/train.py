"""The command ``train-price-model``: train the model and write the checkpoint."""

import argparse
from pathlib import Path

from price_model.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_PATH
from price_model.models import LinearPriceModel
from price_model.training.trainer import PriceModelTrainer, describe


def main() -> None:
    """Read the command-line arguments, train, and print the fitted line."""
    # argparse turns the words typed after the command into Python values.
    parser = argparse.ArgumentParser(description="Train the Airbnb price model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    arguments = parser.parse_args()

    model = LinearPriceModel()  # an unfitted model; the trainer fits it
    trainer = PriceModelTrainer(model, arguments.data_path, arguments.checkpoint_path)
    n_listings = trainer.train()

    print(describe(model.coefficients, n_listings))
    print(f"Checkpoint written to {arguments.checkpoint_path}")


# This block runs only when the file is started directly, not when another file imports it.
if __name__ == "__main__":
    main()
