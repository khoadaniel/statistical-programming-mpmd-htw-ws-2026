"""Settings shared by training and serving: file locations and the names of the data columns."""

from pathlib import Path

# Path(...) is an object that represents a file location. Relative paths are resolved from the
# folder where you start the command, which is why the Makefile always runs from the project root.
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/price_model.joblib")

# The Inside Airbnb file of the course case study, seen from this project folder.
DEFAULT_DATA_PATH = Path("../../../../case-study/airbnb/data/listings.parquet")

# The names of the two columns the model uses: x (the feature) and y (the target).
FEATURE_COLUMN = "accommodates"  # how many guests the listing sleeps
TARGET_COLUMN = "price"  # the price per night in EUR

# Listings rented for this many nights or more are monthly rentals, not nightly stays.
SHORT_STAY_MAX_NIGHTS = 28

# Prices above this are treated as data errors or luxury outliers and left out of the training.
MAX_PRICE_EUR = 1000.0
