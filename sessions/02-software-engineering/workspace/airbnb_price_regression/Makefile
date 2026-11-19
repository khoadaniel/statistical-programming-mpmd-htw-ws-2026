# Shortcuts for the commands of this project. Run `make help` to list them.
# Each target (the word before the colon) is run as `make <target>`; the indented lines are the
# shell commands it runs. The text after ## is what `make help` prints.

# .PHONY says: these targets are commands, not files that make should look for.
# The port can be changed per command: `make serve PORT=8001`.
PORT ?= 8000

.PHONY: help install train serve predict lint format clean

help:  ## Show this list
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

install:  ## Create the environment and install the locked dependencies
	uv sync

train:  ## Train the model and write checkpoints/price_model.joblib
	uv run train-price-model

serve:  ## Start the API on http://127.0.0.1:$(PORT) (docs at /docs)
	uv run fastapi run src/price_model/serving/app.py --host 127.0.0.1 --port $(PORT)

predict:  ## Ask the running API for the price of a listing for 4 guests
	curl -s -X POST http://127.0.0.1:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"accommodates": 4}'

lint:  ## Check the code style
	uv run ruff check . && uv run ruff format --check .

format:  ## Fix the code style automatically
	uv run ruff check --fix . && uv run ruff format .

clean:  ## Delete the checkpoint and the caches
	rm -f checkpoints/price_model.joblib
	rm -rf .ruff_cache
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
