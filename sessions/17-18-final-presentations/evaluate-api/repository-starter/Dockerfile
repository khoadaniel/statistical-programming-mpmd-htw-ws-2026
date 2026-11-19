# Container image of your service. Copy this file to the root of your repository and change the line marked CHANGE.
#   build: docker build -t my-service .
#   run:   docker run --rm -p 8000:8000 my-service      (docs at http://localhost:8000/docs)
# The model files must be in models/ before the build (git lfs pull, or your training script).
FROM python:3.14-slim

# uv from its official image
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1. dependencies first: this layer is reused until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# 2. then the code and the model files
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --no-cache
COPY models ./models
ENV MODEL_DIR=/app/models PATH="/app/.venv/bin:$PATH"

# 3. run as a non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# CHANGE: "your_package.app:app" is the module and the FastAPI object of your service
CMD ["uvicorn", "your_package.app:app", "--host", "0.0.0.0", "--port", "8000"]
