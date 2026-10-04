# Container image of the tariff heading service.
#   build: docker build -t tariff-service:1.0.0 .
#   run:   docker run --rm -p 8000:8000 tariff-service:1.0.0      (docs at http://localhost:8000/docs)
# The model files must be in models/ before the build (train.py, or download them from a release).
FROM python:3.12-slim

# uv from its official image, pinned to a minor version
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1. dependencies first: this layer is reused until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra postgres --no-install-project --no-cache

# 2. then the code (installed as a package, not editable) and the model files
COPY src ./src
RUN uv sync --frozen --no-dev --extra postgres --no-editable --no-cache
COPY models ./models
ENV MODEL_DIR=/app/models PATH="/app/.venv/bin:$PATH"

# 3. run as a non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "tariff_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
