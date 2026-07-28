# Slim, reproducible image using uv.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# libgomp1 is required by lightgbm/xgboost at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (static binary).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system -e ".[dev]"

# App code.
COPY config ./config
COPY scripts ./scripts
COPY tests ./tests

ENV ETF_INTEL_MARKET_SOURCE=synthetic \
    ETF_INTEL_MACRO_SOURCE=synthetic

CMD ["python", "scripts/run_ingest.py", "--source", "synthetic"]
