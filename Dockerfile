# syntax=docker/dockerfile:1
# Two-stage build: dependencies are resolved by uv in the builder, and the final
# image only carries the virtualenv, the source and the data directory.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY app.py ./
COPY data ./data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000 8501

# The API is the default; docker-compose overrides the command for the UI.
CMD ["uvicorn", "insurance_policy_analyzer.api:app", "--host", "0.0.0.0", "--port", "8000"]
