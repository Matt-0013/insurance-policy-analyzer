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
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface


# Bake the embedding model in. Without it the first /ask spends about 50 seconds
# downloading weights, which makes a cold container look broken and makes every
# restarted replica pay that cost again. The chmod matters because the build runs
# as root while the server runs as appuser, which must be able to read the cache.
RUN python -c "from insurance_policy_analyzer.config import EMBEDDING_MODEL; from sentence_transformers import SentenceTransformer; SentenceTransformer(EMBEDDING_MODEL)" \
    && chmod -R a+rX /app/.cache
USER appuser
EXPOSE 8000 8501

# Warm-up takes ~30 s, so the start period has to outlast it: without it the first
# probes land while the app is still loading and the orchestrator restarts a
# container that was only booting. The probe is python because the slim base has
# no curl, and /health answers 503 when the index is missing, which urllib turns
# into a non-zero exit.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# The API is the default; docker-compose overrides the command for the UI.
CMD ["uvicorn", "insurance_policy_analyzer.api:app", "--host", "0.0.0.0", "--port", "8000"]
