FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

FROM python:3.12.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    CLAIM_HOST=0.0.0.0 \
    CLAIM_PORT=8000 \
    CLAIM_DEVICE=cpu \
    CLAIM_MODEL_DIR=/models/bert_mixed \
    CLAIM_CALIBRATOR_PATH=/app/reports/generated/bert_calibration/calibrator.json

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /models \
    && chown app:app /app /models

COPY --chown=10001:10001 pyproject.toml uv.lock README.md LICENSE ./
USER 10001:10001
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 reports/generated/bert_calibration/calibrator.json ./reports/generated/bert_calibration/calibrator.json
COPY --chown=10001:10001 artifacts/bert_mixed /models/bert_mixed
RUN uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

CMD ["claim-api"]
