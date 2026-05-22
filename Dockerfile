FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./

ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 && \
        pip install -r requirements-dev.txt ; \
    else \
        pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 && \
        pip install -r requirements.txt ; \
    fi

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY sql/ ./sql/

RUN mkdir -p /app/data/raw /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
