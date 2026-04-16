FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmagic1 \
    libpoppler-cpp-dev \
    libreoffice-writer-nogui \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM base AS runtime

COPY --from=builder /install /usr/local

RUN useradd --no-create-home --shell /bin/false appuser

COPY src/ ./src/
COPY configs/ ./configs/
COPY migrations/ ./migrations/

RUN mkdir -p /data/images && chown appuser:appuser /data/images

USER appuser

CMD ["celery", "-A", "src.worker", "worker", "--loglevel=info", "--concurrency=4"]
