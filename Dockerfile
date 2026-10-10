# --- frontend build ---------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim

ARG VERSION=dev
LABEL org.opencontainers.image.title="MTG Deck Keeper" \
      org.opencontainers.image.description="Self-hosted MTG decklists from photos using a local vision model" \
      org.opencontainers.image.source="https://github.com/adman234/mtg-deck-keeper" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    PORT=9925 \
    PUID=99 \
    PGID=100 \
    HOME=/tmp

WORKDIR /app
COPY backend/ /app/backend/
COPY --from=frontend /src/backend/deck_keeper/static /app/backend/deck_keeper/static
RUN pip install /app/backend && rm -rf /app/backend/build
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /app/data

EXPOSE 9925
VOLUME /app/data
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','9925'), timeout=4)"
ENTRYPOINT ["/entrypoint.sh"]
