FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

# Pre-download Ghana Entity Articles database (~438MB) during build for 100% offline-readiness
RUN python3 -c "from ghana_mcp.providers.entity_articles import ensure_entity_articles_db; ensure_entity_articles_db()"

ENV PORT=8000
ENV HOST=0.0.0.0
ENV WEB_CONCURRENCY=4
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python3", "-m", "ghana_mcp.server"]
