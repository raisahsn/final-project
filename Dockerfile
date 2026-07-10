# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for TensorFlow and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and Streamlit config
COPY src/ ./src/
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
ENV PYTHONPATH=/app/src
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: run FastAPI. Uses $PORT so it works on Railway/Render/Fly.
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
