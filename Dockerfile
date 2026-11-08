# ---------------------------------------------------------------------------
# Docker image for the E-Commerce Product Return Prediction API
# Build:  docker build -t product-return-api .
# Run:    docker run -p 8000:8000 product-return-api
# ---------------------------------------------------------------------------
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, ML source package and the trained model artifact.
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container-level health check hitting the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
