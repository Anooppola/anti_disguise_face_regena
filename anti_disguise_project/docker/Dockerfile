# ─── Stage 1: Python base ──────────────────────────────────────────────────────
FROM python:3.10-slim AS base

WORKDIR /app

# Copy and install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ─── Stage 2: API image ────────────────────────────────────────────────────────
FROM base AS api

# Copy project source
COPY src/          ./src/
COPY mlflow_utils/ ./mlflow_utils/
COPY api/          ./api/
COPY .env          ./.env

# Create model mount point
RUN mkdir -p saved_models

ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_PATH=saved_models/generator_best.pth

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Stage 3: Streamlit image ──────────────────────────────────────────────────
FROM base AS frontend

COPY frontend/ ./frontend/

ENV API_URL=http://api:8000
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
