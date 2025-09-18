# Build stage for frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci

COPY ui/ ./
RUN npm run build

# Model conversion stage
FROM python:3.12-slim AS model-converter
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install only dependencies needed for model conversion
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY animal_classification/ ./animal_classification/
COPY tools/convert_to_onnx.py ./tools/convert_to_onnx.py
COPY README.md ./README.md
# Install full dependencies for conversion
RUN uv sync

# Copy PyTorch model and convert to ONNX
COPY models/animal-classifier-resnet.pth ./models/animal-classifier-resnet.pth
RUN uv run python tools/convert_to_onnx.py

# Production stage - lightweight runtime
FROM python:3.12-slim AS production

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Create minimal pyproject.toml for production dependencies only
COPY pyproject.toml .
COPY README.md ./README.md
# Copy application code
COPY animal_classification/ ./animal_classification/


# Copy converted ONNX model and metadata
COPY --from=model-converter /app/models/animal-classifier-resnet.onnx ./models/animal-classifier-resnet.onnx
COPY --from=model-converter /app/models/animal-classifier-resnet.json ./models/animal-classifier-resnet.json

# Copy built frontend files
COPY --from=frontend-builder /frontend/dist ./static

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Install minimal production dependencies
COPY infra/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Use exec form for better signal handling
CMD ["python", "-m", "uvicorn", "animal_classification.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]