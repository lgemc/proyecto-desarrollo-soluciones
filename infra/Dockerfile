# Build stage for frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci

COPY ui/ ./
RUN npm run build

# Main stage for backend
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY animal_classification/ ./animal_classification/
COPY README.md .

RUN uv sync

COPY models/animal-classifier-resnet.pth ./models/animal-classifier-resnet.pth

# Copy built frontend files
COPY --from=frontend-builder /frontend/dist ./static

# Environment variable for backend URL
ENV VITE_API_URL=/api/v1

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "animal_classification.app.main:app", "--host", "0.0.0.0", "--port", "8000"]