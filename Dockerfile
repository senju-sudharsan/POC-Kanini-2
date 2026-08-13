# syntax=docker/dockerfile:1
# ============================================================
# Stage 1: Build the React/Vite frontend
# ============================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --prefer-offline

COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Production backend (Python 3.11 slim)
# ============================================================
FROM python:3.11-slim AS backend

# Install system packages needed by pytesseract (optional OCR).
# Remove this block if OCR is not required to reduce image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python backend dependencies first (layer-cached until pyproject.toml changes)
COPY backend/ ./backend/
RUN pip install --no-cache-dir ./backend

# Copy the compiled frontend into the expected location
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# ============================================================
# Runtime configuration
# ============================================================

# Gemini API key must be supplied at runtime — NEVER bake into the image.
# docker run -e GEMINI_API_KEY=your-key ...
ENV GEMINI_API_KEY=""
ENV GEMINI_MODEL="gemini-2.5-flash"
ENV ENVIRONMENT="production"
ENV FRONTEND_BUILD_DIR="/app/frontend/dist"
ENV PYTHONUNBUFFERED=1

# Chroma vector store persists to a volume.
# Mount: docker run -v aura_chroma:/app/data/chroma ...
ENV RAG_VECTOR_STORE_DIR="data/chroma"

EXPOSE 8000

# Run with a single worker for this POC.
# For production scale, use --workers N matching available CPU cores.
CMD ["uvicorn", "poc_kanini.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
