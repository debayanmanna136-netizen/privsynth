# =============================================================================
# PrivSynth - Production Dockerfile (Backend Service)
# Minimal, secure, single-worker container for Python 3.13
# =============================================================================
FROM python:3.13-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    PRIVSYNTH_STORAGE_DIR=/app/storage

WORKDIR /app

# Install minimal runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 10001) and prepare persistent storage directory
RUN useradd -m -u 10001 -s /bin/bash appuser && \
    mkdir -p /app/storage && \
    chown -R appuser:appuser /app

# Install production dependencies first to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt requirements-full.txt ./
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy backend and ML package code
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser privsynth/ /app/privsynth/

# Switch to non-root user
USER appuser

# Expose internal backend port (not published directly to host)
EXPOSE 8000

# Container healthcheck querying the backend health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/api/health || exit 1

# Start Uvicorn with single worker to maintain process-local active-file locks
CMD ["sh", "-c", "exec uvicorn backend.main:app --host ${HOST} --port ${PORT} --workers 1"]

