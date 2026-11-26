# AwaazGuard app image: website (static Next.js export) + API gateway + both ONNX models.
# Designed to run on a free 1 GB virtual machine.

# ---------- stage 1: build the website ----------
FROM node:20-alpine AS web
WORKDIR /web
ENV NEXT_TELEMETRY_DISABLED=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build                      # -> /web/out (static files)

# ---------- stage 2: Python app ----------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 MALLOC_ARENA_MAX=2 \
    FRONTEND_DIR=/app/frontend \
    DATABASE_URL=sqlite:////data/awaazguard.db \
    AUDIO_STORE_DIR=/data/audio \
    DRIFT_REPORT_DIR=/data/drift

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY params.yaml .
COPY src/ src/
COPY --from=web /web/out /app/frontend

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown -R app /data /app
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD curl -fsS http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "src.serving.gateway_app:app", "--host", "0.0.0.0", "--port", "8000"]
