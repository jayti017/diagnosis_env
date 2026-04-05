# DiagnosisEnv — Dockerfile
# ==========================
# Builds and runs the OpenEnv-compliant DiagnosisEnv FastAPI server.
# Compatible with Hugging Face Spaces (port 7860).
#
# Build:  docker build -t diagnosis-env .
# Run:    docker run -p 7860:7860 \
#           -e API_BASE_URL=<url> \
#           -e MODEL_NAME=<model> \
#           -e HF_TOKEN=<token> \
#           diagnosis-env

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY . .

# ── Environment defaults (override at runtime) ────────────────────────────
ENV PORT=7860
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV SERVER_URL="http://localhost:7860"

# ── Expose port ───────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]