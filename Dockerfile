# ============================================================
#  Alpha Sniper v2 — Production Docker Image
#  Multi-stage build · Non-root user · Health check
# ============================================================

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for lxml / numpy wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──
FROM python:3.11-slim AS runtime

# Non-root user
RUN groupadd -r sniper && useradd -r -g sniper -d /app -s /sbin/nologin sniper

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/

# Own everything to sniper
RUN chown -R sniper:sniper /app

USER sniper

# Expose port
EXPOSE 8000

# Health check hitting the /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
