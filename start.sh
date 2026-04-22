#!/usr/bin/env bash
# start.sh — Render start command for EfficientNet-B4 API
# Render calls this instead of running uvicorn directly.
# It downloads the model FIRST, then starts the server.

set -e   # exit immediately on any error

echo "============================================"
echo "  EfficientNet-B4 API — Render startup"
echo "============================================"

# Step 1: Download model from Google Drive (skipped if already cached)
echo "[start] Running model download …"
python download_model.py

# Step 2: Start FastAPI with uvicorn
# PORT is injected by Render automatically
echo "[start] Starting uvicorn on port ${PORT:-8000} …"
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --timeout-keep-alive 30
