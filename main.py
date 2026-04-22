"""
main.py — EfficientNet-B4 Deepfake Detection API
Deployment: Render Web Service (CPU, free tier)

Endpoints:
  GET  /health   → readiness check
  POST /predict  → upload image → REAL/FAKE JSON
"""

import os
import gc
import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_model, DEFAULT_CONFIG
from inference import load_face_net, predict

# ── Memory savings for CPU deployment ────────────────────────────────────────
torch.set_num_threads(2)              # Render free = 0.1 CPU; don't over-subscribe
torch.set_grad_enabled(False)         # global — no training, ever

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("efficientnet-api")

# ── Paths (set via env vars on Render dashboard) ───────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",    "models/best_model_v3.pth")
DETECTOR_DIR = os.getenv("DETECTOR_DIR",  "models/face_detector")

ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png",
                "image/bmp", "image/webp", "image/tiff"}
MAX_BYTES    = 15 * 1024 * 1024   # 15 MB — keep RAM pressure low

_state: dict = {"model": None, "face_net": None, "threshold": 0.46, "ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("EfficientNet-B4 API — starting up (CPU)")
    logger.info("=" * 50)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Checkpoint missing: {MODEL_PATH}")
        logger.error("Make sure start.sh ran download_model.py first.")
    else:
        t0 = time.time()
        model, threshold = load_model(MODEL_PATH)
        _state["model"]     = model
        _state["threshold"] = threshold
        logger.info(f"Model loaded in {time.time()-t0:.1f}s  threshold={threshold:.2f}")

        _state["face_net"] = load_face_net(DETECTOR_DIR)
        logger.info("Face detector ready")
        _state["ready"]    = True
        logger.info("API READY")

    gc.collect()
    yield

    logger.info("Shutting down …")
    _state["model"] = _state["face_net"] = None
    gc.collect()


app = FastAPI(
    title="Deepfake Detection — EfficientNet-B4",
    description="Image deepfake detection using EfficientNet-B4 (78 MB .pth model).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class PredictResponse(BaseModel):
    label        : str
    confidence   : float
    fake_prob    : float
    real_prob    : float
    face_detected: bool
    threshold    : float
    filename     : str
    processing_ms: float


@app.get("/", tags=["Info"])
def root():
    return {
        "service"   : "EfficientNet-B4 Deepfake Detection",
        "model"     : DEFAULT_CONFIG["model_name"],
        "image_size": DEFAULT_CONFIG["image_size"],
        "ready"     : _state["ready"],
        "docs"      : "/docs",
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status"   : "ok" if _state["ready"] else "not_ready",
        "ready"    : _state["ready"],
        "device"   : "cpu",
        "threshold": _state["threshold"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(
    file: UploadFile = File(..., description="Face image (JPG/PNG/BMP/WEBP/TIFF)")
):
    if not _state["ready"]:
        raise HTTPException(503, "Model not ready. Try again in a moment.")

    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported type '{ct}'. Use: {sorted(ALLOWED_MIME)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large ({len(data)//1024} KB). Max 15 MB.")

    t0 = time.perf_counter()
    try:
        result = predict(data, _state["model"], _state["face_net"], _state["threshold"])
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {e}")

    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"{file.filename} → {result['label']} ({result['confidence']:.1f}%) {ms}ms")

    return PredictResponse(**result, filename=file.filename or "unknown", processing_ms=ms)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
