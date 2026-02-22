"""
FastAPI Inference Service for Cats vs Dogs Classification.

Endpoints:
  GET  /health   → Health check
  POST /predict  → Predict cat or dog from uploaded image
  GET  /metrics  → Prometheus metrics
"""

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.predictor import Predictor
from app.schemas import HealthResponse, PredictionResponse

# ── Structured JSON-like logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger("catdog-api")

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "catdog_request_total",
    "Total number of requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "catdog_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTION_LABELS = Counter(
    "catdog_prediction_label_total",
    "Count of predicted labels",
    ["label"],
)

# ── App lifespan (load model once) ───────────────────────────────────────────
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    try:
        predictor = Predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification API for a pet adoption platform.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint — returns service and model status."""
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/health").observe(time.time() - start)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None and predictor.loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Image file (jpg/png)")):
    """
    Accept an image upload and return cat/dog classification.

    - **file**: Image file (JPEG or PNG recommended)
    """
    start = time.time()

    if predictor is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        REQUEST_COUNT.labels(endpoint="/predict", status="400").inc()
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="400").inc()
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    result = predictor.predict(image)

    latency = time.time() - start
    REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
    PREDICTION_LABELS.labels(label=result["label"]).inc()

    logger.info(
        f"predict | label={result['label']} "
        f"confidence={result['confidence']:.4f} "
        f"latency={latency:.3f}s "
        f"file={file.filename}"
    )

    return PredictionResponse(**result)


@app.get("/metrics", tags=["System"], include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
