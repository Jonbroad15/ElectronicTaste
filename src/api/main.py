"""FastAPI prediction service.

Endpoints
---------
GET  /health        — liveness check
POST /predict       — upload an audio file, receive top-3 subgenre predictions

Environment variables (all optional)
--------------------------------------
CLASSIFIER_PATH       Path to classifier checkpoint (default: models/classifier_best.pt)
LABEL_ENCODER_PATH    Path to label encoder JSON  (default: models/label_encoder.json)
MERT_MODEL_ID         HuggingFace model ID        (default: m-a-p/MERT-v1-95M)
DEVICE                Force device: mps/cuda/cpu  (default: auto)

Run locally::

    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.predict import Predictor

# ── Startup / shutdown ────────────────────────────────────────────────────────

_predictor: Predictor | None = None

_CLASSIFIER_PATH    = os.getenv("CLASSIFIER_PATH",    "models/classifier_best.pt")
_LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "models/label_encoder.json")
_MERT_MODEL_ID      = os.getenv("MERT_MODEL_ID",      "m-a-p/MERT-v1-95M")
_DEVICE             = os.getenv("DEVICE",              None)

_ACCEPTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "application/octet-stream",  # fallback when content-type is unset
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release resources on shutdown."""
    global _predictor
    import os as _os
    if _os.path.exists(_CLASSIFIER_PATH) and _os.path.exists(_LABEL_ENCODER_PATH):
        _predictor = Predictor(
            classifier_path=_CLASSIFIER_PATH,
            label_encoder_path=_LABEL_ENCODER_PATH,
            mert_model_id=_MERT_MODEL_ID,
            device=_DEVICE,
        )
    else:
        # Model not yet trained — /predict will return 503 until it exists
        _predictor = None
    yield
    _predictor = None


app = FastAPI(
    title="Electronic Taste — Subgenre Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Response schemas ──────────────────────────────────────────────────────────

class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Return service liveness and the loaded model ID."""
    return HealthResponse(status="ok", model=_MERT_MODEL_ID)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    audio: Annotated[UploadFile, File(description="WAV, MP3, or FLAC audio file")]
) -> PredictResponse:
    """Classify an audio clip and return the top-3 subgenre predictions.

    - Upload a WAV / MP3 / FLAC file (ideally ≥ 5 seconds).
    - The first 5 seconds are used; shorter clips are zero-padded.
    - Returns exactly **3** predictions sorted by descending confidence.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a classifier first (see src/training/train.py).",
        )

    content_type = audio.content_type or ""
    if content_type and content_type not in _ACCEPTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type: {content_type!r}. "
                   "Upload a WAV, MP3, or FLAC file.",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        results = _predictor.predict(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {exc}") from exc

    return PredictResponse(predictions=[Prediction(**r) for r in results])
