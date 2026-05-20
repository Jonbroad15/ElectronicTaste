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

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.predict import Predictor

logger = logging.getLogger(__name__)

# ── Startup / shutdown ────────────────────────────────────────────────────────

_predictor: Predictor | None = None

_CLASSIFIER_PATH    = os.getenv("CLASSIFIER_PATH",    "models/classifier_best.pt")
_LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "models/label_encoder.json")
_MERT_MODEL_ID      = os.getenv("MERT_MODEL_ID",      "m-a-p/MERT-v1-95M")
_DEVICE             = os.getenv("DEVICE",              None)

# Maximum accepted upload size (50 MB).  Larger files are rejected with 413
# before any audio decoding occurs.
_MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Accepted MIME types from the Content-Type header.
# Note: the header alone is insufficient for security — magic byte validation
# (see _sniff_audio) is also applied after reading the payload.
_ACCEPTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    # application/octet-stream intentionally removed — magic byte check covers it.
}


def _sniff_audio(data: bytes) -> bool:
    """Return True if ``data`` begins with a recognised audio file magic bytes.

    Checks WAV (RIFF), FLAC (fLaC), and MP3 (ID3 tag or frame-sync bytes).
    This is NOT a full format validation but catches obviously non-audio payloads.
    """
    if len(data) < 4:
        return False
    h = data[:4]
    return (
        h == b"RIFF"                                                  # WAV
        or h == b"fLaC"                                               # FLAC
        or h[:3] == b"ID3"                                            # MP3 with ID3v2 tag
        or h[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xe0"}  # MP3 frame sync
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release resources on shutdown."""
    global _predictor
    if os.path.exists(_CLASSIFIER_PATH) and os.path.exists(_LABEL_ENCODER_PATH):
        _predictor = Predictor(
            classifier_path=_CLASSIFIER_PATH,
            label_encoder_path=_LABEL_ENCODER_PATH,
            mert_model_id=_MERT_MODEL_ID,
            device=_DEVICE,
        )
    else:
        # Model not yet trained — /predict will return 503 until it exists.
        _predictor = None
    yield
    _predictor = None


app = FastAPI(
    title="Electronic Taste — Subgenre Prediction API",
    version="0.2.0",
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

    - Upload a WAV / MP3 / FLAC file (ideally ≥ 5 seconds; max 50 MB).
    - The first 5 seconds are used; shorter clips are zero-padded.
    - Returns exactly **3** predictions sorted by descending confidence.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a classifier first (see src/training/train.py).",
        )

    # ── Content-type pre-check (client-supplied; not trusted alone) ────
    content_type = audio.content_type or ""
    if content_type and content_type not in _ACCEPTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type: {content_type!r}. "
                   "Upload a WAV, MP3, or FLAC file.",
        )

    # ── Read payload with size cap ─────────────────────────────────────
    audio_bytes = await audio.read(_MAX_UPLOAD_BYTES + 1)
    if len(audio_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum accepted size is "
                   f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # ── Magic byte validation (defends against content-type spoofing) ──
    if not _sniff_audio(audio_bytes):
        raise HTTPException(
            status_code=422,
            detail="File does not appear to be a valid WAV, MP3, or FLAC audio file.",
        )

    # ── Inference ──────────────────────────────────────────────────────
    try:
        results = _predictor.predict(audio_bytes)
    except Exception as exc:
        # Log full details server-side; return a generic message to the caller
        # to avoid leaking internal paths or library version strings.
        logger.exception("Audio processing failed for upload %r", audio.filename)
        raise HTTPException(
            status_code=422,
            detail="Audio processing failed. Check server logs for details.",
        ) from exc

    return PredictResponse(predictions=[Prediction(**r) for r in results])
