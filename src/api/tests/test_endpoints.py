"""API integration tests (V5 in validation.md).

The Predictor is mocked so the tests do NOT require a trained model or the
MERT download.  They validate the HTTP contract only.
"""
from __future__ import annotations

import io
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

_FAKE_PREDICTIONS = [
    {"label": "techno",  "confidence": 0.82},
    {"label": "house",   "confidence": 0.11},
    {"label": "trance",  "confidence": 0.04},
]


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(duration_s * sr))
    return buf.getvalue()


@pytest.fixture()
def client():
    """TestClient with a mocked Predictor injected after lifespan startup."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = _FAKE_PREDICTIONS

    # The lifespan runs on TestClient.__enter__ and may set _predictor = None
    # (no model files on disk during tests).  We inject the mock afterwards.
    with TestClient(app) as c:
        api_main._predictor = mock_predictor
        yield c
        api_main._predictor = None  # reset after each test


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded (simulates pre-training state)."""
    with TestClient(app) as c:
        api_main._predictor = None  # ensure it stays None
        yield c


# ── V5.1 — /health returns 200 ───────────────────────────────────────────────

def test_health_returns_200(client: TestClient) -> None:
    """V5.1 — GET /health → 200 with status='ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


# ── V5.2 — /predict returns exactly 3 predictions ────────────────────────────

def test_predict_returns_exactly_3(client: TestClient) -> None:
    """V5.2 — POST /predict returns a list of exactly 3 predictions."""
    wav = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3


def test_predict_response_keys(client: TestClient) -> None:
    """V5.2 — each prediction has 'label' (str) and 'confidence' (float)."""
    wav = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    for pred in response.json()["predictions"]:
        assert isinstance(pred["label"], str)
        assert isinstance(pred["confidence"], float)


def test_predict_sorted_descending(client: TestClient) -> None:
    """V5.2 — predictions are sorted highest-to-lowest confidence."""
    wav = _make_wav_bytes()
    response = client.post(
        "/predict",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    confs = [p["confidence"] for p in response.json()["predictions"]]
    assert confs == sorted(confs, reverse=True), "Predictions not sorted by confidence"


# ── V5.4 — invalid file type → 422 ───────────────────────────────────────────

def test_invalid_content_type_returns_422(client: TestClient) -> None:
    """V5.4 — uploading a non-audio file returns HTTP 422."""
    response = client.post(
        "/predict",
        files={"audio": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 422


# ── No model loaded → 503 ────────────────────────────────────────────────────

def test_predict_503_when_no_model(client_no_model: TestClient) -> None:
    """If no model is loaded (pre-training), /predict returns 503."""
    wav = _make_wav_bytes()
    response = client_no_model.post(
        "/predict",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert response.status_code == 503


# ── V5.5 — concurrent requests all succeed ───────────────────────────────────

def test_concurrent_requests(client: TestClient) -> None:
    """V5.5 — 5 simultaneous requests all return 200."""
    import concurrent.futures

    wav = _make_wav_bytes()

    def call():
        return client.post(
            "/predict",
            files={"audio": ("test.wav", wav, "audio/wav")},
        ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        statuses = list(pool.map(lambda _: call(), range(5)))

    assert all(s == 200 for s in statuses), f"Some requests failed: {statuses}"
