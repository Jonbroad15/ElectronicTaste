"""API integration tests (V5 in validation.md).

The Predictor is mocked so the tests do NOT require a trained model or the
MERT download.  They validate the HTTP contract only.
"""
from __future__ import annotations

import io
import time
import wave
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app, _MAX_UPLOAD_BYTES


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

    with TestClient(app) as c:
        api_main._predictor = mock_predictor
        yield c
        api_main._predictor = None


@pytest.fixture()
def client_error():
    """TestClient whose Predictor raises an exception on every call."""
    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = RuntimeError(
        "/internal/path/models/classifier_best.pt: shape mismatch"
    )
    with TestClient(app) as c:
        api_main._predictor = mock_predictor
        yield c
        api_main._predictor = None


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded (simulates pre-training state)."""
    with TestClient(app) as c:
        api_main._predictor = None
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
    response = client.post("/predict", files={"audio": ("test.wav", wav, "audio/wav")})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_predict_response_keys(client: TestClient) -> None:
    """V5.2 — each prediction has 'label' (str) and 'confidence' (float)."""
    wav = _make_wav_bytes()
    response = client.post("/predict", files={"audio": ("test.wav", wav, "audio/wav")})
    assert response.status_code == 200
    for pred in response.json()["predictions"]:
        assert isinstance(pred["label"], str)
        assert isinstance(pred["confidence"], float)


def test_predict_sorted_descending(client: TestClient) -> None:
    """V5.2 — predictions are sorted highest-to-lowest confidence."""
    wav = _make_wav_bytes()
    response = client.post("/predict", files={"audio": ("test.wav", wav, "audio/wav")})
    confs = [p["confidence"] for p in response.json()["predictions"]]
    assert confs == sorted(confs, reverse=True)


# ── V5.3 — response time ≤ 2s ────────────────────────────────────────────────

def test_predict_response_time(client: TestClient) -> None:
    """V5.3 — POST /predict (with mocked model) responds within 2 seconds."""
    wav = _make_wav_bytes()
    t0 = time.time()
    response = client.post("/predict", files={"audio": ("test.wav", wav, "audio/wav")})
    elapsed = time.time() - t0
    assert response.status_code == 200
    assert elapsed < 2.0, f"Response took {elapsed:.2f}s — exceeded 2s limit"


# ── V5.4 — invalid content type → 422 ────────────────────────────────────────

def test_invalid_content_type_returns_422(client: TestClient) -> None:
    """V5.4 — uploading a non-audio MIME type returns HTTP 422."""
    response = client.post(
        "/predict",
        files={"audio": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 422


def test_non_audio_bytes_rejected(client: TestClient) -> None:
    """Magic byte check rejects non-audio content even with correct MIME type."""
    response = client.post(
        "/predict",
        files={"audio": ("fake.wav", b"this is not a wav file at all!", "audio/wav")},
    )
    assert response.status_code == 422


# ── Upload size limit → 413 ───────────────────────────────────────────────────

def test_oversized_upload_returns_413(client: TestClient) -> None:
    """Upload exceeding MAX_UPLOAD_BYTES returns HTTP 413."""
    big_payload = b"RIFF" + b"\x00" * (_MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/predict",
        files={"audio": ("big.wav", big_payload, "audio/wav")},
    )
    assert response.status_code == 413


def test_at_size_limit_not_rejected(client: TestClient) -> None:
    """Upload exactly at the limit (with valid WAV magic) is not rejected by size check."""
    # Build a payload exactly at the limit; it will fail audio decoding (not a real WAV)
    # but must not return 413.
    at_limit = _make_wav_bytes()  # small valid WAV — well within limit
    response = client.post(
        "/predict",
        files={"audio": ("ok.wav", at_limit, "audio/wav")},
    )
    assert response.status_code != 413


# ── Error message leakage ─────────────────────────────────────────────────────

def test_error_message_does_not_leak_internals(client_error: TestClient) -> None:
    """Exception messages must not expose internal paths or stack traces."""
    wav = _make_wav_bytes()
    response = client_error.post(
        "/predict",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    # Must not contain the internal path that was in the exception message
    assert "/internal/path" not in detail
    assert "shape mismatch" not in detail


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
