"""Unit tests for src.audio.preprocess (V1 in validation.md)."""
from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import pytest
import torch
import torchaudio

from src.audio.preprocess import (
    CLIP_SAMPLES,
    SAMPLE_RATE,
    load_and_preprocess,
    preprocess_bytes,
    preprocess_tensor,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav_bytes(
    duration_s: float = 5.0,
    sr: int = 22050,
    channels: int = 1,
) -> bytes:
    """Create a minimal PCM WAV file in memory."""
    num_samples = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        # Silence
        wf.writeframes(b"\x00\x00" * num_samples * channels)
    return buf.getvalue()


def _save_wav(path: Path, duration_s: float, sr: int = 22050, channels: int = 1) -> None:
    wav_bytes = _make_wav_bytes(duration_s, sr, channels)
    path.write_bytes(wav_bytes)


# ── V1.1 — output length is exactly CLIP_SAMPLES ─────────────────────────────

def test_output_length_exact(tmp_path: Path) -> None:
    """V1.1 — output tensor has exactly CLIP_SAMPLES samples (5 s @ 24 kHz)."""
    f = tmp_path / "test.wav"
    _save_wav(f, duration_s=5.0, sr=SAMPLE_RATE)
    out = load_and_preprocess(str(f))
    assert out.shape == (CLIP_SAMPLES,), f"Expected {CLIP_SAMPLES}, got {out.shape[0]}"


# ── V1.2 — stereo → mono ──────────────────────────────────────────────────────

def test_stereo_to_mono(tmp_path: Path) -> None:
    """V1.2 — stereo input is collapsed to a 1-D output tensor."""
    f = tmp_path / "stereo.wav"
    _save_wav(f, duration_s=5.0, sr=SAMPLE_RATE, channels=2)
    out = load_and_preprocess(str(f))
    assert out.ndim == 1, f"Expected 1-D tensor, got {out.ndim}-D"


# ── V1.3 — short clip is zero-padded ─────────────────────────────────────────

def test_short_clip_zero_padded(tmp_path: Path) -> None:
    """V1.3 — a 2-second clip is zero-padded to CLIP_SAMPLES."""
    f = tmp_path / "short.wav"
    _save_wav(f, duration_s=2.0, sr=SAMPLE_RATE)
    out = load_and_preprocess(str(f))
    assert out.shape[0] == CLIP_SAMPLES


# ── V1.4 — long clip is truncated ────────────────────────────────────────────

def test_long_clip_truncated(tmp_path: Path) -> None:
    """V1.4 — a 30-second clip is truncated to CLIP_SAMPLES."""
    f = tmp_path / "long.wav"
    _save_wav(f, duration_s=30.0, sr=SAMPLE_RATE)
    out = load_and_preprocess(str(f))
    assert out.shape[0] == CLIP_SAMPLES


# ── V1.5 — peak normalisation ────────────────────────────────────────────────

def test_peak_normalisation(tmp_path: Path) -> None:
    """V1.5 — output values are within [-1, 1]."""
    f = tmp_path / "nonsilent.wav"
    # Write a non-silent sine wave so normalisation actually fires
    sr = SAMPLE_RATE
    t = torch.linspace(0, 5.0, CLIP_SAMPLES)
    sine = (torch.sin(2 * 3.14159 * 440 * t) * 16000).short()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sine.numpy().tobytes())
    f.write_bytes(buf.getvalue())

    out = load_and_preprocess(str(f))
    assert out.abs().max().item() <= 1.0 + 1e-6


# ── V1.6 — format support ─────────────────────────────────────────────────────

def test_wav_format_accepted(tmp_path: Path) -> None:
    """V1.6a — WAV files are accepted without error."""
    f = tmp_path / "test.wav"
    _save_wav(f, duration_s=5.0)
    out = load_and_preprocess(str(f))
    assert out.shape[0] == CLIP_SAMPLES


def test_preprocess_bytes_roundtrip(tmp_path: Path) -> None:
    """V1.6b — preprocess_bytes produces the same result as load_and_preprocess."""
    wav_bytes = _make_wav_bytes(duration_s=5.0, sr=SAMPLE_RATE)
    # Save to file and compare
    f = tmp_path / "test.wav"
    f.write_bytes(wav_bytes)
    from_file = load_and_preprocess(str(f))
    from_bytes = preprocess_bytes(wav_bytes)
    assert torch.allclose(from_file, from_bytes, atol=1e-5)


# ── Resampling ────────────────────────────────────────────────────────────────

def test_resampling_from_22khz(tmp_path: Path) -> None:
    """Waveform at 22050 Hz is correctly resampled to 24000 Hz."""
    f = tmp_path / "22khz.wav"
    _save_wav(f, duration_s=5.0, sr=22050)
    out = load_and_preprocess(str(f))
    assert out.shape[0] == CLIP_SAMPLES


# ── dtype ─────────────────────────────────────────────────────────────────────

def test_output_dtype(tmp_path: Path) -> None:
    """Output tensor must be float32."""
    f = tmp_path / "test.wav"
    _save_wav(f, duration_s=5.0, sr=SAMPLE_RATE)
    out = load_and_preprocess(str(f))
    assert out.dtype == torch.float32
