"""Audio preprocessing utilities.

All audio is normalised to 24 kHz mono, clipped/padded to exactly 5 seconds,
and peak-normalised to [-1, 1] before being handed to the MERT encoder.
"""
from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE: int = 24_000
CLIP_SECONDS: float = 5.0
CLIP_SAMPLES: int = int(SAMPLE_RATE * CLIP_SECONDS)  # 120 000 samples


def preprocess_tensor(
    waveform: torch.Tensor,
    sr: int,
    target_sr: int = SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
) -> torch.Tensor:
    """Apply the full preprocessing pipeline to an already-loaded waveform.

    Args:
        waveform: Tensor of shape ``(channels, samples)`` as returned by
                  ``torchaudio.load``.
        sr:       Original sample rate of ``waveform``.
        target_sr: Desired output sample rate (default 24 kHz for MERT).
        clip_seconds: Length of the output window in seconds.

    Returns:
        1-D float32 tensor of shape ``(target_sr * clip_seconds,)`` with
        values in ``[-1, 1]``.
    """
    # ── Stereo → mono ─────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, samples)

    # ── Resample ──────────────────────────────────────────────────
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # ── Flatten to 1-D ────────────────────────────────────────────
    waveform = waveform.squeeze(0)  # (samples,)

    clip_samples = int(clip_seconds * target_sr)

    # ── Clip or zero-pad ──────────────────────────────────────────
    if waveform.shape[0] > clip_samples:
        waveform = waveform[:clip_samples]
    elif waveform.shape[0] < clip_samples:
        pad = torch.zeros(clip_samples - waveform.shape[0], dtype=waveform.dtype)
        waveform = torch.cat([waveform, pad])

    # ── Peak normalise ────────────────────────────────────────────
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform.float()


def load_and_preprocess(
    path: str,
    target_sr: int = SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
) -> torch.Tensor:
    """Load an audio file from disk and return a preprocessed 1-D tensor.

    Supports any format understood by torchaudio (WAV, MP3, FLAC, …).

    Args:
        path:        Absolute or relative path to the audio file.
        target_sr:   Desired sample rate (default 24 kHz).
        clip_seconds: Length of the output window in seconds (default 5 s).

    Returns:
        1-D float32 tensor of shape ``(target_sr * clip_seconds,)``.
    """
    waveform, sr = torchaudio.load(path)
    return preprocess_tensor(waveform, sr, target_sr=target_sr, clip_seconds=clip_seconds)


def preprocess_bytes(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
) -> torch.Tensor:
    """Load audio from raw bytes (WAV/MP3/FLAC) and preprocess.

    Used by the API endpoint, which receives audio as an uploaded file.

    Returns:
        1-D float32 tensor of shape ``(target_sr * clip_seconds,)``.
    """
    import io
    buffer = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buffer)
    return preprocess_tensor(waveform, sr, target_sr=target_sr, clip_seconds=clip_seconds)
