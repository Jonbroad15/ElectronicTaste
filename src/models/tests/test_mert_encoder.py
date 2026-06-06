"""Unit tests for MERTEncoder (V2 in validation.md).

Tests that require the actual MERT model download are marked ``@pytest.mark.slow``
and skipped by default.  Run them with::

    pytest -m slow src/models/tests/test_mert_encoder.py
"""
from __future__ import annotations

import pytest
import torch

from src.models.mert_encoder import MERTEncoder, get_device, EMBED_DIM, SAMPLE_RATE
from src.audio.preprocess import CLIP_SAMPLES


# ── get_device ────────────────────────────────────────────────────────────────

def test_get_device_cpu_override() -> None:
    """Forcing 'cpu' returns a CPU device."""
    device = get_device("cpu")
    assert device.type == "cpu"


def test_get_device_auto_returns_device() -> None:
    """Auto-detection returns a valid device object."""
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}


# ── MERTEncoder (requires model download — marked slow) ──────────────────────

@pytest.mark.slow
class TestMERTEncoder:
    """Integration tests that load the real MERT model."""

    @pytest.fixture(scope="class")
    def encoder(self) -> MERTEncoder:
        return MERTEncoder(device="cpu")  # CPU for CI determinism

    @pytest.fixture()
    def waveform(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(CLIP_SAMPLES)

    def test_v2_1_output_shape(self, encoder: MERTEncoder, waveform: torch.Tensor) -> None:
        """V2.1 — output embedding shape is (1, 768)."""
        emb = encoder.extract_embedding(waveform)
        assert emb.shape == (1, EMBED_DIM), f"Expected (1, {EMBED_DIM}), got {emb.shape}"

    def test_v2_2_encoder_frozen(self, encoder: MERTEncoder) -> None:
        """V2.2 — all MERT parameters have requires_grad=False."""
        for name, param in encoder.model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen"

    def test_v2_3_cpu_device(self, encoder: MERTEncoder, waveform: torch.Tensor) -> None:
        """V2.3 — embedding is returned on CPU."""
        emb = encoder.extract_embedding(waveform)
        assert emb.device.type == "cpu"

    def test_v2_4_determinism(self, encoder: MERTEncoder, waveform: torch.Tensor) -> None:
        """V2.4 — identical input produces identical embedding."""
        emb1 = encoder.extract_embedding(waveform)
        emb2 = encoder.extract_embedding(waveform)
        assert torch.allclose(emb1, emb2), "Non-deterministic output from MERT encoder"

    def test_output_dtype(self, encoder: MERTEncoder, waveform: torch.Tensor) -> None:
        """Embedding is float32."""
        emb = encoder.extract_embedding(waveform)
        assert emb.dtype == torch.float32
