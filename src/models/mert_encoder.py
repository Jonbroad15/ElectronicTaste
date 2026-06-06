"""MERT encoder wrapper — extracts 768-dim embeddings from raw audio.

The encoder is always frozen (no gradients).  Embeddings are produced by
mean-pooling the last hidden state of MERT-v1-95M over the time dimension.

Device priority: MPS (Apple Silicon) → CUDA → CPU.

Security note on trust_remote_code
------------------------------------
``trust_remote_code=True`` is required by the MERT model and executes Python
code shipped with the HuggingFace repository at load time.  In production:
  1. Pin a specific commit hash in the model ID (e.g. ``m-a-p/MERT-v1-95M@<sha>``).
  2. Set ``TRANSFORMERS_OFFLINE=1`` after the model is cached locally to
     prevent silent updates from running new code on server restarts.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODEL_ID: str = "m-a-p/MERT-v1-95M"
SAMPLE_RATE: int = 24_000
EMBED_DIM: int = 768

_VALID_DEVICE_TYPES = {"cpu", "cuda", "mps"}


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available device.

    Args:
        preferred: If given, force this device string (``"mps"``, ``"cuda"``,
                   ``"cpu"``).  Otherwise auto-select MPS → CUDA → CPU.

    Raises:
        ValueError: if ``preferred`` is not a recognised device type.
    """
    if preferred:
        base = preferred.split(":")[0]  # handle "cuda:0" style strings
        if base not in _VALID_DEVICE_TYPES:
            raise ValueError(
                f"Unknown device {preferred!r}. "
                f"Choose from: {sorted(_VALID_DEVICE_TYPES)}"
            )
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MERTEncoder:
    """Thin wrapper around MERT-v1-95M for audio embedding extraction.

    Usage::

        encoder = MERTEncoder()
        waveform = load_and_preprocess("track.wav")           # 1-D, 24 kHz
        embedding = encoder.extract_embedding(waveform)       # (1, 768)

        # Faster batched path for training:
        embeddings = encoder.extract_embeddings_batch(waveforms)  # (B, 768)
    """

    def __init__(
        self,
        model_id: str = MERT_MODEL_ID,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = get_device(str(device) if device else None)
        self.model_id = model_id

        self.processor: Wav2Vec2FeatureExtractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        )
        self.model: AutoModel = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)

        # Freeze all encoder parameters — only the classification head trains.
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a single 768-dim embedding from a preprocessed waveform.

        For bulk extraction during training prefer :meth:`extract_embeddings_batch`
        which processes a whole DataLoader batch in one GPU forward pass.

        Args:
            waveform: 1-D float32 tensor of shape ``(samples,)`` sampled at
                      24 kHz, as returned by
                      :func:`src.audio.preprocess.load_and_preprocess`.

        Returns:
            Tensor of shape ``(1, 768)`` on CPU.
        """
        return self.extract_embeddings_batch([waveform])  # reuse batched path

    def extract_embeddings_batch(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        """Extract embeddings for a batch of waveforms in a single GPU forward pass.

        This is significantly faster than calling :meth:`extract_embedding`
        in a loop because the processor and MERT forward pass execute once
        for the entire batch.

        Args:
            waveforms: List of 1-D float32 tensors, each of shape ``(samples,)``
                       at 24 kHz.  All tensors are expected to have the same
                       length (the preprocessing pipeline guarantees this).

        Returns:
            Float tensor of shape ``(B, 768)`` on CPU.
        """
        inputs = self.processor(
            [w.cpu().numpy() for w in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # last_hidden_state: (B, T, 768) → mean over T → (B, 768)
        last_hidden = outputs.hidden_states[-1]
        embeddings = last_hidden.mean(dim=1).cpu()
        return embeddings
