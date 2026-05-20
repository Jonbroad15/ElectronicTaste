"""MERT encoder wrapper — extracts 768-dim embeddings from raw audio.

The encoder is always frozen (no gradients).  Embeddings are produced by
mean-pooling the last hidden state of MERT-v1-95M over the time dimension.

Device priority: MPS (Apple Silicon) → CUDA → CPU.
"""
from __future__ import annotations

import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODEL_ID: str = "m-a-p/MERT-v1-95M"
SAMPLE_RATE: int = 24_000
EMBED_DIM: int = 768


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available device.

    Args:
        preferred: If given, force this device string (``"mps"``, ``"cuda"``,
                   ``"cpu"``).  Otherwise auto-select MPS → CUDA → CPU.
    """
    if preferred:
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
        waveform = load_and_preprocess("track.wav")   # 1-D, 24 kHz
        embedding = encoder.extract_embedding(waveform)  # (1, 768)
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

        Args:
            waveform: 1-D float32 tensor of shape ``(samples,)`` sampled at
                      24 kHz, as returned by
                      :func:`src.audio.preprocess.load_and_preprocess`.

        Returns:
            Tensor of shape ``(1, 768)`` on CPU.
        """
        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # last_hidden_state: (1, T, 768) → mean over T → (1, 768)
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden.mean(dim=1).cpu()
        return embedding
