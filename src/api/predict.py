"""Core inference logic: audio bytes → top-3 subgenre predictions.

The :class:`Predictor` is designed to be loaded once at API startup and
reused across requests (singleton pattern in ``main.py``).
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.audio.preprocess import preprocess_bytes
from src.models.mert_encoder import MERTEncoder, get_device
from src.models.classifier import SubgenreClassifier
from src.training.dataset import LabelEncoder


class Predictor:
    """Holds loaded MERT + classifier + label encoder; serves predictions.

    Args:
        classifier_path:    Path to a ``models/classifier_best.pt`` checkpoint.
        label_encoder_path: Path to ``models/label_encoder.json``.
        mert_model_id:      HuggingFace model ID for MERT (default 95M).
        device:             Override device selection.
    """

    def __init__(
        self,
        classifier_path: str | Path,
        label_encoder_path: str | Path,
        mert_model_id: str = "m-a-p/MERT-v1-95M",
        device: torch.device | str | None = None,
    ) -> None:
        self.device = get_device(str(device) if device else None)
        self.encoder = MERTEncoder(model_id=mert_model_id, device=self.device)
        self.classifier = SubgenreClassifier.load(classifier_path, device=self.device)
        self.classifier.eval()
        self.label_encoder = LabelEncoder.load(label_encoder_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, audio_bytes: bytes) -> list[dict]:
        """Run the full inference pipeline on raw audio bytes.

        Steps:
            1. Decode and preprocess the audio (resample → clip → normalise).
            2. Extract a 768-dim MERT embedding.
            3. Run the classification head and return the top-3 results.

        Args:
            audio_bytes: Raw bytes of a WAV, MP3, or FLAC file.

        Returns:
            List of **exactly 3** dicts::

                [
                    {"label": "techno",  "confidence": 0.82},
                    {"label": "house",   "confidence": 0.11},
                    {"label": "trance",  "confidence": 0.04},
                ]

            Sorted highest-to-lowest confidence.
        """
        waveform = preprocess_bytes(audio_bytes)
        embedding = self.encoder.extract_embedding(waveform).to(self.device)
        return self.classifier.predict_top3(embedding, self.label_encoder.names)
