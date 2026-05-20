"""Subgenre classification head on top of MERT embeddings.

Architecture (from tech.md):
    LayerNorm(768) → Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → N)

The model always returns **exactly 3** top-k predictions (FR-3 / FR-5).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TOP_K: int = 3


class SubgenreClassifier(nn.Module):
    """MLP classification head for subgenre prediction.

    The head is designed to sit on top of frozen MERT embeddings.  Only
    this module's parameters are updated during fine-tuning.

    Args:
        num_classes: Number of output classes.
        embed_dim:   Dimensionality of the input embedding (default 768).
    """

    def __init__(self, num_classes: int, embed_dim: int = 768) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, num_classes)``."""
        return self.head(x)

    # ------------------------------------------------------------------
    # Inference helper — always returns top-3
    # ------------------------------------------------------------------

    def predict_top3(
        self,
        embedding: torch.Tensor,
        label_names: list[str],
    ) -> list[dict]:
        """Return the top-3 predictions for a single embedding.

        Args:
            embedding:   Tensor of shape ``(1, embed_dim)``.
            label_names: Ordered list of class-name strings whose indices
                         correspond to the model's output indices.

        Returns:
            List of exactly 3 dicts::

                [
                    {"label": "techno",  "confidence": 0.82},
                    {"label": "house",   "confidence": 0.11},
                    {"label": "trance",  "confidence": 0.04},
                ]

            Sorted highest-to-lowest confidence.  Confidences are softmax
            probabilities and sum to ≤ 1 across the full output space.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(embedding)                    # (1, N)
            probs = F.softmax(logits, dim=-1).squeeze(0)        # (N,)
            top_probs, top_indices = torch.topk(probs, k=TOP_K)

        return [
            {
                "label": label_names[idx.item()],
                "confidence": round(prob.item(), 4),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save classifier weights and metadata to a .pt file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> "SubgenreClassifier":
        """Load a classifier from a .pt checkpoint.

        Args:
            path:   Path to the checkpoint written by :meth:`save`.
            device: Device to map the weights onto.

        Returns:
            Classifier in eval mode on ``device``.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_classes=checkpoint["num_classes"],
            embed_dim=checkpoint["embed_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
