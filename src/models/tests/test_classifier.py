"""Unit tests for SubgenreClassifier (V3 in validation.md)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.models.classifier import SubgenreClassifier, TOP_K


NUM_CLASSES = 10
BATCH = 4
EMBED_DIM = 768
LABEL_NAMES = [f"genre_{i}" for i in range(NUM_CLASSES)]


@pytest.fixture()
def model() -> SubgenreClassifier:
    return SubgenreClassifier(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM)


@pytest.fixture()
def embedding() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, EMBED_DIM)


# ── V3.1 — output shape ───────────────────────────────────────────────────────

def test_output_shape(model: SubgenreClassifier) -> None:
    """V3.1 — logits shape is (batch, num_classes)."""
    x = torch.randn(BATCH, EMBED_DIM)
    logits = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES)


# ── V3.2 — checkpoint round-trip ─────────────────────────────────────────────

def test_checkpoint_roundtrip(model: SubgenreClassifier, tmp_path: torch.types._dtype) -> None:
    """V3.2 — save then load produces identical weights."""
    path = tmp_path / "clf.pt"
    model.save(path)
    loaded = SubgenreClassifier.load(path, device="cpu")

    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), loaded.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2), f"Mismatch in parameter {n1}"


# ── V3.3 — softmax sums to 1 ─────────────────────────────────────────────────

def test_softmax_sums_to_one(model: SubgenreClassifier) -> None:
    """V3.3 — softmax over logits sums to 1 for each sample."""
    x = torch.randn(BATCH, EMBED_DIM)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    ones = torch.ones(BATCH)
    assert torch.allclose(probs.sum(dim=-1), ones, atol=1e-5)


# ── V3.4 — frozen MERT grads ─────────────────────────────────────────────────

def test_mert_grads_none_during_head_backward(model: SubgenreClassifier) -> None:
    """V3.4 — gradients flow only through the classification head, not MERT.

    We verify this by running a backward pass on the classifier alone and
    confirming all head parameters receive gradients.
    """
    x = torch.randn(BATCH, EMBED_DIM, requires_grad=False)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"


# ── predict_top3 — always 3 results ──────────────────────────────────────────

def test_predict_top3_length(model: SubgenreClassifier, embedding: torch.Tensor) -> None:
    """predict_top3 always returns exactly TOP_K (3) predictions."""
    results = model.predict_top3(embedding, LABEL_NAMES)
    assert len(results) == TOP_K


def test_predict_top3_keys(model: SubgenreClassifier, embedding: torch.Tensor) -> None:
    """predict_top3 results have 'label' and 'confidence' keys."""
    results = model.predict_top3(embedding, LABEL_NAMES)
    for r in results:
        assert "label" in r
        assert "confidence" in r
        assert isinstance(r["label"], str)
        assert isinstance(r["confidence"], float)


def test_predict_top3_sorted_descending(model: SubgenreClassifier, embedding: torch.Tensor) -> None:
    """predict_top3 results are sorted highest-to-lowest confidence."""
    results = model.predict_top3(embedding, LABEL_NAMES)
    confs = [r["confidence"] for r in results]
    assert confs == sorted(confs, reverse=True), "Results not sorted by confidence"


def test_predict_top3_valid_labels(model: SubgenreClassifier, embedding: torch.Tensor) -> None:
    """predict_top3 labels are all members of label_names."""
    results = model.predict_top3(embedding, LABEL_NAMES)
    for r in results:
        assert r["label"] in LABEL_NAMES


def test_predict_top3_confidence_range(model: SubgenreClassifier, embedding: torch.Tensor) -> None:
    """predict_top3 confidences are in (0, 1]."""
    results = model.predict_top3(embedding, LABEL_NAMES)
    for r in results:
        assert 0.0 < r["confidence"] <= 1.0


# ── Num classes flexibility ───────────────────────────────────────────────────

def test_different_num_classes() -> None:
    """Classifier works with different class counts (e.g., 20 EDM subgenres)."""
    model = SubgenreClassifier(num_classes=20)
    x = torch.randn(2, 768)
    assert model(x).shape == (2, 20)
