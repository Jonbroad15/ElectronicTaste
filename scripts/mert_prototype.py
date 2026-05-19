#!/usr/bin/env python3
"""
MERT Genre Classification Prototype
====================================

Uses MERT-v1-95M (smaller variant, ~95M params) to extract audio embeddings
from the GTZAN dataset, then trains a simple classifier to predict genre.

This validates:
1. MERT runs locally on Apple Silicon (MPS) with 16GB unified memory
2. MERT embeddings contain enough information for genre classification
3. End-to-end audio → prediction pipeline works

Architecture:
  Audio WAV → MERT encoder → Mean-pooled embeddings → Linear classifier → Genre

Usage: python scripts/mert_prototype.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor


# ─── Configuration ────────────────────────────────────────────────────────────

MERT_MODEL = "m-a-p/MERT-v1-95M"  # 95M params, lighter than 330M
DATA_DIR = Path("data/gtzan_audio")
RESULTS_FILE = Path("data/mert_prototype_results.json")
SAMPLE_RATE = 24000  # MERT expects 24kHz
MAX_AUDIO_LENGTH = 5 * SAMPLE_RATE  # 5 seconds (MERT window)
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_SPLIT = 0.2
SEED = 42

GENRE_NAMES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_device():
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_audio_file(filepath, target_sr=SAMPLE_RATE):
    """Load a WAV file and resample to target sample rate."""
    import soundfile as sf
    from scipy.signal import resample

    audio, sr = sf.read(filepath)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if necessary
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples)

    return audio.astype(np.float32)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class GTZANDataset(Dataset):
    """Simple dataset for loading GTZAN audio files."""

    def __init__(self, file_paths, labels, max_length=MAX_AUDIO_LENGTH):
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio = load_audio_file(self.file_paths[idx])

        # Take the middle segment of max_length
        if len(audio) > self.max_length:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            # Pad with zeros
            pad = np.zeros(self.max_length - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])

        return torch.tensor(audio), self.labels[idx]


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def collect_files():
    """Collect all audio files and labels from GTZAN directory."""
    files = []
    labels = []

    for genre_idx, genre in enumerate(GENRE_NAMES):
        genre_dir = DATA_DIR / genre
        if not genre_dir.exists():
            print(f"  Warning: {genre_dir} not found, skipping")
            continue

        genre_files = sorted(genre_dir.glob("*.wav"))
        files.extend(genre_files)
        labels.extend([genre_idx] * len(genre_files))
        print(f"  {genre}: {len(genre_files)} files")

    return files, labels


def train_test_split_data(files, labels, test_ratio=TEST_SPLIT, seed=SEED):
    """Split data into train and test sets, stratified by genre."""
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(len(files)))
    random.shuffle(indices)

    # Group by label for stratified split
    label_indices = {}
    for idx in indices:
        lbl = labels[idx]
        label_indices.setdefault(lbl, []).append(idx)

    train_idx, test_idx = [], []
    for lbl, idxs in label_indices.items():
        split_point = int(len(idxs) * (1 - test_ratio))
        train_idx.extend(idxs[:split_point])
        test_idx.extend(idxs[split_point:])

    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_files = [files[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_files, train_labels, test_files, test_labels


@torch.no_grad()
def extract_embeddings(model, feature_extractor, dataset, device, batch_size=BATCH_SIZE):
    """Extract MERT embeddings for all audio files."""
    model.eval()
    all_embeddings = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch_idx, (audio_batch, label_batch) in enumerate(loader):
        # Process through feature extractor
        inputs = feature_extractor(
            [a.numpy() for a in audio_batch],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through MERT
        outputs = model(**inputs, output_hidden_states=True)

        # Use the last hidden state, mean-pooled across time
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        last_hidden = outputs.hidden_states[-1]  # (batch, time, hidden_dim)
        embeddings = last_hidden.mean(dim=1)  # (batch, hidden_dim)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(label_batch)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) * batch_size >= len(dataset):
            processed = min((batch_idx + 1) * batch_size, len(dataset))
            print(f"    Extracted {processed}/{len(dataset)} embeddings...")

    return torch.cat(all_embeddings), torch.cat(all_labels)


class GenreClassifier(nn.Module):
    """Simple linear classifier on top of MERT embeddings."""

    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def train_classifier(train_embeds, train_labels, test_embeds, test_labels, input_dim, device):
    """Train a simple classifier on the embeddings."""
    classifier = GenreClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Move data to device
    train_x = train_embeds.to(device)
    train_y = train_labels.to(device)
    test_x = test_embeds.to(device)
    test_y = test_labels.to(device)

    best_acc = 0.0
    train_history = []

    for epoch in range(NUM_EPOCHS):
        # Training
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        train_preds = logits.argmax(dim=1)
        train_acc = (train_preds == train_y).float().mean().item()

        # Evaluation
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_x)
            test_loss = criterion(test_logits, test_y).item()
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == test_y).float().mean().item()

        if test_acc > best_acc:
            best_acc = test_acc

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss.item(), 4),
            "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {loss.item():.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    return classifier, best_acc, train_history


def compute_per_class_accuracy(classifier, test_embeds, test_labels, device):
    """Compute accuracy for each genre."""
    classifier.eval()
    test_x = test_embeds.to(device)

    with torch.no_grad():
        preds = classifier(test_x).argmax(dim=1).cpu()

    per_class = {}
    confusion = {}
    for genre_idx, genre in enumerate(GENRE_NAMES):
        mask = test_labels == genre_idx
        if mask.sum() == 0:
            continue
        genre_preds = preds[mask]
        correct = (genre_preds == genre_idx).sum().item()
        total = mask.sum().item()
        per_class[genre] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total > 0 else 0,
        }

        # Top misclassifications
        wrong_mask = genre_preds != genre_idx
        if wrong_mask.any():
            wrong_preds = genre_preds[wrong_mask]
            for wp in wrong_preds.tolist():
                key = f"{genre} → {GENRE_NAMES[wp]}"
                confusion[key] = confusion.get(key, 0) + 1

    return per_class, confusion


def main():
    print("=" * 60)
    print("MERT Genre Classification Prototype")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Model:  {MERT_MODEL}")
    print(f"Data:   {DATA_DIR}")

    # ── Step 1: Collect files ─────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 1: Collecting audio files...")
    files, labels = collect_files()
    print(f"Total: {len(files)} files across {len(set(labels))} genres")

    if len(files) == 0:
        print("ERROR: No audio files found. Run download_gtzan.py first.")
        sys.exit(1)

    # ── Step 2: Train/test split ──────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 2: Splitting data...")
    train_files, train_labels, test_files, test_labels = train_test_split_data(files, labels)
    print(f"  Train: {len(train_files)} files")
    print(f"  Test:  {len(test_files)} files")

    train_dataset = GTZANDataset(train_files, train_labels)
    test_dataset = GTZANDataset(test_files, test_labels)

    # ── Step 3: Load MERT ─────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 3: Loading MERT model...")
    start_time = time.time()

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MERT_MODEL, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    load_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # ── Step 4: Extract embeddings ────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 4: Extracting embeddings from audio...")
    print("  Processing training set...")
    start_time = time.time()
    train_embeds, train_labels_t = extract_embeddings(model, feature_extractor, train_dataset, device)
    train_time = time.time() - start_time

    print("  Processing test set...")
    start_time = time.time()
    test_embeds, test_labels_t = extract_embeddings(model, feature_extractor, test_dataset, device)
    test_time = time.time() - start_time

    embed_dim = train_embeds.shape[1]
    print(f"\n  Embedding dimension: {embed_dim}")
    print(f"  Train embeddings shape: {train_embeds.shape}")
    print(f"  Test embeddings shape:  {test_embeds.shape}")
    print(f"  Extraction time: {train_time + test_time:.1f}s "
          f"({(train_time + test_time) / len(files):.2f}s per file)")

    # Free MERT model from memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 5: Train classifier ──────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 5: Training genre classifier on embeddings...")
    classifier, best_acc, history = train_classifier(
        train_embeds, train_labels_t,
        test_embeds, test_labels_t,
        embed_dim, device
    )

    # ── Step 6: Per-class analysis ────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Step 6: Per-class accuracy breakdown:")
    per_class, confusion = compute_per_class_accuracy(
        classifier, test_embeds, test_labels_t, device
    )

    print(f"\n  {'Genre':<12} {'Correct':>7} {'Total':>5} {'Accuracy':>8}")
    print(f"  {'─' * 35}")
    for genre, stats in sorted(per_class.items()):
        bar = "█" * int(stats["accuracy"] * 20) + "░" * (20 - int(stats["accuracy"] * 20))
        print(f"  {genre:<12} {stats['correct']:>7} {stats['total']:>5} {stats['accuracy']:>8.1%}  {bar}")

    if confusion:
        print(f"\n  Top misclassifications:")
        for pair, count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
            print(f"    {pair}: {count} errors")

    # ── Step 7: Save results ──────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": MERT_MODEL,
        "device": str(device),
        "dataset": "GTZAN",
        "num_genres": len(GENRE_NAMES),
        "genre_names": GENRE_NAMES,
        "train_size": len(train_files),
        "test_size": len(test_files),
        "embedding_dim": embed_dim,
        "param_count": param_count,
        "best_test_accuracy": round(best_acc, 4),
        "per_class_accuracy": per_class,
        "top_confusions": dict(sorted(confusion.items(), key=lambda x: -x[1])[:10]),
        "training_history": history,
        "extraction_time_seconds": round(train_time + test_time, 2),
        "seconds_per_file": round((train_time + test_time) / len(files), 2),
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:         {MERT_MODEL}")
    print(f"  Best Test Acc: {best_acc:.1%}")
    print(f"  Embed Dim:     {embed_dim}")
    print(f"  Inference:     {results['seconds_per_file']:.2f}s per file")
    print(f"  Results saved: {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
