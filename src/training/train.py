#!/usr/bin/env python3
"""Fine-tune the SubgenreClassifier on frozen MERT embeddings.

The MERT encoder is loaded once, all embeddings extracted and kept in RAM,
then only the classification head is trained.  This is fast (no GPU needed
for the main loop once embeddings are cached) and avoids redundant encoder
forward passes.

Usage::

    python -m src.training.train \\
        --data-dir data/gtzan_audio \\
        --epochs 30 \\
        --batch-size 16

Outputs:
    models/classifier_best.pt      — best checkpoint by val accuracy
    models/label_encoder.json      — label ↔ index mapping
    models/checkpoints/epoch_*.pt  — per-epoch checkpoints
    training_log.csv               — epoch-by-epoch metrics
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from src.models.mert_encoder import MERTEncoder, get_device
from src.models.classifier import SubgenreClassifier
from src.training.dataset import AudioDataset, LabelEncoder


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    encoder: MERTEncoder,
    subset: Subset | AudioDataset,
    batch_size: int = 8,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``subset`` through the frozen MERT encoder; return (embeddings, labels).

    Returns:
        embeddings: Float tensor of shape ``(N, 768)`` on CPU.
        labels:     Long tensor of shape ``(N,)`` on CPU.
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = len(subset)  # type: ignore[arg-type]
    processed = 0
    all_emb: list[torch.Tensor] = []
    all_lbl: list[torch.Tensor] = []

    for waveforms, labels in loader:
        batch_embs = []
        for w in waveforms:
            emb = encoder.extract_embedding(w)  # (1, 768) on CPU
            batch_embs.append(emb)
        all_emb.append(torch.cat(batch_embs, dim=0))
        all_lbl.append(labels)
        processed += len(labels)
        suffix = f" [{label}]" if label else ""
        print(f"  Extracting embeddings{suffix}: {processed}/{total}", end="\r", flush=True)

    print()
    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: SubgenreClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """One full pass over the training embeddings.

    Returns:
        (mean_loss, accuracy) over all training samples.
    """
    model.train()
    perm = torch.randperm(len(embeddings))
    total_loss = 0.0
    correct = 0

    for start in range(0, len(embeddings), batch_size):
        idx = perm[start : start + batch_size]
        x = embeddings[idx].to(device)
        y = labels[idx].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()

    n = len(embeddings)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: SubgenreClassifier,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate the classifier on a fixed embedding set.

    Returns:
        (mean_loss, accuracy).
    """
    model.eval()
    x = embeddings.to(device)
    y = labels.to(device)
    logits = model(x)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SubgenreClassifier on frozen MERT embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True,
                        help="Root directory with per-class audio subdirectories")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data reserved for validation")
    parser.add_argument("--device", default=None,
                        help="Force device (mps / cuda / cpu)")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints",
                        help="Directory for per-epoch checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/embedding_cache",
                        help="Cache directory for preprocessed audio tensors")
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume training from")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice : {device}")
    print(f"Model  : {args.mert_model}")
    print(f"Data   : {args.data_dir}")

    # ── Label encoder ─────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    label_enc = LabelEncoder.from_directory(data_dir)
    print(f"Classes: {len(label_enc)} — {label_enc.names}\n")

    # ── Dataset split ──────────────────────────────────────────────
    dataset = AudioDataset(data_dir, label_enc, cache_dir=args.cache_dir)
    print(f"Total samples: {len(dataset)}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Train: {train_size}   Val: {val_size}\n")

    # ── MERT embedding extraction ──────────────────────────────────
    print(f"Loading MERT ({args.mert_model})…")
    encoder = MERTEncoder(model_id=args.mert_model, device=device)

    train_emb, train_lbl = extract_embeddings(encoder, train_ds, label="train")
    val_emb, val_lbl     = extract_embeddings(encoder, val_ds,   label="val")

    embed_dim = train_emb.shape[1]
    print(f"\nEmbedding dim: {embed_dim}")
    print(f"Train shape: {train_emb.shape}   Val shape: {val_emb.shape}\n")

    # Free MERT from accelerator memory
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Classifier setup ──────────────────────────────────────────
    start_epoch = 1
    classifier = SubgenreClassifier(num_classes=len(label_enc), embed_dim=embed_dim)

    if args.resume:
        classifier = SubgenreClassifier.load(args.resume, device="cpu")
        # Infer start epoch from filename if possible
        stem = Path(args.resume).stem
        if stem.startswith("epoch_"):
            try:
                start_epoch = int(stem.split("_")[1]) + 1
            except ValueError:
                pass
        print(f"Resumed from {args.resume} (starting at epoch {start_epoch})")

    classifier = classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 2
    )
    criterion = nn.CrossEntropyLoss()

    # ── Output directories ─────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    label_enc.save(models_dir / "label_encoder.json")
    print(f"Label encoder → models/label_encoder.json")

    # ── Training loop ──────────────────────────────────────────────
    log_path = Path("training_log.csv")
    best_val_acc = 0.0
    write_header = not log_path.exists() or start_epoch == 1

    with open(log_path, "a" if not write_header else "w", newline="") as log_fh:
        writer = csv.writer(log_fh)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

        header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}"
        print(header)
        print("─" * len(header))

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                classifier, optimizer, criterion,
                train_emb, train_lbl, device, args.batch_size,
            )
            va_loss, va_acc = evaluate(
                classifier, val_emb, val_lbl, device, criterion
            )
            scheduler.step()

            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                              f"{va_loss:.4f}", f"{va_acc:.4f}"])
            log_fh.flush()

            flag = " ← best" if va_acc > best_val_acc else ""
            elapsed = time.time() - t0
            print(
                f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
                f"{va_loss:>8.4f}  {va_acc:>7.4f}{flag}  ({elapsed:.1f}s)"
            )

            # Per-epoch checkpoint
            classifier.save(ckpt_dir / f"epoch_{epoch:03d}.pt")

            # Best model
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                classifier.save(models_dir / "classifier_best.pt")

    print(f"\nBest val accuracy : {best_val_acc:.1%}")
    print(f"Best model        → models/classifier_best.pt")
    print(f"Training log      → {log_path}")


if __name__ == "__main__":
    main()
