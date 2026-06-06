#!/usr/bin/env python3
"""Fine-tune the SubgenreClassifier on frozen MERT embeddings.

The MERT encoder is loaded once, all embeddings extracted using batched GPU
inference, then only the classification head is trained.

Split strategy: 70 % train / 10 % val / 20 % test.
The test split is saved to ``models/split_manifest.json`` so the benchmark
script can evaluate on held-out files only — preventing data leakage.

Usage::

    python -m src.training.train \\
        --data-dir data/gtzan_audio \\
        --epochs 30 \\
        --batch-size 16

    # Resume from epoch 10:
    python -m src.training.train \\
        --data-dir data/gtzan_audio \\
        --resume models/checkpoints/epoch_010.pt

Outputs:
    models/classifier_best.pt      — best inference checkpoint (weights only)
    models/label_encoder.json      — label ↔ index mapping
    models/split_manifest.json     — train/val/test file paths (for benchmark)
    models/checkpoints/epoch_*.pt  — full training checkpoints (resumable)
    training_log.csv               — epoch-by-epoch metrics
"""
from __future__ import annotations

import argparse
import csv
import json
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


# ── Training checkpoint helpers ───────────────────────────────────────────────

def save_training_checkpoint(
    path: Path,
    classifier: SubgenreClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_acc: float,
) -> None:
    """Save full training state to a checkpoint file."""
    torch.save(
        {
            "model_state_dict":     classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_classes":          classifier.num_classes,
            "embed_dim":            classifier.embed_dim,
            "epoch":                epoch,
            "best_val_acc":         best_val_acc,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    device: torch.device,
    lr: float,
    epochs: int,
) -> tuple[SubgenreClassifier, torch.optim.Optimizer,
           torch.optim.lr_scheduler.LRScheduler, int, float]:
    """Restore classifier, optimizer, scheduler, start epoch, and best val acc.

    Returns:
        (classifier, optimizer, scheduler, start_epoch, best_val_acc)
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    classifier = SubgenreClassifier(
        num_classes=ckpt["num_classes"],
        embed_dim=ckpt["embed_dim"],
    )
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr, weight_decay=1e-4
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return classifier, optimizer, scheduler, ckpt["epoch"] + 1, ckpt["best_val_acc"]


# ── Embedding extraction (batched) ────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    encoder: MERTEncoder,
    subset: Subset | AudioDataset,
    batch_size: int = 8,
    label: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``subset`` through the frozen MERT encoder using batched inference.

    Returns:
        embeddings: Float tensor ``(N, 768)`` on CPU.
        labels:     Long tensor ``(N,)`` on CPU.
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = len(subset)  # type: ignore[arg-type]
    processed = 0
    all_emb: list[torch.Tensor] = []
    all_lbl: list[torch.Tensor] = []

    for waveforms, labels in loader:
        # Batched GPU forward pass — much faster than one-at-a-time
        embs = encoder.extract_embeddings_batch(list(waveforms))  # (B, 768)
        all_emb.append(embs)
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
    parser.add_argument("--val-split", type=float, default=0.10,
                        help="Fraction of data for validation (default 10%%)")
    parser.add_argument("--test-split", type=float, default=0.20,
                        help="Fraction of data held out for testing (default 20%%)")
    parser.add_argument("--device", default=None,
                        help="Force device (mps / cuda / cpu)")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints",
                        help="Directory for per-epoch training checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/embedding_cache",
                        help="Cache directory for preprocessed audio tensors")
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--resume", default=None,
                        help="Path to a training checkpoint to resume from")
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

    # ── Three-way split: train / val / test ───────────────────────
    dataset = AudioDataset(data_dir, label_enc, cache_dir=args.cache_dir)
    n = len(dataset)
    print(f"Total samples: {n}")

    test_size  = max(1, int(n * args.test_split))
    val_size   = max(1, int(n * args.val_split))
    train_size = n - val_size - test_size

    if train_size <= 0:
        raise ValueError(
            f"Not enough samples ({n}) for val ({val_size}) + test ({test_size}) splits."
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    print(f"Train: {train_size}   Val: {val_size}   Test: {test_size}\n")

    # ── Persist split manifest (for data-leakage-free benchmarking) ──
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    def _paths(subset: Subset) -> list[str]:
        return [str(dataset.samples[i][0]) for i in subset.indices]

    split_manifest = {
        "seed":      args.seed,
        "data_dir":  str(data_dir),
        "train":     _paths(train_ds),
        "val":       _paths(val_ds),
        "test":      _paths(test_ds),
    }
    manifest_path = models_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2))
    print(f"Split manifest   → {manifest_path}")

    # ── MERT embedding extraction (batched) ───────────────────────
    print(f"Loading MERT ({args.mert_model})…")
    encoder = MERTEncoder(model_id=args.mert_model, device=device)

    train_emb, train_lbl = extract_embeddings(encoder, train_ds, args.batch_size, "train")
    val_emb,   val_lbl   = extract_embeddings(encoder, val_ds,   args.batch_size, "val")

    embed_dim = train_emb.shape[1]
    print(f"\nEmbedding dim: {embed_dim}")
    print(f"Train: {train_emb.shape}   Val: {val_emb.shape}\n")

    # Free MERT from accelerator memory
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Classifier + optimizer + scheduler ───────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print(f"Resuming from {args.resume}…")
        classifier, optimizer, scheduler, start_epoch, best_val_acc = (
            load_training_checkpoint(Path(args.resume), device, args.lr, args.epochs)
        )
        print(f"  Restored to epoch {start_epoch - 1}, best val acc {best_val_acc:.1%}\n")
    else:
        start_epoch  = 1
        best_val_acc = 0.0
        classifier   = SubgenreClassifier(num_classes=len(label_enc), embed_dim=embed_dim)
        classifier.to(device)
        optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=args.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    criterion = nn.CrossEntropyLoss()

    # ── Save label encoder ────────────────────────────────────────
    label_enc.save(models_dir / "label_encoder.json")
    print(f"Label encoder    → models/label_encoder.json")

    # ── Training loop ─────────────────────────────────────────────
    log_path = Path("training_log.csv")
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

            # Full training checkpoint (resumable)
            save_training_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                classifier, optimizer, scheduler, epoch, best_val_acc,
            )

            # Best inference checkpoint (weights only, used by Predictor)
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                classifier.save(models_dir / "classifier_best.pt")

    print(f"\nBest val accuracy : {best_val_acc:.1%}")
    print(f"Best model        → models/classifier_best.pt")
    print(f"Training log      → {log_path}")


if __name__ == "__main__":
    main()
