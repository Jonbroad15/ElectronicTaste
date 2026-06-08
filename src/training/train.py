#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.contrastive_model import RaveNet
from src.training.loss_functions import MultiLabelSupConLoss
from src.training.dataset import ProcessedChunkDataset

def get_device(force_device=None):
    if force_device:
        return torch.device(force_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="models/classifier")
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", default="splits.json", help="Path to splits.json")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    # Ignored legacy args to avoid breaking gcp script
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--temporal-pooling", type=str, default="")
    parser.add_argument("--loss", type=str, default="")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    
    splits_file = args.splits

    print("Loading datasets...")
    train_dataset = ProcessedChunkDataset(train_dir, splits_file, split_name="train")
    val_dataset = ProcessedChunkDataset(val_dir, splits_file, split_name="val", label_list=train_dataset.label_list)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Classes ({len(train_dataset.label_list)}): {train_dataset.label_list}")

    print("Initializing RaveNet...")
    model = RaveNet(mert_model_id=args.mert_model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = MultiLabelSupConLoss(temperature=0.07)
    
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        t0 = time.time()
        
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            batch_size = waveforms.size(0)
            
            optimizer.zero_grad()
            embeddings = model(waveforms)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
        train_loss /= max(1, train_samples)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                batch_size = waveforms.size(0)
                embeddings = model(waveforms)
                loss = criterion(embeddings, labels)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        
        val_loss /= max(1, val_samples)
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = ckpt_dir / "classifier_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best checkpoint to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break

    print("Training finished.")

if __name__ == "__main__":
    main()
