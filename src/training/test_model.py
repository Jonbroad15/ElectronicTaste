#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.models.contrastive_model import RaveNet
from src.training.loss_functions import MultiLabelSupConLoss
from src.training.dataset import ProcessedChunkDataset
from src.training.train import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to classifier_best.pt")
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    test_dir = Path(args.data_dir) / "test"
    splits_file = "splits.json"

    print("Loading test dataset...")
    test_dataset = ProcessedChunkDataset(test_dir, splits_file, split_name="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Initializing RaveNet...")
    model = RaveNet(mert_model_id=args.mert_model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    criterion = MultiLabelSupConLoss(temperature=0.07)
    
    print("Evaluating on Test Set...")
    test_loss = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            batch_size = waveforms.size(0)
            embeddings = model(waveforms)
            loss = criterion(embeddings, labels)
            test_loss += loss.item() * batch_size
            test_samples += batch_size
            
    test_loss /= max(1, test_samples)
    
    print("="*40)
    print(f"FINAL TEST LOSS (Contrastive): {test_loss:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
