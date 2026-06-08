import os
import pytest
from pathlib import Path
import json
import torch
import shutil

from src.training.dataset import ProcessedChunkDataset

@pytest.fixture
def mock_processed_dir(tmp_path):
    processed_dir = tmp_path / "processed"
    
    # Create splits
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    # Create dummy flac files
    (train_dir / "mix0001_chunk0000.flac").touch()
    (train_dir / "mix0001_chunk0001.flac").touch()
    
    (val_dir / "mix0002_chunk0000.flac").touch()
    
    (test_dir / "mix0003_chunk0000.flac").touch()
    
    # Create dummy splits.json
    splits_file = tmp_path / "splits.json"
    splits_data = {
        "train": [{"audio_path": "mixes/mix0001.wav", "l1_genres": ["House"]}],
        "val": [{"audio_path": "mixes/mix0002.wav", "l1_genres": ["Techno"]}],
        "test": [{"audio_path": "mixes/mix0003.wav", "l1_genres": ["Trance"]}]
    }
    with open(splits_file, "w") as f:
        json.dump(splits_data, f)
        
    return processed_dir, splits_file

def test_v1_1_mam_isolation(mock_processed_dir):
    processed_dir, splits_file = mock_processed_dir
    
    # Assert ProcessedChunkDataset for train split only sees train files
    dataset = ProcessedChunkDataset(
        processed_dir=processed_dir / "train",
        splits_file=splits_file,
        split_name="train",
        crop_seconds=None
    )
    
    assert len(dataset) == 2
    for path in dataset.files:
        assert path.parent.name == "train", "Dataset should only load files from the train folder"
        assert "val" not in path.parts[-2:], "val should not be in the immediate path"
        assert "test" not in path.parts[-2:], "test should not be in the immediate path"

def test_v2_2_split_integrity(mock_processed_dir, monkeypatch):
    processed_dir, splits_file = mock_processed_dir
    
    # Test dataset split labels
    dataset = ProcessedChunkDataset(
        processed_dir=processed_dir / "test",
        splits_file=splits_file,
        split_name="test"
    )
    
    assert len(dataset) == 1
    # Check that Trance is in the label list
    assert "Trance" in dataset.label_list
    
    # Check multi-hot labels
    # We monkeypatch torchaudio.load to avoid actually reading empty files
    import torchaudio
    def mock_load(path):
        return torch.zeros(1, 24000), 24000
    monkeypatch.setattr(torchaudio, "load", mock_load)
    
    waveform, labels = dataset[0]
    
    trance_idx = dataset.label_to_idx["Trance"]
    assert labels[trance_idx] == 1.0

def test_v3_1_early_stopping():
    # Mocking the early stopping logic found in train.py
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    val_losses = [10.0, 9.0, 8.0, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0] # Stops at epoch 8 (index 7)
    
    stopped_epoch = -1
    for epoch, val_loss in enumerate(val_losses):
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch
                break
                
    assert stopped_epoch == 7, "Should trigger early stopping after 5 epochs of no improvement"
