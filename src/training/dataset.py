import json
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import random

class ProcessedChunkDataset(Dataset):
    """
    Dataset that loads preprocessed 30s .flac chunks and maps them to multi-hot genre labels
    based on splits.json.
    """
    def __init__(
        self, 
        processed_dir, 
        splits_file, 
        split_name="train", 
        crop_seconds=None, 
        target_sr=24000, 
        label_list=None
    ):
        self.processed_dir = Path(processed_dir)
        self.target_sr = target_sr
        self.crop_seconds = crop_seconds
        
        with open(splits_file, "r") as f:
            splits = json.load(f)
            
        self.files = sorted(list(self.processed_dir.glob("*.flac")))
            
        # Build mapping from file_id -> multi-hot labels
        self.label_list = label_list
        if not self.label_list:
            # build label list from splits
            genres = set()
            for s in ["train", "val", "test"]:
                if s in splits:
                    for item in splits[s]:
                        genres.update(item.get("l1_genres", []))
                        genres.update(item.get("l2_genres", []))
                        genres.update(item.get("l3_genres", []))
            self.label_list = sorted(list(genres))
            
        self.label_to_idx = {name: i for i, name in enumerate(self.label_list)}
        
        # Precompute file -> label tensors
        self.file_to_labels = {}
        for item in splits.get(split_name, []):
            audio_path = item.get("audio_path", "")
            if not audio_path:
                continue
            file_id = Path(audio_path).stem
            
            active_genres = []
            active_genres.extend(item.get("l1_genres", []))
            active_genres.extend(item.get("l2_genres", []))
            active_genres.extend(item.get("l3_genres", []))
            
            multi_hot = torch.zeros(len(self.label_list))
            for g in active_genres:
                if g in self.label_to_idx:
                    multi_hot[self.label_to_idx[g]] = 1.0
            self.file_to_labels[file_id] = multi_hot

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # path name is like mix0002_chunk0000.flac
        file_id = path.name.split("_chunk")[0]
        
        labels = self.file_to_labels.get(file_id, torch.zeros(len(self.label_list)))
        waveform, sr = torchaudio.load(path)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if self.crop_seconds:
            frames = int(self.crop_seconds * sr)
            if waveform.shape[1] > frames:
                start = random.randint(0, waveform.shape[1] - frames)
                waveform = waveform[:, start:start+frames]
            else:
                pad = frames - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
                
        return waveform.squeeze(0), labels
