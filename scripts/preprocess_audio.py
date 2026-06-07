#!/usr/bin/env python3
"""
Preprocesses massive raw DJ mix WAV files into isolated 30-second .flac chunks.

This script reads the splits.json manifest, loops over the assigned audio files,
and uses soundfile streaming to pull 30-second blocks without loading the entire
gigabyte-sized mix into memory.

The resulting chunks are physically isolated into train/val/test directories
to prevent data leakage during Dataloader initialization.
"""

import os
import json
import argparse
from pathlib import Path
import soundfile as sf
import librosa
from tqdm import tqdm
import multiprocessing
from functools import partial

CHUNK_LENGTH_SEC = 30.0
TARGET_SR = 24000

def process_file(file_id, split_dir, source_dir, target_sr=TARGET_SR):
    """
    Reads a single wav file, chops it into 30s chunks, and saves to the split dir.
    """
    source_path = source_dir / f"{file_id}.wav"
    if not source_path.exists():
        print(f"Warning: {source_path} not found. Skipping.")
        return

    # To process large files without OOM, we stream using soundfile
    info = sf.info(str(source_path))
    sr = info.samplerate
    
    # We will read block by block. To get exactly 30s at the target_sr,
    # We need to read enough from the original sr, then resample.
    # block_length must be an integer number of samples
    block_length = int(CHUNK_LENGTH_SEC * sr)
    
    try:
        chunk_idx = 0
        # Use sf.blocks for memory efficient and fast streaming
        for y_block in sf.blocks(str(source_path), blocksize=block_length, always_2d=True):
            # y_block is (frames, channels). Convert to mono for librosa
            y_mono = y_block.mean(axis=1)
            
            # Resample to target SR
            if sr != target_sr:
                y_resampled = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr)
            else:
                y_resampled = y_mono
                
            # y_resampled is 1D: (frames,)
            # Discard blocks that are shorter than 30s (the very end of the mix)
            if len(y_resampled) < int(target_sr * CHUNK_LENGTH_SEC):
                continue
                
            # Save chunk
            chunk_filename = f"{file_id}_chunk{chunk_idx:04d}.flac"
            chunk_path = split_dir / chunk_filename
            sf.write(str(chunk_path), y_resampled, target_sr, format='FLAC')
            
            chunk_idx += 1
            
    except Exception as e:
        print(f"Error processing {file_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess DJ mixes into 30s chunks")
    parser.add_argument("--splits", type=str, default="splits.json", help="Path to splits.json")
    parser.add_argument("--source-dir", type=str, default="data/mixes", help="Directory with raw .wav files")
    parser.add_argument("--target-dir", type=str, default="data/processed", help="Directory to save chunks")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of workers")
    args = parser.parse_args()

    splits_file = Path(args.splits)
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_file}")

    with open(splits_file, "r") as f:
        splits = json.load(f)

    # Make target directories
    for split_name in ["train", "val", "test"]:
        (target_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Gather all files by split
    tasks = []
    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            for item in splits[split_name]:
                # Extract 'mix0002' from 'mixes/mix0002.wav'
                audio_path = item.get("audio_path", "")
                if audio_path:
                    file_id = Path(audio_path).stem
                    tasks.append((file_id, target_dir / split_name))

    print(f"Found {len(tasks)} files to process across splits.")

    # We use multiprocessing to speed up processing across multiple files
    with multiprocessing.Pool(args.workers) as pool:
        # Use partial to pass the fixed arguments
        func = partial(process_file, source_dir=source_dir)
        # pool.starmap takes an iterable of argument tuples
        # Tqdm gives a nice progress bar
        list(tqdm(pool.starmap(func, tasks), total=len(tasks), desc="Processing mixes"))

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
