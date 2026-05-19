#!/usr/bin/env python3
"""
Download the GTZAN dataset from HuggingFace and save audio files locally.

This gives us ~999 WAV files across 10 genres (30s each) to validate
our MERT/CLAP audio classification pipeline end-to-end.

Usage: python scripts/download_gtzan.py
"""

import os
import sys
from pathlib import Path

def main():
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install datasets soundfile")
        sys.exit(1)

    output_dir = Path("data/gtzan_audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading GTZAN dataset from HuggingFace...")
    print("This will download ~1.2GB of audio data.\n")

    # Use the parquet-converted version (the original marsyas/gtzan uses
    # a deprecated loading script format)
    dataset = load_dataset("sanchit-gandhi/gtzan", split="train")

    print(f"Total samples: {len(dataset)}")

    # Get unique genres
    genres = sorted(set(dataset["genre"]))
    print(f"Genres ({len(genres)}): {genres}\n")

    # Create genre subdirectories and save audio files
    genre_counts = {}
    for i, sample in enumerate(dataset):
        genre_label = sample["genre"]
        audio = sample["audio"]

        # genre may be an int index — map it
        if isinstance(genre_label, int):
            genre_names = [
                "blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"
            ]
            genre = genre_names[genre_label]
        else:
            genre = genre_label

        genre_dir = output_dir / genre
        genre_dir.mkdir(exist_ok=True)

        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        filename = f"{genre}.{genre_counts[genre]:05d}.wav"
        filepath = genre_dir / filename

        # Save audio as WAV
        sf.write(str(filepath), audio["array"], audio["sampling_rate"])

        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(dataset)} files...")

    print(f"\nDone! Saved {len(dataset)} audio files to {output_dir}/")
    print("\nBreakdown by genre:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre}: {count} files")

if __name__ == "__main__":
    main()
