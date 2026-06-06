#!/usr/bin/env python3
"""Batch inference CLI — run the classifier over a directory of audio files.

Usage::

    python -m src.scripts.ingest_batch \\
        --audio-dir data/gtzan_audio/test \\
        --classifier models/classifier_best.pt \\
        --label-encoder models/label_encoder.json \\
        --output results/batch_results.csv

Output CSV columns:
    filename, label_1, confidence_1, label_2, confidence_2, label_3, confidence_3

If the audio directory follows the ``<root>/<class>/<file>`` layout, a
``ground_truth`` column is added and top-1 accuracy is printed.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

from src.audio.preprocess import load_and_preprocess
from src.models.mert_encoder import MERTEncoder, get_device
from src.models.classifier import SubgenreClassifier
from src.training.dataset import LabelEncoder

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def collect_files(audio_dir: Path) -> list[tuple[Path, str | None]]:
    """Return ``(path, ground_truth_label | None)`` pairs.

    If ``audio_dir`` has per-class subdirectories, the subdir name is used
    as the ground-truth label.  Otherwise ground truth is ``None``.
    """
    samples: list[tuple[Path, str | None]] = []

    subdirs = [d for d in sorted(audio_dir.iterdir()) if d.is_dir()]
    if subdirs:
        # Structured: root/class/file.wav
        for class_dir in subdirs:
            for f in sorted(class_dir.glob("*")):
                if f.suffix.lower() in _AUDIO_EXTENSIONS:
                    samples.append((f, class_dir.name))
    else:
        # Flat: root/file.wav
        for f in sorted(audio_dir.glob("*")):
            if f.suffix.lower() in _AUDIO_EXTENSIONS:
                samples.append((f, None))

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch inference and write results to CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-dir", required=True,
                        help="Directory of audio files to classify")
    parser.add_argument("--classifier", default="models/classifier_best.pt",
                        help="Path to classifier checkpoint")
    parser.add_argument("--label-encoder", default="models/label_encoder.json",
                        help="Path to label encoder JSON")
    parser.add_argument("--output", default="results/batch_results.csv",
                        help="Output CSV path")
    parser.add_argument("--device", default=None,
                        help="Force device (mps / cuda / cpu)")
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M")
    args = parser.parse_args()

    device = get_device(args.device)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────
    print(f"Loading MERT ({args.mert_model})…")
    encoder = MERTEncoder(model_id=args.mert_model, device=device)

    print(f"Loading classifier ({args.classifier})…")
    classifier = SubgenreClassifier.load(args.classifier, device=device)
    label_enc = LabelEncoder.load(args.label_encoder)
    print(f"Classes: {label_enc.names}\n")

    # ── Collect files ─────────────────────────────────────────────
    samples = collect_files(audio_dir)
    if not samples:
        print(f"ERROR: no audio files found in {audio_dir}")
        sys.exit(1)

    print(f"Found {len(samples)} audio files in {audio_dir}\n")
    has_ground_truth = samples[0][1] is not None

    # ── Run inference ─────────────────────────────────────────────
    rows: list[dict] = []
    correct = 0
    t_start = time.time()

    for i, (path, truth) in enumerate(samples, 1):
        t0 = time.time()
        waveform = load_and_preprocess(str(path))
        embedding = encoder.extract_embedding(waveform).to(device)
        predictions = classifier.predict_top3(embedding, label_enc.names)
        elapsed = time.time() - t0

        row: dict = {"filename": path.name}
        if has_ground_truth:
            row["ground_truth"] = truth or ""

        for rank, pred in enumerate(predictions, 1):
            row[f"label_{rank}"] = pred["label"]
            row[f"confidence_{rank}"] = pred["confidence"]

        rows.append(row)

        top1_correct = truth == predictions[0]["label"] if truth else None
        if top1_correct:
            correct += 1

        status = ""
        if truth is not None:
            status = " ✓" if top1_correct else f" ✗ (truth={truth})"
        print(
            f"[{i:>4}/{len(samples)}] {path.name:<40} "
            f"{predictions[0]['label']:<15} ({predictions[0]['confidence']:.3f}){status}  "
            f"{elapsed:.2f}s"
        )

    total_time = time.time() - t_start

    # ── Write CSV ─────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"Files processed : {len(samples)}")
    print(f"Total time      : {total_time:.1f}s")
    print(f"Per-file time   : {total_time / max(1, len(samples)):.2f}s")
    if has_ground_truth:
        acc = correct / len(samples)
        print(f"Top-1 accuracy  : {acc:.1%}  ({correct}/{len(samples)})")
    print(f"Results written → {output_path}")


if __name__ == "__main__":
    main()
