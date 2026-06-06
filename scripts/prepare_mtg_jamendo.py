#!/usr/bin/env python3
"""Verify and summarise the MTG-Jamendo dataset after audio download.

This script reads the manifest produced by download_mtg_jamendo.py, verifies
that audio files exist on disk, reports per-class statistics, and warns about
classes that fall below the 50-sample training minimum.

Usage::

    python scripts/prepare_mtg_jamendo.py

    # Custom paths:
    python scripts/prepare_mtg_jamendo.py \\
        --manifest data/mtg_jamendo_manifest.csv \\
        --audio-dir data/mtg_jamendo \\
        --taxonomy data/taxonomy.json

Outputs (printed to stdout):
    - Class distribution table (total, present on disk, missing)
    - Warnings for classes with < 50 verified samples
    - Overall training-readiness summary
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_MANIFEST = Path("data/mtg_jamendo_manifest.csv")
DEFAULT_AUDIO_DIR = Path("data/mtg_jamendo")
DEFAULT_TAXONOMY = Path("data/taxonomy.json")

MIN_SAMPLES_PER_CLASS = 50  # Minimum tracks required for training a class


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> list[dict]:
    """Read the manifest CSV and return a list of row dicts."""
    if not manifest_path.exists():
        print(
            f"ERROR: Manifest not found: {manifest_path}\n"
            "Run scripts/download_mtg_jamendo.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print(f"ERROR: Manifest {manifest_path} is empty.", file=sys.stderr)
        sys.exit(1)

    return rows


def check_audio_files(
    rows: list[dict],
    audio_dir: Path,
) -> tuple[Counter, Counter, Counter]:
    """Check which manifest tracks have audio files on disk.

    Returns:
        Tuple of (total_per_class, present_per_class, missing_per_class)
        as Counter objects keyed by canonical label.
    """
    total: Counter = Counter()
    present: Counter = Counter()
    missing: Counter = Counter()

    for row in rows:
        label = row["canonical_label"]
        track_id = row["track_id"]
        total[label] += 1

        # Audio may be .mp3 or .wav (download script writes .mp3)
        audio_path_mp3 = audio_dir / label / f"{track_id}.mp3"
        audio_path_wav = audio_dir / label / f"{track_id}.wav"
        if audio_path_mp3.exists() or audio_path_wav.exists():
            present[label] += 1
        else:
            missing[label] += 1

    return total, present, missing


def load_classes(taxonomy_path: Path) -> list[str]:
    """Load the ordered list of canonical class names from the taxonomy."""
    if not taxonomy_path.exists():
        return []
    with open(taxonomy_path) as fh:
        taxonomy = json.load(fh)
    return taxonomy.get("classes", [])


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_distribution(
    classes: list[str],
    total: Counter,
    present: Counter,
    missing: Counter,
) -> list[str]:
    """Print the class distribution table and return a list of warning messages.

    Returns:
        List of warning strings for classes that need attention.
    """
    warnings: list[str] = []

    col_w = 22
    print(f"\n{'─' * 65}")
    print(
        f"  {'Class':<{col_w}}  {'In manifest':>12}  {'On disk':>8}  {'Missing':>8}"
    )
    print(f"{'─' * 65}")

    all_labels = list(classes) or sorted(set(total.keys()))
    total_manifest = 0
    total_present = 0
    total_missing = 0

    for cls in all_labels:
        n_total = total[cls]
        n_present = present[cls]
        n_missing = missing[cls]
        total_manifest += n_total
        total_present += n_present
        total_missing += n_missing

        row_warnings = []
        if n_total == 0:
            row_warnings.append("NO MANIFEST ENTRIES")
        elif n_present < MIN_SAMPLES_PER_CLASS:
            row_warnings.append(f"< {MIN_SAMPLES_PER_CLASS} verified samples")

        flag = f"  [WARN: {row_warnings[0]}]" if row_warnings else ""
        print(
            f"  {cls:<{col_w}}  {n_total:>12,}  {n_present:>8,}  {n_missing:>8,}{flag}"
        )

        if row_warnings:
            warnings.append(f"{cls}: {row_warnings[0]}")

    print(f"{'─' * 65}")
    print(
        f"  {'TOTAL':<{col_w}}  {total_manifest:>12,}  {total_present:>8,}  {total_missing:>8,}"
    )
    print(f"{'─' * 65}\n")

    return warnings


def print_summary(
    warnings: list[str],
    present: Counter,
    classes: list[str],
    audio_dir: Path,
) -> None:
    """Print training readiness summary."""
    ready_classes = [c for c in classes if present[c] >= MIN_SAMPLES_PER_CLASS]
    not_ready = [c for c in classes if present[c] < MIN_SAMPLES_PER_CLASS]

    print("Training readiness summary:")
    print(f"  Classes ready (>= {MIN_SAMPLES_PER_CLASS} samples): {len(ready_classes)}/{len(classes)}")
    if ready_classes:
        print(f"    {', '.join(ready_classes)}")

    if not_ready:
        print(f"\n  Classes NOT ready (< {MIN_SAMPLES_PER_CLASS} samples):")
        for cls in not_ready:
            print(f"    {cls}: {present[cls]} verified samples")

    if warnings:
        print(f"\n  {len(warnings)} class(es) need attention before training.")
        print("  Run `python scripts/download_mtg_jamendo.py --download-audio`")
        print("  to download missing audio files.")
    else:
        print(f"\n  All {len(classes)} classes are ready for training.")

    print(f"\nNext step:")
    print(
        f"  python -m src.training.train "
        f"--data-dir {audio_dir} "
        f"--epochs 50 "
        f"--lr 1e-3 "
        f"--batch-size 32"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MTG-Jamendo audio files and report training readiness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the manifest CSV written by download_mtg_jamendo.py",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(DEFAULT_AUDIO_DIR),
        help="Root directory containing per-class audio subdirectories",
    )
    parser.add_argument(
        "--taxonomy",
        default=str(DEFAULT_TAXONOMY),
        help="Path to data/taxonomy.json for canonical class ordering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    taxonomy_path = Path(args.taxonomy)

    print(f"Manifest  : {manifest_path}")
    print(f"Audio dir : {audio_dir}")
    print(f"Taxonomy  : {taxonomy_path}")

    rows = load_manifest(manifest_path)
    print(f"\nLoaded {len(rows):,} entries from manifest.")

    classes = load_classes(taxonomy_path)
    if not classes:
        print(
            "[WARN] Could not load taxonomy classes; using labels from manifest.",
            file=sys.stderr,
        )
        classes = sorted({row["canonical_label"] for row in rows})

    print(f"Checking audio files in {audio_dir} …")
    total, present, missing = check_audio_files(rows, audio_dir)

    warnings = print_distribution(classes, total, present, missing)
    print_summary(warnings, present, classes, audio_dir)


if __name__ == "__main__":
    main()
