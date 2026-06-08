#!/usr/bin/env python3
"""
create_splits.py
----------------
Reads labels.json (produced by annotate_raveform.py) and produces splits.json
with deterministic train/val/test assignments that guarantee every genre node
has at least one example in val and test.

For genres with only a single mix, the WAV is split at the 50% point:
  - original      -> train bucket
  - <mix_id>_A.wav -> val bucket
  - <mix_id>_B.wav -> test bucket

Usage
-----
python3 scripts/create_splits.py \
    --labels /mnt/data/djmix/labels.json \
    --mixes-dir /mnt/data/djmix/mixes \
    --splits-dir /mnt/data/djmix/splits \
    --output /mnt/data/djmix/splits.json \
    --seed 42 \
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION_IN = 2
SCHEMA_VERSION_OUT = 1

LEVELS = ("l1_genres", "l2_genres", "l3_genres", "l4_genres")


# ---------------------------------------------------------------------------
# FFprobe / FFmpeg helpers
# ---------------------------------------------------------------------------


def get_duration(path: Path) -> float:
    """Return the duration of an audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    log.debug("ffprobe: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_wav(
    src: Path,
    out_a: Path,
    out_b: Path,
    dry_run: bool = False,
) -> None:
    """
    Losslessly split *src* at its midpoint into *out_a* (first half) and
    *out_b* (second half) using stream-copy so no re-encoding occurs.

    If both output files already exist the split is skipped entirely.
    """
    a_exists = out_a.exists()
    b_exists = out_b.exists()

    if a_exists and b_exists:
        log.info("Split already exists, skipping: %s / %s", out_a.name, out_b.name)
        return

    duration = get_duration(src)
    half = duration / 2.0
    log.info(
        "Splitting %s (%.1f s) -> %s | %s",
        src.name,
        duration,
        out_a.name,
        out_b.name,
    )

    if dry_run:
        log.info("[dry-run] Would split %s at %.3f s", src, half)
        return

    out_a.parent.mkdir(parents=True, exist_ok=True)

    if not a_exists:
        cmd_a = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-t", str(half),
            "-c", "copy",
            str(out_a),
        ]
        log.debug("ffmpeg A: %s", " ".join(cmd_a))
        subprocess.run(cmd_a, check=True, capture_output=True)

    if not b_exists:
        cmd_b = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ss", str(half),
            "-c", "copy",
            str(out_b),
        ]
        log.debug("ffmpeg B: %s", " ".join(cmd_b))
        subprocess.run(cmd_b, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------


def build_genre_buckets(
    files: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """
    For every genre tag at every depth level, return the list of mix_ids
    (filenames as they appear in labels.json) that carry that genre.

    Key format: "<level>/<genre_name>", e.g. "l2_genres/Dub Techno".
    """
    buckets: dict[str, list[str]] = {}
    for mix_id, info in files.items():
        for level in LEVELS:
            for genre in info.get(level, []):
                key = f"{level}/{genre}"
                buckets.setdefault(key, [])
                buckets[key].append(mix_id)
    return buckets


def get_file_duration_cached(
    mix_id: str,
    files: dict[str, dict[str, Any]],
    mixes_dir: Path,
    cache: dict[str, float],
) -> float:
    """Return cached duration for a mix, fetching from disk if needed."""
    if mix_id not in cache:
        audio_path = files[mix_id]["audio_path"]
        full_path = mixes_dir / Path(audio_path).name
        try:
            cache[mix_id] = get_duration(full_path)
        except (subprocess.CalledProcessError, ValueError) as exc:
            log.warning("Could not get duration for %s: %s", mix_id, exc)
            cache[mix_id] = 0.0
    return cache[mix_id]


def assign_splits(
    files: dict[str, dict[str, Any]],
    mixes_dir: Path,
    splits_dir: Path,
    seed: int,
    dry_run: bool,
) -> tuple[
    set[str],           # train set  (mix_ids + pseudo split-half ids)
    set[str],           # val set
    set[str],           # test set
    list[str],          # uncovered genres
    list[str],          # mix_ids that got WAV-split
    dict[str, dict],    # extra record info keyed by pseudo split-half id
]:
    """
    Core algorithm.  Returns sets of mix_ids plus auxiliary metadata.

    Splitting rules:
        0 files  -> log as uncovered_genre
        1 file   -> WAV split at 50%: original->train, _A->val, _B->test
        2 files  -> file[0]->val, file[1]->test
        >=3 files -> sort by duration; median->val, median-1->test, rest->train

    Conflict resolution (after all buckets): test > val > train
    """
    rng = random.Random(seed)

    buckets = build_genre_buckets(files)
    log.info("Found %d genre-level buckets across %d files.", len(buckets), len(files))

    # Per-file voting sets (real mix_ids and pseudo split-half ids)
    train_votes: set[str] = set()
    val_votes:   set[str] = set()
    test_votes:  set[str] = set()

    uncovered_genres: list[str] = []
    wav_split_ids: list[str] = []
    split_half_entries: dict[str, dict] = {}

    duration_cache: dict[str, float] = {}

    for genre_key, mix_ids in sorted(buckets.items()):
        n = len(mix_ids)
        level_name, genre_name = genre_key.split("/", 1)

        # ---- 0 files --------------------------------------------------------
        if n == 0:
            log.warning("Genre '%s' (%s) has 0 files — uncovered.", genre_name, level_name)
            uncovered_genres.append(genre_name)
            continue

        # ---- 1 file: WAV split ----------------------------------------------
        if n == 1:
            mix_id = mix_ids[0]
            log.info(
                "Genre '%s' (%s): 1 file — WAV split on %s",
                genre_name, level_name, mix_id,
            )
            # Original stays in train (unless overridden by conflict resolution)
            train_votes.add(mix_id)

            stem = Path(mix_id).stem          # e.g. "mix_0000099"
            src_path = mixes_dir / mix_id
            out_a = splits_dir / f"{stem}_A.wav"
            out_b = splits_dir / f"{stem}_B.wav"

            if src_path.exists() or dry_run:
                try:
                    split_wav(src_path, out_a, out_b, dry_run=dry_run)
                except subprocess.CalledProcessError as exc:
                    log.error("ffmpeg failed for %s: %s", mix_id, exc)
                    # Best-effort fallback: original covers both val and test
                    val_votes.add(mix_id)
                    test_votes.add(mix_id)
                    continue
            else:
                log.warning("Source WAV not found: %s — skipping split.", src_path)

            pseudo_a = f"{stem}_A.wav"
            pseudo_b = f"{stem}_B.wav"

            val_votes.add(pseudo_a)
            test_votes.add(pseudo_b)

            orig_info = files[mix_id]
            for pseudo_id, rel_path in [
                (pseudo_a, f"splits/{stem}_A.wav"),
                (pseudo_b, f"splits/{stem}_B.wav"),
            ]:
                split_half_entries[pseudo_id] = {
                    "audio_path":    rel_path,
                    "l1_genres":     orig_info.get("l1_genres", []),
                    "l2_genres":     orig_info.get("l2_genres", []),
                    "l3_genres":     orig_info.get("l3_genres", []),
                    "l4_genres":     orig_info.get("l4_genres", []),
                    "is_split_half": True,
                    "original_file": orig_info["audio_path"],
                }

            if mix_id not in wav_split_ids:
                wav_split_ids.append(mix_id)
            continue

        # ---- 2 files --------------------------------------------------------
        if n == 2:
            log.info(
                "Genre '%s' (%s): 2 files — file[0]->val, file[1]->test",
                genre_name, level_name,
            )
            val_votes.add(mix_ids[0])
            test_votes.add(mix_ids[1])
            continue

        # ---- >= 3 files -----------------------------------------------------
        # Shuffle deterministically first, then sort by duration so the
        # tie-breaking within equal durations is reproducible.
        shuffled = list(mix_ids)
        rng.shuffle(shuffled)

        shuffled.sort(
            key=lambda m: get_file_duration_cached(m, files, mixes_dir, duration_cache)
        )

        median_idx = len(shuffled) // 2
        val_pick   = shuffled[median_idx]
        test_pick  = shuffled[median_idx - 1]
        rest       = [
            m for i, m in enumerate(shuffled)
            if i not in (median_idx, median_idx - 1)
        ]

        log.info(
            "Genre '%s' (%s): %d files — val=%s, test=%s, train=%d others",
            genre_name, level_name, n, val_pick, test_pick, len(rest),
        )

        val_votes.add(val_pick)
        test_votes.add(test_pick)
        for m in rest:
            train_votes.add(m)

    # -------------------------------------------------------------------------
    # Conflict resolution: test > val > train
    # -------------------------------------------------------------------------
    final_test  = set(test_votes)
    final_val   = val_votes  - final_test
    final_train = train_votes - final_test - final_val

    return final_train, final_val, final_test, uncovered_genres, wav_split_ids, split_half_entries


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def build_record(
    mix_id: str,
    files: dict[str, dict[str, Any]],
    split_half_entries: dict[str, dict],
) -> dict[str, Any]:
    """Build a single output record for a mix_id or a pseudo split-half id."""
    if mix_id in split_half_entries:
        return dict(split_half_entries[mix_id])

    info = files[mix_id]
    return {
        "audio_path": info["audio_path"],
        "l1_genres":  info.get("l1_genres", []),
        "l2_genres":  info.get("l2_genres", []),
        "l3_genres":  info.get("l3_genres", []),
        "l4_genres":  info.get("l4_genres", []),
    }


def assemble_output(
    train: set[str],
    val:   set[str],
    test:  set[str],
    files: dict[str, dict[str, Any]],
    split_half_entries: dict[str, dict],
    uncovered_genres: list[str],
    wav_split_count: int,
    seed: int,
) -> dict[str, Any]:
    """Assemble the final splits.json structure (schema_version 1)."""

    def build_list(id_set: set[str]) -> list[dict[str, Any]]:
        records = []
        for mid in sorted(id_set):
            if mid not in files and mid not in split_half_entries:
                log.warning("Skipping unknown id in set: %s", mid)
                continue
            records.append(build_record(mid, files, split_half_entries))
        return records

    train_list = build_list(train)
    val_list   = build_list(val)
    test_list  = build_list(test)

    return {
        "schema_version": SCHEMA_VERSION_OUT,
        "seed":           seed,
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_train":      len(train_list),
            "total_val":        len(val_list),
            "total_test":       len(test_list),
            "genres_wav_split": wav_split_count,
            "uncovered_genres": sorted(set(uncovered_genres)),
        },
        "train": train_list,
        "val":   val_list,
        "test":  test_list,
    }


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------


def write_json_atomic(data: dict[str, Any], dest: Path, dry_run: bool) -> None:
    """Write *data* as pretty-printed JSON to *dest* atomically via a .tmp file."""
    if dry_run:
        log.info("[dry-run] Would write %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest)
        log.info("Wrote %s", dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce deterministic train/val/test splits from labels.json, "
            "guaranteeing every genre node has >=1 example in val and test."
        )
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to labels.json (schema version 2).",
    )
    parser.add_argument(
        "--mixes-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing the raw mix WAV files.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory where WAV half-clips are written. "
            "Defaults to <mixes-dir>/../splits."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Destination path for splits.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for deterministic shuffling within genre buckets (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without running ffmpeg or writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve splits directory default
    splits_dir: Path = (
        args.splits_dir if args.splits_dir else (args.mixes_dir.parent / "splits")
    )

    log.info("Labels:     %s", args.labels)
    log.info("Mixes dir:  %s", args.mixes_dir)
    log.info("Splits dir: %s", splits_dir)
    log.info("Output:     %s", args.output)
    log.info("Seed:       %d", args.seed)
    if args.dry_run:
        log.info("DRY-RUN mode -- no files will be written.")

    # ---- Load labels.json --------------------------------------------------
    if not args.labels.exists():
        log.error("labels.json not found: %s", args.labels)
        return 1

    log.info("Loading labels from %s ...", args.labels)
    with args.labels.open(encoding="utf-8") as fh:
        labels_data: dict[str, Any] = json.load(fh)

    schema_ver = labels_data.get("schema_version")
    if schema_ver != SCHEMA_VERSION_IN:
        log.warning(
            "Expected schema_version=%d in labels.json, got %s. Proceeding anyway.",
            SCHEMA_VERSION_IN,
            schema_ver,
        )

    files: dict[str, dict[str, Any]] = labels_data.get("files", {})
    if not files:
        log.error("No files found in labels.json.")
        return 1

    log.info("Loaded %d files from labels.json.", len(files))

    # ---- Run splitting algorithm -------------------------------------------
    train, val, test, uncovered, wav_splits, split_half_entries = assign_splits(
        files=files,
        mixes_dir=args.mixes_dir,
        splits_dir=splits_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    log.info(
        "Split sizes -- train: %d  val: %d  test: %d",
        len(train),
        len(val),
        len(test),
    )
    if uncovered:
        log.warning(
            "Uncovered genres (%d): %s",
            len(set(uncovered)),
            ", ".join(sorted(set(uncovered))),
        )
    if wav_splits:
        log.info(
            "WAV-split mixes (%d): %s",
            len(wav_splits),
            ", ".join(wav_splits),
        )

    # ---- Assemble and write output -----------------------------------------
    output_data = assemble_output(
        train=train,
        val=val,
        test=test,
        files=files,
        split_half_entries=split_half_entries,
        uncovered_genres=uncovered,
        wav_split_count=len(wav_splits),
        seed=args.seed,
    )

    write_json_atomic(output_data, args.output, dry_run=args.dry_run)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
