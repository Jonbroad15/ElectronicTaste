#!/usr/bin/env python3
"""Download and prepare the MTG-Jamendo electronic music subset.

This script:
1. Downloads the MTG-Jamendo metadata TSV from GitHub.
2. Filters to tracks that have at least one tag matching the taxonomy in
   data/taxonomy.json.
3. Maps each track to its canonical EDM subgenre (best-matching class).
4. Writes a manifest CSV to data/mtg_jamendo_manifest.csv.
5. Optionally downloads 30-second preview audio clips to
   data/mtg_jamendo/<canonical_label>/ when --download-audio is passed.

Usage::

    # Manifest only (fast — no audio download):
    python scripts/download_mtg_jamendo.py

    # Manifest + audio (slow — downloads ~16K clips):
    python scripts/download_mtg_jamendo.py --download-audio

    # Limit to N tracks per class for quick testing:
    python scripts/download_mtg_jamendo.py --download-audio --max-per-class 100

    # Use a custom output directory:
    python scripts/download_mtg_jamendo.py --data-dir data/custom_jamendo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

# Official MTG-Jamendo metadata TSV (30s clips, clean tags, ≥50 artists)
METADATA_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
    "data/raw_30s_cleantags_50artists.tsv"
)

# Jamendo audio preview CDN pattern.
# The TSV's TRACK_ID column is a plain integer; the full 30s preview URL uses:
#   https://cdn.freemusicarchive.org/… is NOT the right CDN for Jamendo.
# MTG-Jamendo distributes audio as tar archives from:
#   https://cdn.freemusicarchive.org is wrong; use:
#   https://cdn.jamendo.com/tracks/files/<track_id>
# Per the official docs:
#   mp3 preview (96kbps 30s):
#     https://www.jamendo.com/track/<track_id>/audio
# For programmatic access Jamendo provides:
#   https://api.jamendo.com/v3.0/tracks/file/?client_id=<ID>&id=<track_id>&audioformat=mp32
# We use the public preview redirect which does NOT require an API key:
#   https://www.jamendo.com/track/<track_id>/audio (redirects to CDN)
# The MTG-Jamendo paper recommends their provided download scripts for bulk
# use; here we construct per-track preview URLs that work without credentials.

AUDIO_URL_TEMPLATE = "https://www.jamendo.com/track/{track_id}/audio"

# Fallback direct CDN pattern (avoids redirect for batch downloads)
AUDIO_CDN_TEMPLATE = (
    "https://storage.jamendo.com/?trackid={track_id}&format=mp32&from=app-97dab294"
)

DEFAULT_TAXONOMY_PATH = Path("data/taxonomy.json")
DEFAULT_MANIFEST_PATH = Path("data/mtg_jamendo_manifest.csv")
DEFAULT_AUDIO_DIR = Path("data/mtg_jamendo")


# ── Tag normalisation ─────────────────────────────────────────────────────────

def _normalise(tag: str) -> str:
    """Lowercase, strip category prefix, collapse internal whitespace.

    MTG-Jamendo tags use a ``category---value`` format, e.g. ``genre---techno``.
    We strip everything up to and including ``---`` before looking up in the
    taxonomy, so the tag_map can use plain human-readable strings.
    """
    tag = tag.lower().strip()
    if "---" in tag:
        tag = tag.split("---", 1)[1]
    return " ".join(tag.split())


def build_tag_lookup(taxonomy: dict) -> Dict[str, str]:
    """Return a dict mapping every normalised raw tag → canonical class name.

    Args:
        taxonomy: Parsed taxonomy.json contents.

    Returns:
        Lookup dict e.g. ``{"techno": "techno", "chicago house": "house", ...}``
    """
    lookup: Dict[str, str] = {}
    for canonical, raw_tags in taxonomy["tag_map"].items():
        for raw in raw_tags:
            lookup[_normalise(raw)] = canonical
    return lookup


# ── TSV parsing ───────────────────────────────────────────────────────────────

def parse_tsv_row(
    header: List[str],
    row: List[str],
    tag_lookup: Dict[str, str],
) -> Optional[Tuple[str, str, str, str]]:
    """Extract relevant fields from a single TSV row.

    MTG-Jamendo TSV layout
    ----------------------
    The first 5 fixed columns are TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION.
    All remaining columns are tag strings (one tag per column, variable count).
    The header only declares 6 columns (the 6th being "TAGS") but rows routinely
    have 7–15 columns when a track has multiple tags.  We therefore read the
    first 5 fixed fields and treat everything from index 5 onward as tags.

    Args:
        header:     Column names from the TSV header row (used only to locate
                    the TRACK_ID column index).
        row:        Field values for one track.
        tag_lookup: Normalised tag → canonical class mapping.

    Returns:
        ``(track_id, audio_url, canonical_label, raw_tags_str)`` or ``None``
        if the track has no matching subgenre tag.
    """
    if len(row) < 6:
        return None

    # Fixed columns by position (more robust than dict lookup given variable width)
    track_id = row[0].strip()
    # Tags start at column 5; every column from 5 onward is one tag string.
    tag_columns = [c.strip() for c in row[5:] if c.strip()]

    if not track_id or not tag_columns:
        return None

    raw_tags = [_normalise(t) for t in tag_columns]

    # Determine canonical label: first tag that hits the lookup wins.
    # Iterating in column order preserves MTG-Jamendo's tag priority.
    canonical = None
    matched_tags: List[str] = []

    for tag in raw_tags:
        if tag in tag_lookup:
            matched_tags.append(tag)
            if canonical is None:
                canonical = tag_lookup[tag]

    if canonical is None:
        return None

    audio_url = AUDIO_URL_TEMPLATE.format(track_id=track_id)
    return track_id, audio_url, canonical, ",".join(matched_tags)


# ── Downloading helpers ───────────────────────────────────────────────────────

def download_text(url: str, timeout: int = 60) -> str:
    """Download a URL and return its content as a UTF-8 string."""
    req = urllib.request.Request(url, headers={"User-Agent": "ElectronicTaste/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def download_audio(
    url: str,
    dest: Path,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 30,
) -> bool:
    """Download an audio file to ``dest``, with simple retry logic.

    Returns:
        ``True`` on success, ``False`` after exhausting retries.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ElectronicTaste/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as exc:
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                print(f"  [WARN] Failed to download {url}: {exc}", file=sys.stderr)
    return False


# ── Main logic ────────────────────────────────────────────────────────────────

def build_manifest(
    taxonomy_path: Path,
    manifest_path: Path,
) -> List[dict]:
    """Download TSV, filter to electronic tracks, write manifest CSV.

    Returns:
        List of manifest row dicts for further processing.
    """
    print(f"Loading taxonomy from {taxonomy_path} …")
    with open(taxonomy_path) as fh:
        taxonomy = json.load(fh)

    classes = taxonomy["classes"]
    tag_lookup = build_tag_lookup(taxonomy)
    print(f"  {len(classes)} canonical classes, {len(tag_lookup)} raw tags indexed.")

    print(f"\nDownloading MTG-Jamendo metadata from:\n  {METADATA_URL}")
    tsv_text = download_text(METADATA_URL)
    lines = tsv_text.splitlines()
    print(f"  Downloaded {len(lines):,} lines.")

    # Parse header — the raw_30s_cleantags_50artists.tsv uses tab separators
    header = [col.strip() for col in lines[0].split("\t")]
    print(f"  Columns: {header}")

    # Counters
    per_class: Dict[str, List[dict]] = {c: [] for c in classes}
    skipped = 0

    for line in lines[1:]:
        if not line.strip():
            continue
        row = [field.strip() for field in line.split("\t")]
        result = parse_tsv_row(header, row, tag_lookup)
        if result is None:
            skipped += 1
            continue
        track_id, audio_url, canonical, raw_tags = result
        per_class[canonical].append(
            {
                "track_id": track_id,
                "url": audio_url,
                "canonical_label": canonical,
                "raw_tags": raw_tags,
            }
        )

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    for rows in per_class.values():
        all_rows.extend(rows)

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["track_id", "url", "canonical_label", "raw_tags"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Print statistics
    print(f"\n{'─' * 50}")
    print(f"{'Class':<22}  {'Tracks':>7}")
    print(f"{'─' * 50}")
    total = 0
    for cls in classes:
        n = len(per_class[cls])
        total += n
        flag = "  [< 50 — warn]" if 0 < n < 50 else ""
        print(f"  {cls:<20}  {n:>7}{flag}")
    print(f"{'─' * 50}")
    print(f"  {'TOTAL':<20}  {total:>7}")
    print(f"  {'(skipped)':<20}  {skipped:>7}")
    print(f"{'─' * 50}\n")
    print(f"Manifest written → {manifest_path}  ({total} rows)")

    return all_rows


def download_audio_files(
    rows: List[dict],
    audio_dir: Path,
    max_per_class: Optional[int],
) -> None:
    """Download 30-second preview clips for all rows in the manifest.

    Audio is saved to ``audio_dir/<canonical_label>/<track_id>.mp3``.
    """
    # Apply per-class cap
    if max_per_class is not None:
        from collections import Counter
        counts: Counter = Counter()
        capped: List[dict] = []
        for row in rows:
            lbl = row["canonical_label"]
            if counts[lbl] < max_per_class:
                capped.append(row)
                counts[lbl] += 1
        rows = capped
        print(f"Capped at {max_per_class} per class → {len(rows)} tracks to download.")

    print(f"\nDownloading {len(rows)} audio clips to {audio_dir} …")
    ok = 0
    fail = 0
    for i, row in enumerate(rows, 1):
        dest = audio_dir / row["canonical_label"] / f"{row['track_id']}.mp3"
        if dest.exists():
            ok += 1
            continue
        success = download_audio(row["url"], dest)
        if success:
            ok += 1
        else:
            fail += 1
        if i % 50 == 0 or i == len(rows):
            print(
                f"  Progress: {i}/{len(rows)}  ok={ok}  fail={fail}",
                end="\r",
                flush=True,
            )

    print(f"\nDownload complete: {ok} ok, {fail} failed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and filter the MTG-Jamendo electronic music subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--taxonomy",
        default=str(DEFAULT_TAXONOMY_PATH),
        help="Path to data/taxonomy.json",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_AUDIO_DIR),
        help="Root directory for downloaded audio files",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        default=False,
        help="Download 30-second audio preview clips (slow — may take hours)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        metavar="N",
        help="Maximum tracks to download per class (for quick testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    taxonomy_path = Path(args.taxonomy)
    manifest_path = Path(args.manifest)
    audio_dir = Path(args.data_dir)

    if not taxonomy_path.exists():
        print(
            f"ERROR: Taxonomy file not found: {taxonomy_path}\n"
            "Run from the project root or pass --taxonomy <path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = build_manifest(taxonomy_path, manifest_path)

    if args.download_audio:
        download_audio_files(rows, audio_dir, args.max_per_class)
    else:
        print(
            "Skipping audio download (pass --download-audio to fetch clips).\n"
            f"Audio directory would be: {audio_dir}/<canonical_label>/<track_id>.mp3"
        )


if __name__ == "__main__":
    main()
