#!/usr/bin/env python3
"""
annotate_raveform.py
--------------------
Annotates already-downloaded Raveform mixes with hierarchical genre labels
from the pulse.roots taxonomy and consolidates them into a flat mixes/ directory.

Usage:
    python3 scripts/annotate_raveform.py \
        --djmix-dir /mnt/data/djmix \
        --manifest /mnt/data/djmix/djmix_manifest_raw.json \
        --output /mnt/data/djmix/labels.json \
        [--dry-run]
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAXONOMY_URL = (
    "https://raw.githubusercontent.com/Mendiak/pulse.roots/main/data/pulseroots.genres.json"
)
GENRE_TAXONOMY_DISPLAY_URL = (
    "https://github.com/Mendiak/pulse.roots/blob/main/data/pulseroots.genres.json"
)
SCHEMA_VERSION = 2
MAX_DEPTH = 4

# Additional alias mappings: normalised alias -> canonical pulseroots node name.
# Only aliases whose right-hand side actually exists in the parsed taxonomy are kept.
ALIAS_CANDIDATES: dict = {
    "drum & bass": "Drum and Bass",
    "drum_and_bass": "Drum and Bass",
    "dnb": "Drum and Bass",
    "d&b": "Drum and Bass",
    "psytrance": "Psytrance",
    "minimal": "Minimal Techno",
    "deep tech house": "Tech House",
    "dub techno": "Dub Techno",
    "acid": "Acid Techno",
    "progressive": "Progressive House",
    "uk garage": "UK Garage",
    "hardcore": "Hardcore",
    "breakbeat hardcore": "Breakbeat hardcore",
    "liquid": "Liquid Drum and Bass",
    "neurofunk": "Neurofunk",
    "darkstep": "Darkstep",
    "jump up": "Jump Up",
    "footwork": "Footwork",
    "future bass": "Future bass",
    "nu disco": "Nu-disco",
    "italo disco": "Italo disco",
    "moombahton": "Moombahton",
    "gqom": "Gqom",
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 - Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> dict:
    """
    Load djmix_manifest_raw.json and return a dict keyed by mix id.

    Each entry has at minimum:
        id: str | int
        audio_url: str
        tags: list[{"key": "Category:Techno", "url": "..."}]
    """
    log.info("Loading manifest from %s", manifest_path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw: list = json.load(fh)

    index: dict = {}
    for entry in raw:
        mid = str(entry["id"])
        index[mid] = entry

    log.info("Loaded %d manifest entries", len(index))
    return index


# ---------------------------------------------------------------------------
# Step 2 - Fetch taxonomy JSON
# ---------------------------------------------------------------------------


def fetch_taxonomy(url: str) -> list:
    """Download and parse the pulse.roots genre taxonomy JSON."""
    log.info("Fetching genre taxonomy from %s", url)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    log.info("Fetched taxonomy (%d top-level nodes)", len(data))
    return data


# ---------------------------------------------------------------------------
# Step 3 - Build taxonomy index
# ---------------------------------------------------------------------------


def _walk_taxonomy(
    nodes: list,
    ancestors: list,
    depth: int,
    index: dict,
) -> None:
    """Recursively walk a taxonomy tree, populating *index*."""
    for node in nodes:
        name: str = node.get("style") or node.get("name")
        if not name:
            raise KeyError(f"Node missing both 'style' and 'name' keys: {node.keys()}")
        current_ancestors = ancestors + [name]
        index[name] = {
            "depth": depth,
            "ancestors": current_ancestors,
        }
        substyles = node.get("substyles") or []
        if substyles:
            _walk_taxonomy(substyles, current_ancestors, depth + 1, index)


def build_taxonomy_index(taxonomy: list) -> dict:
    """Return a flat dict of node name -> {depth, ancestors}."""
    index: dict = {}
    _walk_taxonomy(taxonomy, [], 1, index)
    log.info("Built taxonomy index with %d nodes", len(index))
    return index


# ---------------------------------------------------------------------------
# Step 4 - Build TAG_TO_GENRE mapping
# ---------------------------------------------------------------------------


def _normalise_tag(raw: str) -> str:
    """Normalise a raw MixesDB tag string for lookup."""
    s = raw.lower()
    # Strip category: prefix
    if s.startswith("category:"):
        s = s[len("category:"):]
    s = s.strip()
    s = s.replace("_", " ")
    return s


def build_tag_to_genre(taxonomy_index: dict) -> dict:
    """
    Build a mapping from normalised tag string -> canonical pulseroots node name.

    Includes:
      - Direct matches for every pulseroots node name (normalised -> canonical)
      - Curated alias candidates (only where the right-hand side exists in the taxonomy)
    """
    mapping: dict = {}

    # Direct matches: all pulseroots node names (normalised -> canonical)
    for canonical in taxonomy_index:
        normalised = _normalise_tag(canonical)
        mapping[normalised] = canonical

    # Alias candidates - only add if the target canonical name exists
    skipped_aliases: list = []
    added_aliases: list = []
    for alias_normalised, target_canonical in ALIAS_CANDIDATES.items():
        if target_canonical in taxonomy_index:
            mapping[alias_normalised] = target_canonical
            added_aliases.append(alias_normalised)
        else:
            skipped_aliases.append(f"{alias_normalised!r} -> {target_canonical!r}")

    log.info(
        "TAG_TO_GENRE: %d entries (%d direct, %d aliases added)",
        len(mapping),
        len(taxonomy_index),
        len(added_aliases),
    )
    if skipped_aliases:
        log.warning(
            "Skipped %d aliases whose target was not found in taxonomy: %s",
            len(skipped_aliases),
            skipped_aliases,
        )

    return mapping


# ---------------------------------------------------------------------------
# Step 5 - Compute hierarchical labels for a manifest entry
# ---------------------------------------------------------------------------


def compute_labels(
    entry: dict,
    tag_to_genre: dict,
    taxonomy_index: dict,
) -> dict:
    """
    Given a manifest entry, return:
        {
            "l1_genres": [...],
            "l2_genres": [...],
            "l3_genres": [...],
            "l4_genres": [...],
        }
    Labels are deduped and sorted.
    """
    tags: list = entry.get("tags") or []

    # Collect all matched canonical node names from the entry's tags
    matched_nodes: set = set()
    for tag_obj in tags:
        raw_key = tag_obj.get("key", "")
        normalised = _normalise_tag(raw_key)
        canonical = tag_to_genre.get(normalised)
        if canonical:
            matched_nodes.add(canonical)

    # For each matched node, walk the ancestor chain and assign labels by depth
    labels_by_depth: dict = defaultdict(set)
    for node_name in matched_nodes:
        node_info = taxonomy_index[node_name]
        ancestors: list = node_info["ancestors"]
        # ancestors[0] = depth-1 root, ancestors[-1] = node_name itself
        for ancestor_name in ancestors:
            ancestor_info = taxonomy_index[ancestor_name]
            depth = ancestor_info["depth"]
            labels_by_depth[depth].add(ancestor_name)

    return {
        "l1_genres": sorted(labels_by_depth.get(1, set())),
        "l2_genres": sorted(labels_by_depth.get(2, set())),
        "l3_genres": sorted(labels_by_depth.get(3, set())),
        "l4_genres": sorted(labels_by_depth.get(4, set())),
    }


# ---------------------------------------------------------------------------
# Step 6 - Scan class subdirs and process files
# ---------------------------------------------------------------------------


def iter_wav_files(djmix_dir: Path) -> list:
    """
    Return a list of (wav_path, mix_id) tuples for every .wav file found in
    all class subdirectories of djmix_dir (everything except the mixes/ output dir).
    """
    results: list = []
    for child in sorted(djmix_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "mixes":
            continue  # skip the output dir
        for wav_path in sorted(child.glob("*.wav")):
            mix_id = wav_path.stem  # e.g. "mix_0000001"
            results.append((wav_path, mix_id))
    return results


# ---------------------------------------------------------------------------
# Atomic JSON write helper
# ---------------------------------------------------------------------------


def write_json_atomic(data: Any, dest: Path) -> None:
    """Write *data* as JSON to *dest* atomically via a .tmp file."""
    tmp = dest.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Label-count aggregation helper
# ---------------------------------------------------------------------------


def compute_label_counts(files: dict) -> dict:
    """Aggregate per-depth label frequency counts across all file entries."""
    counts: dict = {
        "l1": defaultdict(int),
        "l2": defaultdict(int),
        "l3": defaultdict(int),
        "l4": defaultdict(int),
    }
    for file_entry in files.values():
        for level, key in [
            ("l1", "l1_genres"),
            ("l2", "l2_genres"),
            ("l3", "l3_genres"),
            ("l4", "l4_genres"),
        ]:
            for genre in file_entry.get(key, []):
                counts[level][genre] += 1

    return {level: dict(sorted(v.items())) for level, v in counts.items()}


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate Raveform mixes with hierarchical genre labels."
    )
    parser.add_argument(
        "--djmix-dir",
        required=True,
        type=Path,
        help="Root directory containing class subdirs (house/, techno/, ...).",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to djmix_manifest_raw.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for labels.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would happen without moving files or writing anything.",
    )
    args = parser.parse_args()

    djmix_dir: Path = args.djmix_dir.resolve()
    manifest_path: Path = args.manifest.resolve()
    output_path: Path = args.output.resolve()
    dry_run: bool = args.dry_run

    if dry_run:
        log.info("*** DRY-RUN MODE - no files will be moved or written ***")

    if not djmix_dir.is_dir():
        log.error("djmix-dir does not exist: %s", djmix_dir)
        sys.exit(1)
    if not manifest_path.is_file():
        log.error("Manifest file does not exist: %s", manifest_path)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Load manifest
    # ------------------------------------------------------------------
    manifest_index = load_manifest(manifest_path)

    # ------------------------------------------------------------------
    # Step 2: Fetch taxonomy
    # ------------------------------------------------------------------
    taxonomy = fetch_taxonomy(TAXONOMY_URL)

    # ------------------------------------------------------------------
    # Step 3: Build taxonomy index
    # ------------------------------------------------------------------
    taxonomy_index = build_taxonomy_index(taxonomy)

    # ------------------------------------------------------------------
    # Step 4: Build TAG_TO_GENRE
    # ------------------------------------------------------------------
    tag_to_genre = build_tag_to_genre(taxonomy_index)

    # ------------------------------------------------------------------
    # Step 7 (pre): Load existing labels.json for idempotency
    # ------------------------------------------------------------------
    existing_labels: dict = {}
    if output_path.is_file():
        log.info("Loading existing labels.json from %s (idempotent mode)", output_path)
        with output_path.open("r", encoding="utf-8") as fh:
            existing_doc = json.load(fh)
        existing_labels = existing_doc.get("files", {})
        log.info("Found %d already-annotated files", len(existing_labels))

    # Ensure the mixes/ output directory exists
    mixes_dir = djmix_dir / "mixes"
    if not dry_run:
        mixes_dir.mkdir(exist_ok=True)
    else:
        log.info("[dry-run] Would create directory: %s", mixes_dir)

    # ------------------------------------------------------------------
    # Step 6: Scan class subdirs and process .wav files
    # ------------------------------------------------------------------
    wav_files = iter_wav_files(djmix_dir)
    log.info("Found %d .wav files across class subdirs", len(wav_files))

    # Start from existing state
    files_out: dict = dict(existing_labels)

    processed = 0
    skipped_existing = 0
    skipped_no_manifest = 0
    class_dirs_touched: set = set()

    for wav_path, mix_id in wav_files:
        dest_filename = f"{mix_id}.wav"

        # Idempotency: skip if already in labels.json
        if dest_filename in files_out:
            log.debug("Skipping already-annotated file: %s", dest_filename)
            skipped_existing += 1
            continue

        # Look up manifest entry.
        # mix_id is typically "mix_0000001"; the manifest id field may be a
        # number ("1") or the full string.  Try both forms.
        entry = None
        if mix_id.startswith("mix_"):
            numeric_str = mix_id[len("mix_"):]
            try:
                numeric_id = str(int(numeric_str))
                entry = manifest_index.get(numeric_id) or manifest_index.get(mix_id)
            except ValueError:
                entry = manifest_index.get(mix_id)
        else:
            entry = manifest_index.get(mix_id)

        if entry is None:
            log.warning(
                "No manifest entry for mix_id=%s (%s) - skipping",
                mix_id,
                wav_path.name,
            )
            skipped_no_manifest += 1
            continue

        # Compute hierarchical labels
        labels = compute_labels(entry, tag_to_genre, taxonomy_index)

        # Collect original tag keys for provenance
        source_tags = [t.get("key", "") for t in (entry.get("tags") or [])]

        dest_path = mixes_dir / dest_filename

        if dry_run:
            log.info(
                "[dry-run] Would move %s -> %s | l1=%s l2=%s l3=%s l4=%s",
                wav_path,
                dest_path,
                labels["l1_genres"],
                labels["l2_genres"],
                labels["l3_genres"],
                labels["l4_genres"],
            )
        else:
            log.debug("Moving %s -> %s", wav_path, dest_path)
            os.rename(wav_path, dest_path)
            class_dirs_touched.add(wav_path.parent)

        files_out[dest_filename] = {
            "audio_path": f"mixes/{dest_filename}",
            "source_tags": source_tags,
            "l1_genres": labels["l1_genres"],
            "l2_genres": labels["l2_genres"],
            "l3_genres": labels["l3_genres"],
            "l4_genres": labels["l4_genres"],
        }
        processed += 1

    log.info(
        "Processing complete: %d processed, %d skipped (already annotated), "
        "%d skipped (no manifest entry)",
        processed,
        skipped_existing,
        skipped_no_manifest,
    )

    # ------------------------------------------------------------------
    # Step 7: Write labels.json atomically
    # ------------------------------------------------------------------
    label_counts = compute_label_counts(files_out)

    output_doc: dict = {
        "schema_version": SCHEMA_VERSION,
        "genre_taxonomy_url": GENRE_TAXONOMY_DISPLAY_URL,
        "max_depth": MAX_DEPTH,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_counts": label_counts,
        "files": files_out,
    }

    if dry_run:
        log.info(
            "[dry-run] Would write labels.json to %s (%d file entries)",
            output_path,
            len(files_out),
        )
        # Print a preview of what the label counts would look like
        for level, level_counts in label_counts.items():
            top5 = sorted(level_counts.items(), key=lambda x: -x[1])[:5]
            log.info("[dry-run] Top-5 %s labels: %s", level, top5)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_doc, output_path)
        log.info(
            "Wrote labels.json to %s (%d total file entries)",
            output_path,
            len(files_out),
        )

    # ------------------------------------------------------------------
    # Step 8: Remove now-empty class subdirs
    # ------------------------------------------------------------------
    if not dry_run:
        for class_dir in sorted(class_dirs_touched):
            remaining = list(class_dir.iterdir())
            if not remaining:
                try:
                    class_dir.rmdir()
                    log.info("Removed empty class dir: %s", class_dir)
                except OSError as exc:
                    log.warning("Could not remove %s: %s", class_dir, exc)
            else:
                log.info(
                    "Class dir %s still has %d item(s), leaving in place",
                    class_dir,
                    len(remaining),
                )
    else:
        # In dry-run, report which class dirs would be removed
        all_class_dirs: set = set()
        for wav_path, _ in wav_files:
            all_class_dirs.add(wav_path.parent)
        for class_dir in sorted(all_class_dirs):
            remaining = list(class_dir.iterdir())
            if not remaining:
                log.info("[dry-run] Would remove empty class dir: %s", class_dir)

    # ------------------------------------------------------------------
    # Step 9: Final summary
    # ------------------------------------------------------------------
    log.info("--- Summary ---")
    log.info("  Total files in labels.json : %d", len(files_out))
    log.info("  Files processed this run   : %d", processed)
    log.info("  Files skipped (existing)   : %d", skipped_existing)
    log.info("  Files skipped (no manifest): %d", skipped_no_manifest)
    for level in ("l1", "l2", "l3", "l4"):
        total = sum(label_counts[level].values())
        unique = len(label_counts[level])
        log.info(
            "  %s: %d unique labels, %d total occurrences", level, unique, total
        )


if __name__ == "__main__":
    main()
