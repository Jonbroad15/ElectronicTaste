#!/usr/bin/env python3
"""Download djmix-dataset audio onto the GCP VM.

Fetches the djmix-dataset manifest from GitHub, filters to the 5 target
EDM classes, downloads each mix with yt-dlp (SoundCloud / Mixcloud /
YouTube / HTTP), and organises files into per-class subdirectories.

Usage (on GCP VM)::

    python scripts/download_raveform.py \
        --output-dir /mnt/data/djmix \
        --workers 8 \
        --max-per-class 600

Outputs:
    <output-dir>/<class>/mix_XXXXXXX.wav
    <output-dir>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

MANIFEST_URL = (
    "https://raw.githubusercontent.com/mir-aidj/djmix-dataset"
    "/main/dataset/djmix-dataset.json"
)

# Map raw MixesDB category tags → our 5-class taxonomy
TAG_TO_CLASS: dict[str, str] = {
    "category:techno":          "techno",
    "category:house":           "house",
    "category:tech_house":      "house",
    "category:tech house":      "house",
    "category:trance":          "trance",
    "category:drum & bass":     "drum and bass",
    "category:drum_and_bass":   "drum and bass",
    "category:drum and bass":   "drum and bass",
    "category:dubstep":         "dubstep",
}

log = logging.getLogger(__name__)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def fetch_manifest(url: str) -> list[dict]:
    log.info("Fetching manifest from %s …", url)
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def classify_mix(entry: dict) -> str | None:
    """Return the target class for an entry, or None if it should be skipped."""
    for tag in entry.get("tags", []):
        key = tag.get("key", "").lower().strip()
        cls = TAG_TO_CLASS.get(key)
        if cls:
            return cls
    return None


def filter_manifest(
    entries: list[dict],
    max_per_class: int | None,
) -> list[tuple[dict, str]]:
    """Return (entry, class_name) pairs respecting per-class caps."""
    counts: dict[str, int] = {}
    result: list[tuple[dict, str]] = []

    for entry in entries:
        if not entry.get("audio_url"):
            continue
        cls = classify_mix(entry)
        if cls is None:
            continue
        if max_per_class and counts.get(cls, 0) >= max_per_class:
            continue
        counts[cls] = counts.get(cls, 0) + 1
        result.append((entry, cls))

    log.info("Selected %d mixes: %s", len(result), counts)
    return result


# ── Download logic ────────────────────────────────────────────────────────────

def _dest_path(output_dir: Path, cls: str, mix_id: str) -> Path:
    return output_dir / cls / f"{mix_id}.wav"


def download_mix(
    entry: dict,
    cls: str,
    output_dir: Path,
    retries: int = 2,
) -> tuple[str, bool, str]:
    """Download a single mix with yt-dlp.  Returns (mix_id, success, message)."""
    mix_id = entry["id"]
    url = entry["audio_url"]
    dest = _dest_path(output_dir, cls, mix_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 1_000_000:
        return mix_id, True, "already exists"

    # yt-dlp: download best audio, convert to mono WAV 24 kHz
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 24000",
        "--concurrent-fragments", "4",
        "--output", str(dest.with_suffix("")),  # yt-dlp appends extension
        url,
    ]

    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30-min cap per mix
            )
            if result.returncode == 0 and dest.exists():
                return mix_id, True, "ok"
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            if attempt < retries:
                time.sleep(5 * (attempt + 1))
        except subprocess.TimeoutExpired:
            err = "timeout"
            if attempt < retries:
                time.sleep(10)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if attempt < retries:
                time.sleep(5)

    return mix_id, False, err


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download djmix-dataset audio for EDM subgenre classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/djmix",
                        help="Root output directory (class subdirs created automatically)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Cap per class (0 = unlimited)")
    parser.add_argument("--manifest-url", default=MANIFEST_URL)
    parser.add_argument("--manifest-cache", default=None,
                        help="Local path to cache the raw JSON manifest")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch or load manifest
    cache = Path(args.manifest_cache) if args.manifest_cache else output_dir / "djmix_manifest_raw.json"
    if cache.exists():
        log.info("Loading cached manifest from %s", cache)
        entries = json.loads(cache.read_text())
    else:
        entries = fetch_manifest(args.manifest_url)
        cache.write_text(json.dumps(entries, indent=2))
        log.info("Manifest cached to %s (%d entries)", cache, len(entries))

    max_per_class = args.max_per_class if args.max_per_class > 0 else None
    selected = filter_manifest(entries, max_per_class)

    if not selected:
        log.error("No mixes matched the target classes — check manifest format.")
        sys.exit(1)

    # Download with thread pool
    successes: list[dict] = []
    failures: list[dict] = []
    total = len(selected)

    log.info("Downloading %d mixes with %d workers …", total, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_mix, entry, cls, output_dir): (entry, cls)
            for entry, cls in selected
        }
        done = 0
        for future in as_completed(futures):
            entry, cls = futures[future]
            try:
                mix_id, ok, msg = future.result()
            except Exception as exc:  # noqa: BLE001
                mix_id, ok, msg = entry["id"], False, str(exc)

            done += 1
            status = "OK" if ok else "FAIL"
            log.info("[%d/%d] %s  %s  (%s)", done, total, status, mix_id, msg)

            record = {"id": mix_id, "class": cls, "url": entry["audio_url"]}
            (successes if ok else failures).append(record)

    # Write manifest
    manifest = {
        "total": total,
        "success": len(successes),
        "failed": len(failures),
        "class_counts": {},
        "entries": successes,
        "failed_entries": failures,
    }
    for rec in successes:
        manifest["class_counts"][rec["class"]] = manifest["class_counts"].get(rec["class"], 0) + 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info("")
    log.info("Download complete: %d/%d succeeded", len(successes), total)
    log.info("Class counts: %s", manifest["class_counts"])
    log.info("Manifest written → %s", manifest_path)
    if failures:
        log.warning("%d mixes failed — see manifest.json failed_entries", len(failures))


if __name__ == "__main__":
    main()
