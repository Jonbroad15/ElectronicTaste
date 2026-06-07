#!/usr/bin/env python3
"""Download djmix-dataset audio onto the GCP VM.

Fetches the djmix-dataset manifest from GitHub (or a local cache), filters
to ALL entries that match at least one node in the pulseroots genre taxonomy,
downloads each mix with yt-dlp (SoundCloud / Mixcloud / YouTube / HTTP),
and writes files to a flat ``mixes/`` directory.

After each successful download the mix's hierarchical labels are computed and
appended atomically to ``labels.json``.  Any mix ID already recorded in
``labels.json`` is skipped, so the script is fully resumable.

Usage (on GCP VM)::

    python3 scripts/download_raveform.py \\
        --output-dir /mnt/data/djmix \\
        --workers 8 \\
        --manifest-cache /mnt/data/djmix/djmix_manifest_raw.json \\
        --labels /mnt/data/djmix/labels.json \\
        --taxonomy-cache /mnt/data/djmix/pulseroots_taxonomy.json

Outputs:
    <output-dir>/mixes/<mix_id>.wav          (flat, no class subdirs)
    <output-dir>/labels.json                 (appended atomically per download)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

# ── Constants ─────────────────────────────────────────────────────────────────

MANIFEST_URL = (
    "https://raw.githubusercontent.com/mir-aidj/djmix-dataset"
    "/main/dataset/djmix-dataset.json"
)

PULSEROOTS_TAXONOMY_URL = (
    "https://raw.githubusercontent.com/Mendiak/pulse.roots"
    "/main/data/pulseroots.genres.json"
)

LABELS_SCHEMA_VERSION = 2

# Hand-curated alias map: normalised MixesDB tag text -> pulseroots node name.
# Keys are lower-cased and stripped; values must exactly match a node "name"
# (or L1 "style") in the pulseroots taxonomy.
TAG_TO_GENRE: dict[str, str] = {
    # ── Techno family ──────────────────────────────────────────────────────
    "techno":                   "Techno",
    "category:techno":          "Techno",
    "minimal":                  "Minimal Techno",
    "minimal techno":           "Minimal Techno",
    "dub techno":               "Dub Techno",
    "acid":                     "Acid Techno",
    "acid techno":              "Acid Techno",
    "industrial techno":        "Industrial Techno",
    "hardcore techno":          "Hardcore Techno",
    "hardtek":                  "Hardtek",
    "hard techno":              "Hard Techno",
    "gabber":                   "Gabber",
    "schranz":                  "Schranz",
    # ── House family ───────────────────────────────────────────────────────
    "house":                    "House",
    "category:house":           "House",
    "tech house":               "Tech House",
    "category:tech_house":      "Tech House",
    "category:tech house":      "Tech House",
    "deep tech house":          "Tech House",
    "deep house":               "Deep House",
    "afro house":               "Afro House",
    "soulful house":            "Soulful House",
    "funky house":              "Funky House",
    "chicago house":            "Chicago House",
    "progressive house":        "Progressive House",
    "progressive":              "Progressive House",
    "electro house":            "Electro House",
    "big room house":           "Big Room House",
    "big room":                 "Big Room House",
    "tribal house":             "Tribal House",
    "garage house":             "Garage House",
    "uk garage":                "UK Garage",
    "speed garage":             "Speed Garage",
    # ── Trance family ──────────────────────────────────────────────────────
    "trance":                   "Trance",
    "category:trance":          "Trance",
    "progressive trance":       "Progressive Trance",
    "psytrance":                "Psytrance",
    "psy trance":               "Psytrance",
    "psychedelic trance":       "Psytrance",
    "goa trance":               "Goa Trance",
    "dark psytrance":           "Dark Psytrance",
    "full on":                  "Full On",
    "forest psytrance":         "Forest Psytrance",
    "uplifting trance":         "Uplifting Trance",
    "vocal trance":             "Vocal Trance",
    "tech trance":              "Tech Trance",
    # ── Drum and Bass / Jungle family ──────────────────────────────────────
    "drum and bass":            "Drum and Bass",
    "drum & bass":              "Drum and Bass",
    "category:drum & bass":     "Drum and Bass",
    "category:drum_and_bass":   "Drum and Bass",
    "category:drum and bass":   "Drum and Bass",
    "dnb":                      "Drum and Bass",
    "d&b":                      "Drum and Bass",
    "liquid":                   "Liquid Drum and Bass",
    "liquid drum and bass":     "Liquid Drum and Bass",
    "neurofunk":                "Neurofunk",
    "darkstep":                 "Darkstep",
    "jump up":                  "Jump Up",
    "jungle":                   "Jungle",
    "ragga jungle":             "Ragga Jungle",
    # ── Dubstep / Bass Music family ────────────────────────────────────────
    "dubstep":                  "Dubstep",
    "category:dubstep":         "Dubstep",
    "brostep":                  "Brostep",
    "post-dubstep":             "Post-Dubstep",
    "future garage":            "Future Garage",
    "riddim":                   "Riddim",
    "trap":                     "Trap (EDM)",
    "edm trap":                 "Trap (EDM)",
    "future bass":              "Future bass",
    "bass music":               "Bass Music",
    "footwork":                 "Footwork",
    "grime":                    "Grime",
    "uk funky":                 "UK Funky",
    "moombahton":               "Moombahton",
    # ── Electro ────────────────────────────────────────────────────────────
    "electro":                  "Electro",
    "electroclash":             "Electroclash",
    "new wave":                 "New Wave",
    # ── Disco family ───────────────────────────────────────────────────────
    "disco":                    "Disco",
    "nu disco":                 "Nu-disco",
    "nu-disco":                 "Nu-disco",
    "italo disco":              "Italo disco",
    "cosmic disco":             "Cosmic Disco",
    "funk":                     "Funk",
    "boogie":                   "Boogie",
    # ── Ambient / Downtempo ────────────────────────────────────────────────
    "ambient":                  "Ambient",
    "downtempo":                "Downtempo",
    "chill out":                "Chillout",
    "chillout":                 "Chillout",
    "trip hop":                 "Trip hop",
    "trip-hop":                 "Trip hop",
    # ── Hardcore / Breakbeat ───────────────────────────────────────────────
    "hardcore":                 "Hardcore",
    "breakbeat hardcore":       "Breakbeat hardcore",
    "happy hardcore":           "Happy Hardcore",
    "breakbeat":                "Breakbeat",
    "breaks":                   "Breakbeat",
    "big beat":                 "Big Beat",
    # ── Afro / Global ──────────────────────────────────────────────────────
    "afrobeat":                 "Afrobeats",
    "afrobeats":                "Afrobeats",
    "gqom":                     "Gqom",
    "amapiano":                 "Amapiano",
    # ── Experimental / IDM ─────────────────────────────────────────────────
    "idm":                      "IDM (Intelligent Dance Music)",
    "intelligent dance music":  "IDM (Intelligent Dance Music)",
    "glitch":                   "Glitch",
}

log = logging.getLogger(__name__)


# ── Taxonomy loading ──────────────────────────────────────────────────────────

def _fetch_json(url_or_path: str) -> object:
    """Load JSON from a URL or local file path."""
    p = Path(url_or_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    with urlopen(url_or_path, timeout=60) as resp:
        return json.loads(resp.read().decode())


def load_taxonomy(
    taxonomy_json_path_or_url: str,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Load and index the pulseroots genre taxonomy.

    Parameters
    ----------
    taxonomy_json_path_or_url:
        Either a local filesystem path or an HTTPS URL pointing to the
        ``pulseroots.genres.json`` file.

    Returns
    -------
    tag_to_genre : dict[str, str]
        Maps normalised MixesDB tag text (lower-cased, stripped) to a
        canonical pulseroots node name.  Combines the hand-curated
        ``TAG_TO_GENRE`` alias map above with every node name itself so
        that tags that already spell out the exact pulseroots name are
        matched without needing an explicit alias.

    taxonomy_index : dict[str, dict]
        Maps every pulseroots node name to ``{"depth": int, "path":
        list[str]}``.  ``depth`` is 1-based (L1 = 1 ... L4 = 4).
        ``path`` is the list of ancestor names from L1 down to (and
        including) the node itself, e.g.
        ``["Techno", "Minimal Techno", "Dub Techno"]``.
    """
    raw = _fetch_json(taxonomy_json_path_or_url)
    taxonomy_index: dict[str, dict] = {}

    def _walk(nodes: list[dict], parent_path: list[str], depth: int) -> None:
        for node in nodes:
            # Top-level objects use the key "style"; nested ones use "name".
            name = node.get("name") or node.get("style", "")
            if not name:
                continue
            current_path = parent_path + [name]
            taxonomy_index[name] = {"depth": depth, "path": current_path}
            substyles = node.get("substyles", [])
            if substyles:
                _walk(substyles, current_path, depth + 1)

    _walk(raw, [], 1)

    # Build combined tag_to_genre: start with the hand-curated aliases, then
    # add every node name (lower-cased) that is not already covered, so that
    # tags which already use exact pulseroots spellings are matched for free.
    tag_to_genre: dict[str, str] = dict(TAG_TO_GENRE)
    for node_name in taxonomy_index:
        key = node_name.lower().strip()
        if key not in tag_to_genre:
            tag_to_genre[key] = node_name

    log.info(
        "Taxonomy loaded: %d nodes, %d tag aliases",
        len(taxonomy_index),
        len(tag_to_genre),
    )
    return tag_to_genre, taxonomy_index


# ── Label building ────────────────────────────────────────────────────────────

def build_labels(
    entry_tags: list[dict],
    tag_to_genre: dict[str, str],
    taxonomy_index: dict[str, dict],
) -> dict:
    """Derive hierarchical multi-labels from a manifest entry's raw tags.

    Parameters
    ----------
    entry_tags:
        The ``tags`` list from a djmix-dataset entry, each element a dict
        with at least a ``"key"`` field (e.g. ``{"key": "Category:Techno"}``).
    tag_to_genre:
        Output of :func:`load_taxonomy`.
    taxonomy_index:
        Output of :func:`load_taxonomy`.

    Returns
    -------
    dict with keys:
        ``source_tags``  - list of raw tag key strings that matched,
        ``l1_genres``    - deduplicated list of L1 genre names,
        ``l2_genres``    - deduplicated list of L2 genre names,
        ``l3_genres``    - deduplicated list of L3 genre names,
        ``l4_genres``    - deduplicated list of L4 genre names.

    Hierarchy propagation is automatic: tagging a leaf node (e.g. Dub Techno
    at L3) also records all ancestors (Minimal Techno at L2, Techno at L1).
    """
    source_tags: list[str] = []
    levels: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    for tag in entry_tags:
        raw_key = tag.get("key", "")
        # Try both the full raw key and the bare text after stripping any
        # "Category:" / "category:" type prefix so hand-curated aliases fire
        # correctly regardless of the prefix casing used in the manifest.
        normalised_raw = raw_key.lower().strip()
        bare = normalised_raw
        for prefix in ("category:", "style:", "genre:"):
            if normalised_raw.startswith(prefix):
                bare = normalised_raw[len(prefix):].strip()
                break

        genre_name = tag_to_genre.get(bare) or tag_to_genre.get(normalised_raw)
        if genre_name is None:
            continue

        node = taxonomy_index.get(genre_name)
        if node is None:
            # Alias points to a name not present in the taxonomy — skip.
            continue

        source_tags.append(raw_key)

        # Propagate through all ancestors in the path (including self).
        for ancestor_name in node["path"]:
            anc_node = taxonomy_index.get(ancestor_name)
            if anc_node is None:
                continue
            depth = anc_node["depth"]
            if depth in levels and ancestor_name not in levels[depth]:
                levels[depth].append(ancestor_name)

    return {
        "source_tags": source_tags,
        "l1_genres": levels[1],
        "l2_genres": levels[2],
        "l3_genres": levels[3],
        "l4_genres": levels[4],
    }


# ── Manifest helpers ──────────────────────────────────────────────────────────

def fetch_manifest(url: str) -> list[dict]:
    """Download the raw djmix-dataset JSON manifest from *url*."""
    log.info("Fetching manifest from %s ...", url)
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode())


def filter_manifest(
    entries: list[dict],
    tag_to_genre: dict[str, str],
    already_done: set[str],
) -> list[dict]:
    """Return entries that have >=1 pulseroots-matching tag and are not done.

    Parameters
    ----------
    entries:
        Full djmix-dataset manifest list.
    tag_to_genre:
        Combined alias + node-name map from :func:`load_taxonomy`.
    already_done:
        Set of mix IDs already present in ``labels.json``; these are skipped.
    """
    result: list[dict] = []
    skipped_no_url = 0
    skipped_no_match = 0
    skipped_done = 0

    for entry in entries:
        if not entry.get("audio_url"):
            skipped_no_url += 1
            continue

        mix_id = entry.get("id", "")
        if mix_id in already_done:
            skipped_done += 1
            continue

        # Check if at least one tag resolves to a pulseroots node.
        matched = False
        for tag in entry.get("tags", []):
            raw = tag.get("key", "").lower().strip()
            bare = raw
            for prefix in ("category:", "style:", "genre:"):
                if raw.startswith(prefix):
                    bare = raw[len(prefix):].strip()
                    break
            if bare in tag_to_genre or raw in tag_to_genre:
                matched = True
                break

        if not matched:
            skipped_no_match += 1
            continue

        result.append(entry)

    log.info(
        "Filter: %d selected  |  %d already done  |  %d no-audio  |  %d no-match",
        len(result),
        skipped_done,
        skipped_no_url,
        skipped_no_match,
    )
    return result


# ── labels.json helpers ───────────────────────────────────────────────────────

def _load_labels(labels_path: Path) -> dict:
    """Load labels.json, or return a fresh skeleton if it does not exist."""
    if labels_path.exists():
        try:
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            # Ensure required top-level keys exist (forward compatibility).
            data.setdefault("schema_version", LABELS_SCHEMA_VERSION)
            data.setdefault("genre_taxonomy_url", PULSEROOTS_TAXONOMY_URL)
            data.setdefault("max_depth", 4)
            data.setdefault("label_counts", {"l1": {}, "l2": {}, "l3": {}, "l4": {}})
            data.setdefault("files", {})
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s (%s) — starting fresh.", labels_path, exc)

    return {
        "schema_version": LABELS_SCHEMA_VERSION,
        "genre_taxonomy_url": PULSEROOTS_TAXONOMY_URL,
        "max_depth": 4,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label_counts": {"l1": {}, "l2": {}, "l3": {}, "l4": {}},
        "files": {},
    }


def _already_done_ids(labels_path: Path) -> set[str]:
    """Return the set of mix IDs already recorded in labels.json."""
    if not labels_path.exists():
        return set()
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        # File keys are "<mix_id>.wav" — strip suffix to get the bare mix ID.
        return {k.removesuffix(".wav") for k in data.get("files", {})}
    except (json.JSONDecodeError, OSError):
        return set()


def _append_to_labels(
    labels_path: Path,
    lock: threading.Lock,
    mix_filename: str,
    audio_rel_path: str,
    mix_labels: dict,
) -> None:
    """Atomically append one file entry to labels.json.

    The write is atomic (write to a sibling .tmp file then os.replace) and the
    entire load-modify-write cycle is protected by *lock* to prevent data races
    between concurrent download worker threads.
    """
    with lock:
        data = _load_labels(labels_path)

        # Add / overwrite the file entry.
        data["files"][mix_filename] = {
            "audio_path": audio_rel_path,
            "source_tags": mix_labels["source_tags"],
            "l1_genres": mix_labels["l1_genres"],
            "l2_genres": mix_labels["l2_genres"],
            "l3_genres": mix_labels["l3_genres"],
            "l4_genres": mix_labels["l4_genres"],
        }

        # Update aggregate label_counts.
        for level_key, genres in [
            ("l1", mix_labels["l1_genres"]),
            ("l2", mix_labels["l2_genres"]),
            ("l3", mix_labels["l3_genres"]),
            ("l4", mix_labels["l4_genres"]),
        ]:
            for genre in genres:
                data["label_counts"][level_key][genre] = (
                    data["label_counts"][level_key].get(genre, 0) + 1
                )

        # Atomic write via sibling .tmp + os.replace (POSIX-atomic).
        tmp_path = labels_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, labels_path)


# ── Download logic ────────────────────────────────────────────────────────────

def _dest_path(output_dir: Path, mix_id: str) -> Path:
    """Return the flat destination path for a mix WAV file."""
    return output_dir / "mixes" / f"{mix_id}.wav"


def download_mix(
    entry: dict,
    output_dir: Path,
    labels_path: Path,
    labels_lock: threading.Lock,
    tag_to_genre: dict[str, str],
    taxonomy_index: dict[str, dict],
    retries: int = 2,
) -> tuple[str, bool, str]:
    """Download a single mix with yt-dlp.

    Parameters
    ----------
    entry:
        Manifest entry dict (must have ``"id"`` and ``"audio_url"``).
    output_dir:
        Root output directory (``mixes/`` subdirectory is created inside it).
    labels_path:
        Path to ``labels.json`` for skip-checking and atomic appending.
    labels_lock:
        Thread lock that guards all read-modify-write operations on
        ``labels.json``.
    tag_to_genre, taxonomy_index:
        Taxonomy helpers from :func:`load_taxonomy`.
    retries:
        Number of extra attempts after the first (default 2 = 3 total tries).

    Returns
    -------
    (mix_id, success, message)
    """
    mix_id: str = entry["id"]
    url: str = entry["audio_url"]
    dest: Path = _dest_path(output_dir, mix_id)
    mix_filename = f"{mix_id}.wav"
    audio_rel_path = f"mixes/{mix_filename}"

    # ── Skip-check 1: already in labels.json ──────────────────────────────
    # This is faster than a disk stat on cold storage and avoids wasted work
    # when the script is restarted after a partial run.
    with labels_lock:
        if labels_path.exists():
            try:
                data = json.loads(labels_path.read_text(encoding="utf-8"))
                if mix_filename in data.get("files", {}):
                    return mix_id, True, "already in labels.json"
            except (json.JSONDecodeError, OSError):
                pass

    # ── Skip-check 2: file already on disk ────────────────────────────────
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        # File exists but not in labels.json — annotate it now.
        mix_labels = build_labels(
            entry.get("tags", []), tag_to_genre, taxonomy_index
        )
        _append_to_labels(
            labels_path, labels_lock, mix_filename, audio_rel_path, mix_labels
        )
        return mix_id, True, "already on disk (annotated)"

    # ── yt-dlp download ────────────────────────────────────────────────────
    # Download best available audio then convert to mono WAV at 24 kHz so
    # that every file in mixes/ has identical format for the training pipeline.
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 24000",
        "--concurrent-fragments", "4",
        # yt-dlp appends the file extension automatically, so strip .wav here.
        "--output", str(dest.with_suffix("")),
        url,
    ]

    err = "unknown error"
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30-minute hard cap per mix
            )
            if result.returncode == 0 and dest.exists():
                # Success — compute labels and append atomically.
                mix_labels = build_labels(
                    entry.get("tags", []), tag_to_genre, taxonomy_index
                )
                _append_to_labels(
                    labels_path, labels_lock, mix_filename, audio_rel_path, mix_labels
                )
                return mix_id, True, "ok"

            # Non-zero exit — capture last stderr line as error summary.
            stderr_lines = result.stderr.strip().splitlines()
            err = stderr_lines[-1] if stderr_lines else "yt-dlp non-zero exit"

        except subprocess.TimeoutExpired:
            err = "timeout (1800s)"
        except Exception as exc:  # noqa: BLE001
            err = str(exc)

        if attempt < retries:
            backoff = 5 * (attempt + 1)
            log.debug(
                "  Retrying %s in %ds (attempt %d/%d)...",
                mix_id, backoff, attempt + 1, retries,
            )
            time.sleep(backoff)

    return mix_id, False, err


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download djmix-dataset audio for all pulseroots-matched genres, "
            "writing files to a flat mixes/ directory and appending "
            "hierarchical labels to labels.json after each successful download."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="data/djmix",
        help="Root output directory (mixes/ subdir is created automatically)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    parser.add_argument(
        "--manifest-url",
        default=MANIFEST_URL,
        help="URL of the djmix-dataset manifest JSON",
    )
    parser.add_argument(
        "--manifest-cache",
        default=None,
        help=(
            "Local path to cache / load the raw manifest JSON.  "
            "If the file already exists it is used directly (no network fetch).  "
            "Defaults to <output-dir>/djmix_manifest_raw.json."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Path to labels.json.  Existing entries are loaded to skip "
            "already-downloaded mixes; new entries are appended atomically "
            "after each successful download.  "
            "Defaults to <output-dir>/labels.json."
        ),
    )
    parser.add_argument(
        "--taxonomy-cache",
        default=None,
        help=(
            "Local path to cache / load the pulseroots taxonomy JSON.  "
            "If the file already exists it is used directly (no network fetch).  "
            "Defaults to <output-dir>/pulseroots_taxonomy.json."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = (
        Path(args.labels) if args.labels else output_dir / "labels.json"
    )
    taxonomy_cache = (
        Path(args.taxonomy_cache)
        if args.taxonomy_cache
        else output_dir / "pulseroots_taxonomy.json"
    )
    manifest_cache = (
        Path(args.manifest_cache)
        if args.manifest_cache
        else output_dir / "djmix_manifest_raw.json"
    )

    # ── 1. Load (or fetch + cache) the pulseroots taxonomy ────────────────────
    if taxonomy_cache.exists():
        log.info("Loading cached taxonomy from %s", taxonomy_cache)
        taxonomy_source = str(taxonomy_cache)
    else:
        log.info(
            "Fetching pulseroots taxonomy from %s ...", PULSEROOTS_TAXONOMY_URL
        )
        # We pass the URL directly to load_taxonomy, which calls _fetch_json.
        taxonomy_source = PULSEROOTS_TAXONOMY_URL

    tag_to_genre, taxonomy_index = load_taxonomy(taxonomy_source)

    # Cache the raw taxonomy JSON locally for future runs without re-fetching.
    if not taxonomy_cache.exists():
        raw_taxonomy = _fetch_json(PULSEROOTS_TAXONOMY_URL)
        taxonomy_cache.write_text(
            json.dumps(raw_taxonomy, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Taxonomy cached -> %s", taxonomy_cache)

    # ── 2. Load (or fetch + cache) the djmix-dataset manifest ─────────────────
    if manifest_cache.exists():
        log.info("Loading cached manifest from %s", manifest_cache)
        entries: list[dict] = json.loads(
            manifest_cache.read_text(encoding="utf-8")
        )
    else:
        entries = fetch_manifest(args.manifest_url)
        manifest_cache.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(
            "Manifest cached -> %s (%d entries)", manifest_cache, len(entries)
        )

    log.info("Manifest loaded: %d total entries", len(entries))

    # ── 3. Determine already-downloaded IDs from labels.json ──────────────────
    already_done = _already_done_ids(labels_path)
    log.info("Labels file: %d mixes already recorded", len(already_done))

    # ── 4. Filter to pulseroots-matching, not-yet-done entries ────────────────
    selected = filter_manifest(entries, tag_to_genre, already_done)

    if not selected:
        log.info("No new mixes to download — all done!")
        sys.exit(0)

    # ── 5. Download with thread pool ──────────────────────────────────────────
    labels_lock = threading.Lock()
    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    total = len(selected)

    log.info("Downloading %d mixes with %d workers ...", total, args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_mix,
                entry,
                output_dir,
                labels_path,
                labels_lock,
                tag_to_genre,
                taxonomy_index,
            ): entry
            for entry in selected
        }

        done = 0
        for future in as_completed(futures):
            entry = futures[future]
            try:
                mix_id, ok, msg = future.result()
            except Exception as exc:  # noqa: BLE001
                mix_id = entry.get("id", "unknown")
                ok, msg = False, str(exc)

            done += 1
            status = "OK  " if ok else "FAIL"
            log.info("[%d/%d] %s  %s  (%s)", done, total, status, mix_id, msg)

            if ok:
                successes.append(mix_id)
            else:
                failures.append((mix_id, msg))

    # ── 6. Final summary ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(
        "Download complete: %d/%d succeeded", len(successes), total
    )
    if failures:
        log.warning("%d mixes failed:", len(failures))
        for fid, ferr in failures:
            log.warning("  FAIL  %s  --  %s", fid, ferr)

    # Log final label_counts from labels.json.
    if labels_path.exists():
        try:
            final_data = json.loads(labels_path.read_text(encoding="utf-8"))
            counts = final_data.get("label_counts", {})
            total_files = len(final_data.get("files", {}))
            log.info("labels.json now contains %d annotated files", total_files)
            for level in ("l1", "l2", "l3", "l4"):
                level_counts = counts.get(level, {})
                if level_counts:
                    top5 = sorted(
                        level_counts.items(), key=lambda x: -x[1]
                    )[:5]
                    log.info(
                        "  %s unique genres: %d  (top-5: %s)",
                        level.upper(),
                        len(level_counts),
                        ", ".join(f"{g}={n}" for g, n in top5),
                    )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "Could not read final labels.json for summary: %s", exc
            )

    log.info("labels.json -> %s", labels_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
