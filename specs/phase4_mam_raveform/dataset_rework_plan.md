# Raveform Dataset Rework Plan — Hierarchical Multi-Label Annotation

> **Goal**: Expand the existing 5-class Raveform dataset (already downloaded on the GCP 2 TB data disk) into a rich, hierarchically-structured multi-label dataset governed by the [pulseroots genre taxonomy](https://github.com/Mendiak/pulse.roots/blob/main/data/pulseroots.genres.json), without re-downloading any audio that already exists on disk. Additionally create deterministic train/val splits (`splits.json`) that guarantee coverage of every genre node.

---

## 1. Context — What Already Exists on GCP

The 2 TB persistent disk (`electronic-taste-raveform-data`, zone `us-central1-a`) is already provisioned and populated from the first download run. It is **not attached to any running VM** (the training VM was deleted to stop billing).

Current disk layout at `/mnt/data/djmix/`:

```
/mnt/data/djmix/
  drum and bass/        ← ~507 mixes (capped at 600 per class)
  dubstep/              ← ~54 mixes
  house/                ← ~600 mixes (capped)
  techno/               ← ~600 mixes (capped)
  trance/               ← ~559 mixes
  djmix_manifest_raw.json   ← complete 5,040-entry manifest (16 MB)
  manifest.json             ← filtered 5-class manifest (327 KB)
  labels.json               ← (created in Phase B)
  splits.json               ← (created in Phase D+)
```

**What's missing from the original dataset:**
- ~1,410 Tech House mixes (were merged → `house` or dropped)
- ~672 Progressive House mixes
- ~330 Progressive Trance mixes
- ~305 Deep House mixes
- ~113 Disco mixes
- ~70 Electro mixes
- ~66 Psytrance mixes
- Dozens of other genre-tagged mixes (Minimal Techno, Dub Techno, Hardcore, etc.)
- All labels are flat — no hierarchy, no multi-label, no sub-genre detail

---

## 2. Label Schema

### 2.1 Hierarchy Source

The [pulseroots genre taxonomy](https://github.com/Mendiak/pulse.roots/blob/main/data/pulseroots.genres.json) defines **307 genre nodes** across **14 top-level families**, with a **maximum depth of 4**.

| Level | Count | Example |
|---|---|---|
| L1 | 14 nodes | `Techno`, `House`, `Bass Music` |
| L2 | 100 nodes | `Minimal Techno`, `Deep House`, `Drum and Bass` |
| L3 | 162 nodes | `Dub Techno`, `Soulful House`, `Liquid Drum and Bass` |
| L4 | 31 nodes | `Dark Psytrance`, `Gabber`, `Liquid Funk` |

### 2.2 Per-File Label Format

Each file entry in `labels.json` stores labels bucketed by hierarchy depth. A file may have labels from **multiple separate branches** simultaneously (e.g. a mix spanning Techno and Electro).

```json
{
  "mix_0001234.wav": {
    "audio_path": "mixes/mix_0001234.wav",
    "source_tags": ["Category:Techno", "Category:Dub_Techno", "Category:Electro"],
    "l1_genres": ["Techno", "Electro"],
    "l2_genres": ["Minimal Techno"],
    "l3_genres": ["Dub Techno"],
    "l4_genres": []
  }
}
```

> [!NOTE]
> **Hierarchy propagation is automatic.** Tagging a leaf `Dub Techno (L3)` also adds its ancestors `Minimal Techno (L2)` and `Techno (L1)` for free. Every matched tag enriches all ancestor levels.

---

## 3. Tag-to-Genre Mapping

### 3.1 Updated `TAG_TO_GENRE` Dictionary

The existing `TAG_TO_CLASS` dict in `download_raveform.py` is replaced with `TAG_TO_GENRE` which:
- Maps raw MixesDB `Category:*` tags to their pulseroots node names
- **No longer collapses** `tech house → house`; Tech House is its own L2 node
- Covers all 40 directly matchable tags plus a hand-curated alias list

**Key removals from old mapping:**
- ~~`category:tech_house` → `house`~~ → now maps to `Tech House` (L2 under House)
- ~~`category:tech house` → `house`~~ → now maps to `Tech House` (L2 under House)

**Key aliases added** (MixesDB variant → pulseroots canonical):

| Raw MixesDB tag | Pulseroots node |
|---|---|
| `drum & bass`, `drum_and_bass`, `drum and bass`, `dnb`, `d&b` | `Drum and Bass` |
| `psytrance` | `Psytrance` |
| `minimal` | `Minimal Techno` |
| `deep tech house` | `Tech House` |
| `dub techno` | `Dub Techno` |
| `acid` | `Acid Techno` |
| `progressive` | `Progressive House` |
| `uk garage` | `UK Garage` |
| `hardcore` | `Hardcore` |
| `breakbeat hardcore` | `Breakbeat hardcore` |
| `liquid` | `Liquid Drum and Bass` |
| `neurofunk` | `Neurofunk` |
| `darkstep` | `Darkstep` |
| `jump up` | `Jump Up` |
| `footwork` | `Footwork` |
| `trap` | `Trap (EDM)` |
| `future bass` | `Future bass` |
| `nu disco` | `Nu-disco` |
| `italo disco` | `Italo disco` |
| `moombahton` | `Moombahton` |
| `gqom` | `Gqom` |

### 3.2 Hierarchy Propagation Algorithm

Once a tag maps to any pulseroots node (leaf or otherwise), the full path from root is walked to populate all ancestor levels:

```python
def propagate(leaf_name: str, taxonomy_index: dict) -> dict[int, list[str]]:
    """Given a matched node name, return labels at each depth level including ancestors."""
    node = taxonomy_index[leaf_name]    # look up node + path from root
    result = {1: [], 2: [], 3: [], 4: []}
    for ancestor in node.path_from_root:   # e.g. [Techno(L1), Minimal Techno(L2), Dub Techno(L3)]
        result[ancestor.depth].append(ancestor.name)
    return result
```

Per-file labels are the **union** of propagated results across all matched tags (deduped per level).

---

## 4. Target File Structure (on `/mnt/data/`)

All existing audio is **moved** (not copied) from per-class subdirs into a single flat `mixes/` directory. Class subdirs are then removed. `labels.json` becomes the single source of truth for all labels.

```
/mnt/data/djmix/
  mixes/
    mix_0000001.wav         ← moved from techno/
    mix_0000002.wav         ← moved from house/
    mix_0000099.wav         ← newly downloaded (expanded genres)
    ...
  splits/
    mix_0000099_A.wav       ← first half of a rare-genre file → train
    mix_0000099_B.wav       ← second half → val
    ...
  djmix_manifest_raw.json   ← retained untouched (source of truth for raw tags)
  labels.json               ← NEW: hierarchical multi-label manifest
  splits.json               ← NEW: train/val split assignments
  download_raveform.log     ← existing download log (retained)
```

---

## 5. New & Updated Scripts

| Script | Status | Purpose |
|---|---|---|
| `scripts/annotate_raveform.py` | **NEW** | Run on GCP VM: flatten dirs → `mixes/`, annotate existing files → `labels.json` |
| `scripts/download_raveform.py` | **UPDATED** | Expanded to all pulseroots-matching genres, flat output to `mixes/`, appends to `labels.json`, skips already-annotated IDs |
| `scripts/gcp_setup_download.sh` | **UPDATED** | Installs deps, runs annotate synchronously, then launches expanded download in tmux |
| `scripts/create_splits.py` | **NEW** | Reads `labels.json`, produces `splits.json` with guaranteed per-genre coverage; splits WAVs in half for rare genres |
| `src/training/dataset.py` | **UPDATED** | Reads `labels.json` + `splits.json`, supports configurable depth level + multi-label BCE loss |

---

## 6. Phased Execution on GCP

### Phase A — Re-provision the Download VM (run locally)

The 2 TB data disk already exists. We only need to spin up the cheap CPU VM and re-attach it.
`gcp_provision_raveform.sh` is already idempotent — it skips disk/VM creation if they already exist.

```bash
# Run locally — creates download VM and re-attaches existing 2 TB disk
bash scripts/gcp_provision_raveform.sh

# SSH into download VM
gcloud compute ssh electronic-taste-download \
    --zone=us-central1-a \
    --project=project-58e658a7-9bc6-41eb-876
```

---

### Phase B — Annotate Existing Files (on download VM, synchronous ~5 min)

After SSHing in, run the updated setup script. The updated `gcp_setup_download.sh` will:
1. Mount `/mnt/data/`
2. Install deps (`yt-dlp`, `ffmpeg`, `tmux`, `python3-pip`, `requests`)
3. Run `annotate_raveform.py` **synchronously** before launching any downloads

```bash
# On electronic-taste-download VM:
bash ~/ElectronicTaste/scripts/gcp_setup_download.sh
```

**What `annotate_raveform.py` does:**
1. Loads `/mnt/data/djmix/djmix_manifest_raw.json` (already on disk — no network needed)
2. Fetches `pulseroots.genres.json` from GitHub (~100 KB, one HTTP call)
3. Builds the `TAG_TO_GENRE` alias map + full hierarchy index (depth, parent path per node)
4. Scans all 5 class subdirs (`house/`, `techno/`, etc.) for `.wav` files
5. For each file, extracts the mix ID from the filename, looks up its entry in the raw manifest
6. Applies tag → pulseroots mapping + ancestor propagation → builds `l1_genres`…`l4_genres`
7. **Moves** the file to `mixes/<mix_id>.wav` (`os.rename` — instant on same filesystem)
8. Appends its entry to `labels.json`
9. After all files are processed, removes now-empty class subdirectories

**Expected terminal output:**
```
Loading manifest: /mnt/data/djmix/djmix_manifest_raw.json (5040 entries)
Fetching pulseroots taxonomy...
Building TAG_TO_GENRE index (307 nodes)...
Scanning existing class dirs...
Annotating 2320 existing mixes...
  [2320/2320] mix_0003847.wav → l1:[Techno] l2:[Minimal Techno] l3:[Dub Techno]
Moved 2320 files to mixes/
Wrote labels.json (2320 entries)
L1 coverage: 8/14 top-level genres
Mean labels/file: 2.8
Done in 4m 12s
```

---

### Phase C — Download Remaining Mixes (on download VM, in tmux ~20–30 hr)

The setup script then automatically launches the expanded download in a background tmux session:

```bash
# Attach to monitor progress:
tmux attach -t download

# Or tail the log:
tail -f /mnt/data/download_raveform_expanded.log
```

**How the updated `download_raveform.py` differs:**
- Iterates **all** 5,040 manifest entries, not just the original 5-class whitelist
- Includes a mix if it has ≥ 1 tag mapping to any pulseroots node
- Skips any mix ID already present in `labels.json` (the annotated existing files)
- Downloads directly to `/mnt/data/djmix/mixes/<mix_id>.wav` (no class subdirs)
- After each successful download, annotates + appends atomically to `labels.json`

**Updated launch args in `gcp_setup_download.sh`:**
```bash
python3 scripts/download_raveform.py \
    --output-dir /mnt/data/djmix \
    --workers 8 \
    --manifest-cache /mnt/data/djmix/djmix_manifest_raw.json \
    --labels /mnt/data/djmix/labels.json \
    2>&1 | tee /mnt/data/download_raveform_expanded.log
```

Estimated additional downloads: **~2,700 mixes** (~1,500 new-genre matches not in original 5 classes).

---

### Phase D — Create Splits & Verify (on download VM, then locally)

**Step 1 — Generate `splits.json`** (run after download finishes):

```bash
# On VM:
python3 scripts/create_splits.py \
    --labels /mnt/data/djmix/labels.json \
    --mixes-dir /mnt/data/djmix/mixes \
    --splits-dir /mnt/data/djmix/splits \
    --output /mnt/data/djmix/splits.json
```

**Step 2 — Verify final state:**

```bash
python3 -c "
import json, statistics
labels = json.load(open('/mnt/data/djmix/labels.json'))
files = labels['files']
print(f'Total annotated mixes: {len(files)}')
counts = [sum(len(v) for k,v in f.items() if k.startswith('l')) for f in files.values()]
print(f'Mean labels/file: {statistics.mean(counts):.2f}')
for lvl in ['l1','l2','l3','l4']:
    print(f'  {lvl} unique genres: {len(labels[\"label_counts\"][lvl])}')

splits = json.load(open('/mnt/data/djmix/splits.json'))
print(f'Train files: {len(splits[\"train\"])}')
print(f'Val files:   {len(splits[\"val\"])}')
print(f'Test files:  {len(splits[\"test\"])}')
print(f'Genres with WAV splits: {splits[\"stats\"][\"genres_wav_split\"]}')
"
ls /mnt/data/djmix/mixes/ /mnt/data/djmix/splits/ | wc -l
df -h /mnt/data
```

Then stop billing by deleting the CPU VM from your **local machine** (disk is preserved):

```bash
gcloud compute instances delete electronic-taste-download \
    --zone=us-central1-a \
    --project=project-58e658a7-9bc6-41eb-876 \
    --quiet
```

---

### Phase E — Training (GPU VM, same as existing plan)

Provision the GPU VM per the existing `gcp_provision_training.sh`. The training script is updated to read `labels.json` instead of scanning class subdirs.

```bash
# Local: spin up GPU VM, re-attach data disk, sync updated code
bash scripts/gcp_provision_training.sh

# SSH into training VM
gcloud compute ssh electronic-taste-train \
    --zone=us-central1-a \
    --project=project-58e658a7-9bc6-41eb-876

# On VM: mount disk, install PyTorch, launch MAM pre-training
bash ~/ElectronicTaste/scripts/gcp_setup_training.sh
```

---

## 7. `labels.json` Full Schema

```json
{
  "schema_version": 2,
  "genre_taxonomy_url": "https://github.com/Mendiak/pulse.roots/blob/main/data/pulseroots.genres.json",
  "max_depth": 4,
  "created_at": "2026-06-07T00:00:00Z",
  "label_counts": {
    "l1": {"Techno": 1500, "House": 2200, "Bass Music": 520, "Trance": 559},
    "l2": {"Minimal Techno": 300, "Tech House": 1410, "Deep House": 305},
    "l3": {"Dub Techno": 85},
    "l4": {}
  },
  "files": {
    "mix_0000001.wav": {
      "audio_path": "mixes/mix_0000001.wav",
      "source_tags": ["Category:Techno", "Category:Dub_Techno"],
      "l1_genres": ["Techno"],
      "l2_genres": ["Minimal Techno"],
      "l3_genres": ["Dub Techno"],
      "l4_genres": []
    },
    "mix_0000002.wav": {
      "audio_path": "mixes/mix_0000002.wav",
      "source_tags": ["Category:Tech_House", "Category:House", "Category:Electro"],
      "l1_genres": ["House", "Electro"],
      "l2_genres": ["Tech House"],
      "l3_genres": [],
      "l4_genres": []
    }
  }
}
```

---

## 8. `splits.json` — Train/Val Split Strategy

### 8.1 Goal

Every genre node present in `labels.json` must appear in all three splits — **train, val, and test** — giving the downstream classifier held-out coverage of every genre at both validation and evaluation time.

The split is constructed **per genre node** (independently at all active levels L1–L4), not per file. A single file may appear in multiple genre buckets.

### 8.2 Splitting Rules

| Genre mix count | Action |
|---|---|
| ≥ 3 mixes | 1 file → `val`, 1 file → `test`, rest → `train`. Prefer the two files closest to the median duration for val/test to avoid outliers. |
| 2 mixes | 1 → `val`, 1 → `test`. These 2 files still cover all three splits because train inherits them from other genre buckets where they appear. |
| 1 mix | Split the WAV at the 50% mark → `splits/<mix_id>_A.wav` (val) + `splits/<mix_id>_B.wav` (test). The original full file stays in `mixes/` and goes to train. All three clips inherit the parent's full label set. |
| 0 mixes | Genre node has no audio — logged as `uncovered_genres` in `splits.json`. |

> [!NOTE]
> The audio halving for rare genres (1-mix case) is done with `ffmpeg -c copy` — **lossless WAV stream copy, no re-encoding**. The original file in `mixes/` is **retained** and assigned to train.

### 8.3 Audio Splitting Script — `scripts/create_splits.py`

**Inputs:**
- `--labels /mnt/data/djmix/labels.json`
- `--mixes-dir /mnt/data/djmix/mixes`
- `--splits-dir /mnt/data/djmix/splits` (created if absent)
- `--output /mnt/data/djmix/splits.json`
- `--seed 42` (optional; controls shuffling within genre buckets with > 3 members)

**Algorithm (pseudocode):**

```python
for level in [l1, l2, l3, l4]:
    for genre in all_genres_at_level:
        mixes = files_with_genre_at_level(genre, level)  # from labels.json
        mixes = sort_by_duration(mixes)                  # stable ordering

        if len(mixes) == 0:
            mark uncovered_genres

        elif len(mixes) == 1:
            mix = mixes[0]
            duration = get_wav_duration(mix)             # via ffprobe
            half = duration / 2
            run(ffmpeg -i mix -t half      -c copy splits/<id>_A.wav)  # val
            run(ffmpeg -i mix -ss half     -c copy splits/<id>_B.wav)  # test
            train.add(mix)                               # original → train
            val.add(splits/<id>_A.wav,  labels=mix.labels, is_split_half=True)
            test.add(splits/<id>_B.wav, labels=mix.labels, is_split_half=True)

        elif len(mixes) == 2:
            val.add(mixes[0])
            test.add(mixes[1])
            # train gets 0 from this bucket — covered via other genre memberships

        else:  # >= 3
            mid = len(mixes) // 2
            val.add(mixes[mid])
            test.add(mixes[mid - 1])                    # neighbour to median
            train.add_all(mixes - {mixes[mid], mixes[mid - 1]})

# Conflict resolution: test > val > train (strictest held-out wins)
train = deduplicate(train - val - test)
val   = deduplicate(val   - test)
test  = deduplicate(test)
```

> [!IMPORTANT]
> **Conflict resolution — test > val > train.** Since genre buckets are independent, a file can be nominated as `train` by one genre and `val` by another. The strictest held-out assignment wins: if a file appears in any `test` bucket it goes to test; if it appears in any `val` bucket (but no test) it goes to val; otherwise it goes to train.

### 8.4 `splits.json` Schema

```json
{
  "schema_version": 2,
  "seed": 42,
  "stats": {
    "total_train": 3180,
    "total_val": 94,
    "total_test": 94,
    "genres_wav_split": 12,
    "uncovered_genres": ["Gqom", "Bakalao", "Makina"]
  },
  "train": [
    {
      "audio_path": "mixes/mix_0000001.wav",
      "l1_genres": ["Techno"],
      "l2_genres": ["Minimal Techno"],
      "l3_genres": ["Dub Techno"],
      "l4_genres": []
    },
    {
      "audio_path": "mixes/mix_0000099.wav",
      "l1_genres": ["House"],
      "l2_genres": ["Garage House"],
      "l3_genres": [],
      "l4_genres": [],
      "note": "original retained as train; A/B halves used for val+test"
    }
  ],
  "val": [
    {
      "audio_path": "splits/mix_0000099_A.wav",
      "l1_genres": ["House"],
      "l2_genres": ["Garage House"],
      "l3_genres": [],
      "l4_genres": [],
      "is_split_half": true,
      "original_file": "mixes/mix_0000099.wav"
    }
  ],
  "test": [
    {
      "audio_path": "splits/mix_0000099_B.wav",
      "l1_genres": ["House"],
      "l2_genres": ["Garage House"],
      "l3_genres": [],
      "l4_genres": [],
      "is_split_half": true,
      "original_file": "mixes/mix_0000099.wav"
    }
  ]
}
```

### 8.5 File Layout After Splits

```
/mnt/data/djmix/
  mixes/
    mix_0000001.wav     ← full mix → train
    mix_0000099.wav     ← full mix → train (original of a 1-mix genre split)
    ...
  splits/
    mix_0000099_A.wav   ← first half → val
    mix_0000099_B.wav   ← second half → test
    ...
  labels.json
  splits.json
```

---

## 10. Compute & Cost

The 2 TB data disk already exists and was paid for in the original run. The rework only adds:

| Phase | VM | Est. Duration | Est. Cost |
|---|---|---|---|
| Annotate existing ~2,320 files | `electronic-taste-download` (n2-standard-4, $0.19/hr) | ~10 min | ~$0.03 |
| Download remaining ~2,700 mixes | same VM | ~20–30 hr | ~$4–6 |
| Training (Phase E, unchanged) | `electronic-taste-train` (g2-standard-8 + L4, $1.16/hr) | ~6–8 hr | ~$7–9 |
| Data disk retention through training | `electronic-taste-raveform-data` 2 TB pd-standard | ~1 week | ~$19 |
| **Total** | | | **~$30–35** |

---

## 11. Success Criteria

| Metric | Target |
|---|---|
| Files in `mixes/` after annotation + expanded download | ≥ 4,000 |
| Mean labels per file (all depth levels combined) | ≥ 2.5 |
| Files with ≥ 1 L2 genre label | ≥ 80% |
| Files with ≥ 1 L3 genre label | ≥ 20% |
| Unique L1 genres covered | ≥ 8 of 14 |
| Unique L2 genres covered | ≥ 25 of 100 |
| Downstream classifier top-1 accuracy (L2, test set) | ≥ 50% |
