# Raveform / djmix-dataset Plan

> Phase 3 — replace MTG-Jamendo with high-quality, DJ-curated EDM mixes from the
> djmix-dataset (aka "Raveform") for both MAM pre-training and downstream
> fine-tuning.

---

## Motivation

MTG-Jamendo labels are user-contributed tags — noisy, overlapping, and not
consistently applied.  DJ live sets are a far richer signal:

* **DJ selection = implicit genre curation.**  A techno DJ set is overwhelmingly
  techno; a drum and bass set is overwhelmingly drum and bass.  Genre identity
  is carried by the DJ's intent, not crowd-sourced tags.
* **Long-form audio.**  A 60–90-minute mix gives thousands of independent 5-second
  windows per file — massive augmentation for MAM pre-training at no labelling cost.
* **Tracklist ground truth.**  Where tracklists with timestamps are available (via
  Beatport links in the metadata), per-segment labels are extractable without any
  manual annotation.

---

## Dataset: djmix-dataset

| Property | Value |
|---|---|
| Repository | `mir-aidj/djmix-dataset` (GitHub) |
| Manifest URL | `https://raw.githubusercontent.com/mir-aidj/djmix-dataset/main/dataset/djmix-dataset.json` |
| Total mixes | 5,040 |
| Total with `audio_url` | 5,040 |
| Source breakdown | SoundCloud 3,760 · Mixcloud 941 · HTML audio 230 · YouTube 109 |

### Genre distribution (from manifest tags `Category:*`)

| Genre | Mixes | Use |
|---|---|---|
| House | 1,555 | target class |
| Techno | 1,547 | target class |
| Tech House | 1,410 | merge → house or drop |
| Trance | 559 | target class |
| Drum and Bass | 507 | target class |
| Dubstep | 54 | target class (supplement if needed) |

**Dubstep gap**: only 54 mixes.  Mitigation options (in priority order):

1. Accept class imbalance and rely on weighted loss during fine-tuning.
2. Supplement with MTG-Jamendo dubstep tracks for fine-tuning only (MAM uses
   all audio regardless).
3. Use tracklist-level windowing to multiply the effective dubstep sample count.

### Metadata format

```json
{
  "id": "mix_0000001",
  "audio_url": "https://soundcloud.com/...",
  "tags": [
    {"key": "Category:Techno", "url": "https://www.mixesdb.com/w/Category:Techno"}
  ],
  "tracklist": [...]
}
```

Category tags live under the `"Category:"` prefix.  The download script maps:

| Raw tag | Target class |
|---|---|
| `Category:Techno` | `techno` |
| `Category:House` | `house` |
| `Category:Tech_House`, `Category:Tech House` | `house` |
| `Category:Trance` | `trance` |
| `Category:Drum_and_Bass`, `Category:Drum and Bass` | `drum and bass` |
| `Category:Dubstep` | `dubstep` |

All other categories are skipped.

---

## Download Strategy

* Tool: **yt-dlp** — handles SoundCloud, Mixcloud, YouTube, and generic HTTP audio
* Format: `bestaudio/best`, post-processed to mono WAV 24 kHz (ffmpeg)
* Parallelism: `--concurrent-fragments 4`, outer pool of 8 workers
* Output tree:

```
data/djmix/
    techno/
        mix_0000003.wav
        mix_0000009.wav
        ...
    house/
    trance/
    drum_and_bass/
    dubstep/
    manifest.json          ← written after download completes
```

* Expected compressed size: **~500–700 GB** (average 90-min mix at 128 kbps →
  ~86 MB; 4,000 downloaded × 86 MB ≈ 344 GB best case; 600 GB with headroom)
* All download and training happens on the GCP VM — no local storage required.

---

## Compute (GCP) — two-VM strategy

Using two VMs sequentially on the same data disk saves ~$19 vs running the
GPU instance during the download (GPU is idle during that phase).

### VM 1 — Download (`electronic-taste-download`)

| Resource | Spec |
|---|---|
| Machine type | `n2-standard-4` (4 vCPU, 16 GB RAM) |
| GPU | none |
| Image | `debian-12` from `debian-cloud` |
| Boot disk | 20 GB pd-standard |
| Data disk | **2 TB pd-standard** (`electronic-taste-raveform-data`) |
| Rate | ~$0.19/hr |
| Estimated duration | 16–24 hr |
| Estimated cost | ~$3–5 |

### VM 2 — Training (`electronic-taste-train`)

| Resource | Spec |
|---|---|
| Machine type | `g2-standard-8` (8 vCPU, 32 GB RAM) |
| GPU | 1× NVIDIA L4 24 GB |
| Image | `pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260518` from `deeplearning-platform-release` |
| Boot disk | 100 GB pd-ssd (minimum required by DL image) |
| Data disk | same 2 TB pd-standard re-attached |
| Rate | ~$1.16/hr |
| Estimated duration | 5–7 hr (MAM + fine-tuning) |
| Estimated cost | ~$6–8 |

### Data disk

| Resource | Spec |
|---|---|
| Name | `electronic-taste-raveform-data` |
| Size | 2 TB |
| Type | pd-standard (`$0.04/GB/month` → $80/month; delete after saving checkpoints) |
| Zone | `us-central1-a` |

### Total estimated cost

| Item | Cost |
|---|---|
| Download VM (~20 hr) | ~$4 |
| Training VM (~6 hr) | ~$7 |
| Data disk (1 week) | ~$19 |
| **Total** | **~$30–35** |

---

## File Locations

| File | Purpose |
|---|---|
| `scripts/download_raveform.py` | Download script (runs on download VM) |
| `scripts/gcp_provision_raveform.sh` | Step 1: create data disk + CPU download VM |
| `scripts/gcp_setup_download.sh` | Step 1b (on VM): mount disk, install yt-dlp, launch download |
| `scripts/gcp_provision_training.sh` | Step 2: delete CPU VM, create GPU training VM |
| `scripts/gcp_setup_training.sh` | Step 2b (on VM): mount disk, install PyTorch, launch MAM + fine-tuning |
| `/mnt/data/djmix/` | Audio on data disk |
| `/mnt/data/models/mam_pretrain/latest.pt` | MAM checkpoint after pre-training |

---

## Phased Execution

### Phase A — Download (CPU VM, no GPU cost)

```bash
# Local: create data disk + CPU VM, transfer code
bash scripts/gcp_provision_raveform.sh

# SSH into download VM
gcloud compute ssh electronic-taste-download --zone=us-central1-a \
    --project=project-58e658a7-9bc6-41eb-876

# On VM: mount disk, install yt-dlp, launch download (~16–24 hr)
bash ~/ElectronicTaste/scripts/gcp_setup_download.sh
```

### Phase B — Training (GPU VM)

```bash
# Local: after download finishes — delete CPU VM, create GPU VM, transfer code
bash scripts/gcp_provision_training.sh

# SSH into training VM
gcloud compute ssh electronic-taste-train --zone=us-central1-a \
    --project=project-58e658a7-9bc6-41eb-876

# On VM: mount disk, install PyTorch, launch MAM + print fine-tuning command
bash ~/ElectronicTaste/scripts/gcp_setup_training.sh
```

### Phase C — Retrieve checkpoint

```bash
# Copy final MAM checkpoint and classifier weights locally
gcloud compute scp electronic-taste-train:/mnt/data/models/mam_pretrain/latest.pt \
    models/mam_pretrain/latest.pt --zone=us-central1-a \
    --project=project-58e658a7-9bc6-41eb-876

# Then delete the training VM and data disk to stop billing
gcloud compute instances delete electronic-taste-train \
    --zone=us-central1-a --project=project-58e658a7-9bc6-41eb-876
gcloud compute disks delete electronic-taste-raveform-data \
    --zone=us-central1-a --project=project-58e658a7-9bc6-41eb-876
```

---

## Success Criteria

| Metric | Target |
|---|---|
| Mixes successfully downloaded | ≥ 3,000 (of 5,040 attempted) |
| MAM loss at step 20k | < 4.0 |
| Downstream top-1 accuracy (5-class, test) | ≥ 55 % |
| Worst-class accuracy | ≥ 35 % |
