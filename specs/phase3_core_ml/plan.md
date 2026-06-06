# Phase 3 Implementation Plan: Core Audio Processing & ML Prototype

> References: [tech.md](../tech.md), [requirements.md](requirements.md)

---

## Directory Layout

```
src/
  audio/
    preprocess.py       # resample, clip/pad, normalize
  models/
    mert_encoder.py     # MERT loading + embedding extraction
    classifier.py       # classification head definition
  training/
    train.py            # fine-tuning loop
    dataset.py          # AudioDataset: file loading + labels
  api/
    main.py             # FastAPI app
    predict.py          # /predict endpoint handler
  scripts/
    ingest_batch.py     # CLI: run inference on a directory
    benchmark.py        # CLI: measure accuracy + speed
models/                 # saved checkpoints (.pt files)
data/
  gtzan_audio/          # already downloaded (Phase 2)
  mtg_jamendo/          # to be downloaded this phase
```

---

## Step-by-Step Plan

### Step 1 — Audio Preprocessing Module (`src/audio/preprocess.py`)

1. Implement `load_and_preprocess(path: str, target_sr: int = 24000, clip_seconds: float = 5.0) -> torch.Tensor`:
   - Use `torchaudio.load` to read the file (avoids Librosa as primary path; retain Librosa only for BPM/augmentation).
   - Resample to 24kHz with `torchaudio.transforms.Resample`.
   - Convert stereo → mono by averaging channels.
   - Clip to `clip_seconds * target_sr` samples; zero-pad shorter clips.
   - Peak-normalize to [-1, 1].
2. Return a 1D float32 tensor ready for MERT's processor.

---

### Step 2 — MERT Encoder Wrapper (`src/models/mert_encoder.py`)

1. Load model and processor from HuggingFace:
   ```python
   from transformers import AutoModel, Wav2Vec2FeatureExtractor
   processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
   model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
   ```
2. Implement `extract_embedding(waveform: torch.Tensor, device: str) -> torch.Tensor`:
   - Run `processor(waveform, sampling_rate=24000, return_tensors="pt")`.
   - Forward through MERT; extract `last_hidden_state` (shape: `[1, T, 768]`).
   - Mean-pool over the time dimension → `[1, 768]`.
3. Freeze all MERT parameters by default (`param.requires_grad = False`).
4. Device selection: prefer `mps` → `cuda` → `cpu` (auto-detected, overridable via `--device` flag).

---

### Step 3 — Classification Head (`src/models/classifier.py`)

Implement as a `torch.nn.Module` matching the architecture in `tech.md`:

```python
class SubgenreClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 768):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(x)
```

- `num_classes` is configurable; start with 10 (GTZAN), extend to 20+ (MTG-Jamendo).
- Save/load the head independently of the MERT encoder for fast iteration.

---

### Step 4 — Dataset Loader (`src/training/dataset.py`)

1. Implement `AudioDataset(torch.utils.data.Dataset)`:
   - Accept a directory (with per-class subdirs) or a CSV manifest (`path, label`).
   - Apply `load_and_preprocess` from Step 1 on each item.
   - Cache preprocessed tensors to disk (pickle) to avoid repeated I/O.
2. Build a `LabelEncoder` that maps string subgenre names ↔ integer indices; persist to `models/label_encoder.json`.

---

### Step 5 — Training Script (`src/training/train.py`)

1. CLI args: `--data-dir`, `--epochs`, `--batch-size`, `--lr`, `--device`, `--checkpoint-dir`, `--seed`.
2. Training loop:
   - MERT encoder frozen; only `SubgenreClassifier` parameters updated.
   - Optimizer: `AdamW`, lr=1e-3, weight decay 1e-4.
   - Loss: `CrossEntropyLoss`.
   - Scheduler: `CosineAnnealingLR`.
3. After each epoch: compute validation accuracy, log to `training_log.csv`, save checkpoint if best val accuracy.
4. Final output: `models/classifier_best.pt` + `models/label_encoder.json`.

---

### Step 6 — Batch Inference Script (`src/scripts/ingest_batch.py`)

CLI tool: `python ingest_batch.py --audio-dir <path> --model <checkpoint> --output results.csv`

- Iterates audio files, runs preprocessing + MERT + classifier head.
- Writes CSV with columns: `filename, label_1, confidence_1, label_2, confidence_2, label_3, confidence_3`.
- Prints summary stats (top-1 accuracy if ground-truth labels are available via directory structure).

---

### Step 7 — FastAPI Prediction Endpoint (`src/api/`)

`main.py`:
- `POST /predict`: accepts `multipart/form-data` audio file upload.
- Loads model once at startup (singleton), runs inference, returns JSON predictions.
- Health check: `GET /health` → `{"status": "ok", "model": "MERT-v1-95M"}`.

`predict.py`:
- Wraps Steps 1–3 into a single `predict(audio_bytes) -> list[dict]` function.
- Internally calls `torch.topk(probs, k=3)` on the softmax output; always returns exactly 3 dicts with keys `label` and `confidence`.

Run locally: `uvicorn src.api.main:app --reload`

---

### Step 8 — Benchmark Script (`src/scripts/benchmark.py`)

CLI: `python benchmark.py --data-dir <gtzan_test> --model <checkpoint>`

Outputs:
- Overall accuracy, per-class accuracy.
- Confusion matrix (saved as `benchmark_confusion_matrix.png`).
- Inference time: mean ± std per clip, total throughput (files/sec).
- Pass/fail against NFR-1 (≥70%) and NFR-2 (≤0.5s/clip) thresholds.

---

### Step 9 — MTG-Jamendo Download & Fine-Tuning

1. Download MTG-Jamendo electronic subset (~16K tracks) using their official download script.
2. Filter to electronic subgenre tags; map to a 20-class taxonomy (defined in `data/taxonomy.json`).
3. Re-run Steps 4–5 on the new dataset.
4. Re-run Step 8 to benchmark EDM subgenre accuracy.

---

## Dependencies

Add to `requirements.txt`:

```
torch>=2.2
torchaudio>=2.2
transformers>=4.40
fastapi>=0.110
uvicorn[standard]>=0.29
librosa>=0.10          # augmentation / BPM
soundfile>=0.12
scikit-learn>=1.4      # metrics, label encoding
matplotlib>=3.8        # confusion matrix plots
```

---

## Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | Preprocessing + MERT embedding extraction working end-to-end | Single file → 768-dim tensor, no errors |
| M2 | GTZAN zero-shot baseline established | Accuracy logged in benchmark output |
| M3 | Classification head fine-tuned on GTZAN | Val accuracy ≥ 70% |
| M4 | FastAPI endpoint running locally | `curl /predict` returns JSON in < 2s |
| M5 | MTG-Jamendo fine-tuning complete | EDM subgenre accuracy ≥ 60% |
| M6 | Benchmark script passes all NFR thresholds | benchmark.py exits with PASS |
