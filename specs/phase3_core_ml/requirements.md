# Phase 3 Requirements: Core Audio Processing & ML Prototype

## Goal

Deliver a working backend pipeline that ingests raw audio files, extracts embeddings via the MERT model, and predicts electronic music subgenres. Validate accuracy and inference speed before moving to cloud deployment.

---

## Functional Requirements

### FR-1: Audio Ingestion
- Accept audio files in common formats: WAV, MP3, FLAC.
- Resample input audio to 24kHz mono (MERT's expected format).
- Support both single-file and batch-directory ingestion.
- Clip or pad audio to a fixed 5-second window before inference.

### FR-2: MERT Embedding Extraction
- Load MERT-v1-95M from HuggingFace (`m-a-p/MERT-v1-95M`).
- Extract per-time-step hidden states from the final transformer layer.
- Mean-pool the time-step embeddings to produce a single 768-dim vector per clip.
- Support MPS (Apple Silicon) and CUDA backends; fall back to CPU.

### FR-3: Subgenre Classification
- Attach the classification head (LayerNorm → Linear 768→256 → ReLU → Dropout 0.3 → Linear 256→N) to the frozen MERT encoder.
- Output the **top 3** predicted subgenre labels ranked by confidence score; always return exactly 3 results regardless of total class count.
- The initial label taxonomy must cover at least the 10 GTZAN genres for pipeline validation, with a path to expand to EDM subgenres via MTG-Jamendo fine-tuning.

### FR-4: Fine-Tuning Support
- Provide a training script that fine-tunes the classification head (encoder frozen) on a labeled audio dataset.
- Support resumable training: save/load checkpoints after each epoch.
- Log training loss, validation accuracy, and per-class metrics to stdout and a file.

### FR-5: Inference API (local)
- Expose a FastAPI endpoint `POST /predict` that accepts a raw audio file and always returns exactly 3 predictions:
  ```json
  {
    "predictions": [
      {"label": "techno",  "confidence": 0.82},
      {"label": "house",   "confidence": 0.11},
      {"label": "trance",  "confidence": 0.04}
    ]
  }
  ```
- Predictions are ordered highest-to-lowest confidence; confidences are softmax probabilities over all classes and sum to ≤ 1.
- Endpoint must respond within 2 seconds for a 5-second clip on local hardware.

---

## Non-Functional Requirements

### NFR-1: Accuracy
- Zero-shot (frozen MERT, no fine-tuning) accuracy on GTZAN test split: establish baseline.
- Fine-tuned accuracy target: **≥ 70%** on GTZAN test split (10-class).
- After MTG-Jamendo fine-tuning: target **≥ 60%** on EDM subgenre classification (20+ classes).

### NFR-2: Inference Speed
- Local inference (Apple Silicon MPS): **≤ 0.5s per 5-second clip** (validated at 0.14s in Phase 2).
- Batch throughput: process a 200-file test set in under 60 seconds.

### NFR-3: Reproducibility
- All scripts must accept seeds and produce deterministic results given the same input.
- Model checkpoints and dataset splits must be versioned and stored in `data/` or `models/`.

### NFR-4: Code Quality
- Python 3.10+, typed with `mypy`-compatible annotations.
- All scripts runnable from the command line with `--help` flags documented.
- No hardcoded paths; use config files or CLI args.

---

## Out of Scope (Phase 3)

- Cloud deployment (Phase 4).
- Mobile app (Phase 5).
- User authentication or feedback storage (Phase 6).
- RLHF fine-tuning loop (Phase 6+).
