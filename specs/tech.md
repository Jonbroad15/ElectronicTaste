# Tech Stack

## Overview

This document defines the finalized technologies and architecture for the Electric Taste project. Decisions are based on the Phase 2 research findings, including a validated MERT prototype that achieved 64.5% accuracy on 10-genre classification using a frozen encoder on local hardware.

---

## Classification Architecture (Finalized)

### Primary Pipeline: On-Device MERT Inference

```
Mobile Mic → Audio Capture (30s clip) → On-Device Preprocessing (24kHz Mono)
                                                    ↓
                                      On-Device MERT Encoder (CoreML / ONNX)
                                                    ↓
                                      768-dim Embedding
                                                    ↓
                                      Linear Classification Head → Subgenre Prediction (On-Device)
                                                    ↓
                                      Feedback/Analytics Sync → Cloud Database (Async)
```

### Core Model: MERT (Music Audio Representation Transformer)

| Property | Value |
|---|---|
| **Model** | [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) (primary on-device encoder) |
| **Type** | Self-supervised music foundation model (BERT-style transformer) |
| **Input** | Raw audio waveform (24kHz, 30s fixed duration for MVP, or variable with `ct.RangeDim`) |
| **Output** | 768-dim embeddings per time step → mean-pooled |
| **Parameters** | 94.4M (95M variant) |
| **On-Device Format** | **iOS**: CoreML `.mlpackage` FP16 (189 MB) / INT8 (95 MB) <br>**Android**: ONNX FP32 (361 MB + `.data` sidecar) / INT8 (95 MB via `QLinearConv`) |
| **Inference Speed** | **iOS (Neural Engine)**: Expected sub-1s / sub-10s (Mac CPU path: 1.1s) <br>**Android (ORT CPU)**: **7.3s** for 30s clip (validated on Pixel 10 Pro XL) |
| **Memory/App Impact** | **iOS**: ~95 MB (INT8) <br>**Android**: ~95 MB (INT8) (both well under the 200 MB app store limit) |
| **License** | Open source |

**Why MERT**: Purpose-built for music understanding. Directly ingests raw audio — no manual feature extraction needed. Captures both timbral and structural features through dual-teacher SSL (acoustic RVQ-VAE + musical CQT). On-device deployment eliminates cloud GPU hosting costs and connectivity issues in loud venues/festivals.

### Classification Head

```python
nn.Sequential(
    nn.LayerNorm(768),
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_subgenres),
)
```

### Fallback / Comparison Models

| Model | Role | Notes |
|---|---|---|
| [CLAP](https://github.com/LAION-AI/CLAP) (86M params) | Zero-shot prototyping | Classify with text prompts, no training needed |
| [PANNs CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn) (80M) | Baseline comparison | Well-documented, fast |
| [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) (8.2B) | Future RLHF reasoning layer | Conversational UX, too large for primary classifier |

---

## Audio Feature Extraction (Supplementary)

> **Note**: MERT ingests raw audio directly. Librosa is retained for data augmentation, preprocessing, and any tempo-feature fusion experiments.

| Tool / Library | Purpose |
|---|---|
| [Librosa](https://librosa.org/) | Audio preprocessing, BPM detection, tempogram extraction for augmentation |

### Audio Preprocessing Pipeline
1. **Capture**: 30-second audio clip from mobile microphone (fixed shape for MVP)
2. **Resample**: Native on-device conversion to 24kHz mono (MERT's expected input format)
3. **Normalize**: Peak normalize to [-1, 1]
4. **Execute**: Run on-device inference using embedded CoreML (iOS) or ONNX Runtime (Android) model

---

## Data & Datasets (Finalized)

| Source | Has Audio? | Subgenres | Status |
|---|---|---|---|
| **GTZAN** (HuggingFace) | ✅ 30s WAV | 10 broad genres | ✅ Downloaded (`data/gtzan_audio/`) — pipeline validation |
| **Beatport Top 100** (Kaggle) | ❌ Features CSV only | 20+ EDM subgenres | ✅ In `data/` — feature-based baseline |
| **MTG-Jamendo** (HuggingFace) | ✅ Full-length MP3 | ~16K electronic tracks | 🔜 Next download — EDM fine-tuning |
| User-generated data | ✅ | Custom taxonomy | Future — RLHF feedback loop |

---

## Mobile Application

| Technology | Purpose |
|---|---|
| **React Native** or **Flutter** | Cross-platform mobile framework (iOS + Android) |
| **CoreML** | On-device execution framework for iOS (FP16/INT8 models) |
| **ONNX Runtime Mobile** | On-device execution framework for Android (`onnxruntime-android` AAR package) |
| Native audio APIs | Microphone access and real-time audio capture |
| Native bridges / plugins | Custom native bridges for audio resampling (24kHz mono) and model execution |
| REST API | Lightweight communication with the backend service for profile syncing, analytics, and RLHF feedback collection |

---

## Backend & Infrastructure (Lightweight MVP)

| Technology | Purpose |
|---|---|
| **Python (FastAPI)** | Lightweight backend API for syncing user profiles, history, and feedback |
| **PostgreSQL** or **SQLite** | Database for storing user accounts, ratings, history, and RLHF tags |
| **Docker** | Containerized deployment of backend services |
| **Fly.io / AWS Lightsail / Supabase** | Cost-effective, CPU-only hosting for the API and database (NO active GPU hosting required) |
| **GCP (Phase 4 & Retraining)** | Used in Phase 4 for dataset download/MAM pre-training/fine-tuning, and offline retraining workloads |

### Hardware & Inference Requirements

| Stage | Hardware | Notes |
|---|---|---|
| **Development / prototyping** | Apple Silicon Mac (16GB) | MERT-95M runs locally via MPS backend / CoreML |
| **Model Training (Phase 4)** | GCP A100 or Colab T4 | Provisioned for MAM pre-training on Raveform mixes and EDM fine-tuning |
| **Production Inference (MVP)** | **On-Device** (Mobile CPU / Neural Engine) | Runs natively on user device: eliminates server costs and connection latency |
| **Model Retraining (Post-MVP)** | GCP A100 or Colab T4 | Scheduled offline batch jobs for MERT retraining based on user feedback |

---

## Recommendation Engine (Phase 8)

| Approach | Description |
|---|---|
| **Content-based filtering** | Recommend based on MERT embedding similarity to tracks the user has rated highly |
| **Collaborative filtering** | Recommend based on similar users' preferences |
| **Hybrid approach** | Combine embedding similarity and collaborative signals |

> **Note**: MERT embeddings double as the recommendation engine's feature backbone. Tracks with similar 768-dim embeddings will sound similar — enabling "find more like this" without a separate feature pipeline.

---

## RLHF / Feedback Loop (Phase 7+)

```
User hears track → App predicts "Melodic Techno"
                      ↓
            User confirms ✅ or corrects → "Progressive House"
                      ↓
            Feedback stored in database
                      ↓
            Periodic fine-tuning of classification head
                      ↓
            (Future) Qwen2-Audio reasoning layer for ambiguous cases
```

---

## Validated Decisions (Phase 2 & Spike)

| Decision | Status | Evidence |
|---|---|---|
| MERT as primary classifier | ✅ Validated | 64.5% on GTZAN (frozen encoder, 5s clips, minimal classifier) |
| Raw audio ingestion (not feature→LLM) | ✅ Decided | End-to-end models outperform feature pipelines; removes Librosa fragility |
| Apple Silicon local inference | ✅ Validated | 0.14s/file, ~2–3GB memory on MPS |
| Librosa → Qwen text pipeline | ❌ Deprecated | Fragile, lower accuracy ceiling, unnecessary indirection |
| GTZAN for pipeline validation | ✅ Complete | 999 files processed, musically-sensible confusions |
| Beatport CSV for feature baseline | ✅ Available | In `data/`, ready for RF/XGBoost comparison |
| **On-device CoreML (iOS)** | ✅ Validated | Successfully converted to FP16 (189 MB) and INT8 (95 MB). High-accuracy parity (0.9997 / 0.9988 cosine similarity vs PyTorch) |
| **On-device ONNX (Android)** | ✅ Validated | Successfully validated ONNX FP32 (361 MB) on Pixel 10 Pro XL. Latency **7.3 seconds** for 30-second audio clip (well below 20s budget) |
| **Defer Cloud GPU API** | ✅ Decided | On-device execution solves the festival/club offline connectivity problem and reduces backend hosting costs to near-zero |
