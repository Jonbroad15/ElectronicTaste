# Tech Stack

## Overview

This document defines the finalized technologies and architecture for the Electronic Taste project. Decisions are based on the Phase 2 research findings, including a validated MERT prototype that achieved 64.5% accuracy on 10-genre classification using a frozen encoder on local hardware.

---

## Classification Architecture (Finalized)

### Primary Pipeline: MERT + Fine-Tuning

```
Mobile Mic → Audio Capture (5s clip) → Backend API
                                          ↓
                                   MERT Encoder (frozen or fine-tuned)
                                          ↓
                                   768-dim Embedding
                                          ↓
                                   Classification Head → Subgenre Prediction
                                          ↓
                                   User Feedback (RLHF)
```

### Core Model: MERT (Music Audio Representation Transformer)

| Property | Value |
|---|---|
| **Model** | [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) (primary) / [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M) (upgrade path) |
| **Type** | Self-supervised music foundation model (BERT-style transformer) |
| **Input** | Raw audio waveform (24kHz) |
| **Output** | 768-dim embeddings per time step → mean-pooled |
| **Parameters** | 94.4M (95M variant) / 330M (330M variant) |
| **Inference Speed** | 0.14s per 5s clip (validated on Apple Silicon MPS) |
| **Memory** | ~2–3 GB (fits comfortably in 16GB unified memory) |
| **License** | Open source |

**Why MERT**: Purpose-built for music understanding. Directly ingests raw audio — no manual feature extraction needed. Captures both timbral and structural features through dual-teacher SSL (acoustic RVQ-VAE + musical CQT). Validated locally with musically-sensible confusion patterns.

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
1. **Capture**: 5-second audio clip from mobile microphone
2. **Resample**: Convert to 24kHz mono (MERT's expected input format)
3. **Normalize**: Peak normalize to [-1, 1]
4. **Send**: Transmit to backend API for MERT inference

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
| **React Native** or **Flutter** | Cross-platform mobile framework (iOS + Android) — to be decided in Phase 3 |
| Native audio APIs | Microphone access and real-time audio capture |
| REST / WebSocket API | Communication between the mobile client and the backend prediction service |

---

## Backend & Infrastructure

| Technology | Purpose |
|---|---|
| **Python (FastAPI)** | Backend API for receiving audio data and returning predictions |
| **PyTorch + HuggingFace Transformers** | MERT model loading, inference, and fine-tuning |
| **PostgreSQL** or **SQLite** | Database for user profiles, categorization history, and feedback |
| **Redis** (optional) | Caching layer for frequently accessed data |
| **Docker** | Containerized deployment for the ML model and API |
| **GCP** (primary) | Hosting for the prediction service — GPU instances for inference |

### Inference Hardware Requirements

| Stage | Hardware | Notes |
|---|---|---|
| **Development / prototyping** | Apple Silicon Mac (16GB) | MERT-95M runs locally via MPS backend |
| **Production inference** | GCP T4 or L4 GPU | ~$0.35–0.70/hr, handles MERT-330M comfortably |
| **Fine-tuning** | GCP A100 or Colab T4 | 6–12 hours for MERT fine-tuning on ~16K tracks |

---

## Recommendation Engine (Phase 5)

| Approach | Description |
|---|---|
| **Content-based filtering** | Recommend based on MERT embedding similarity to tracks the user has rated highly |
| **Collaborative filtering** | Recommend based on similar users' preferences |
| **Hybrid approach** | Combine embedding similarity and collaborative signals |

> **Note**: MERT embeddings double as the recommendation engine's feature backbone. Tracks with similar 768-dim embeddings will sound similar — enabling "find more like this" without a separate feature pipeline.

---

## RLHF / Feedback Loop (Phase 5+)

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

## Validated Decisions (Phase 2 Research)

| Decision | Status | Evidence |
|---|---|---|
| MERT as primary classifier | ✅ Validated | 64.5% on GTZAN (frozen encoder, 5s clips, minimal classifier) |
| Raw audio ingestion (not feature→LLM) | ✅ Decided | End-to-end models outperform feature pipelines; removes Librosa fragility |
| Apple Silicon local inference | ✅ Validated | 0.14s/file, ~2–3GB memory on MPS |
| Librosa → Qwen text pipeline | ❌ Deprecated | Fragile, lower accuracy ceiling, unnecessary indirection |
| GTZAN for pipeline validation | ✅ Complete | 999 files processed, musically-sensible confusions |
| Beatport CSV for feature baseline | ✅ Available | In `data/`, ready for RF/XGBoost comparison |
