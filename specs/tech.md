# Tech Stack

## Overview

This document outlines the technologies and tools under consideration for the Electronic Taste project. Choices may evolve as the research phase (Phase 1) progresses and we learn more about what works best for real-time audio classification.

---

## Audio Feature Extraction

| Tool / Library | Purpose |
|---|---|
| [Librosa](https://librosa.org/) | Python library for audio analysis — BPM detection, spectrograms, MFCCs, chroma features, onset detection |
| [Essentia](https://essentia.upf.edu/) | Open-source C++/Python library for audio analysis and music information retrieval (MIR) |
| [Aubio](https://aubio.org/) | Lightweight library for real-time audio labeling — pitch, onset, and BPM detection |

### Key Features to Extract
- **BPM (tempo)** — Critical differentiator across electronic subgenres
- **Spectral features** — MFCCs, spectral centroid, spectral rolloff
- **Rhythmic patterns** — Onset density, beat grid regularity
- **Vocal detection** — Presence or absence of vocals
- **Instrumentation cues** — Synth types, bass weight, percussive characteristics
- **Energy / loudness profile** — Dynamic range, drop detection

---

## Machine Learning & Classification

| Approach | Description |
|---|---|
| **Feature-based classifier** | Extract audio features → feed into a traditional ML model (Random Forest, SVM, XGBoost) or a small neural network |
| **End-to-end deep learning** | Feed raw audio (or spectrograms/mel-spectrograms) into a CNN or transformer-based model |
| **Pre-trained audio models** | Leverage existing models like [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), [OpenL3](https://github.com/marl/openl3), or [CLAP](https://github.com/LAION-AI/CLAP) for audio embeddings, then fine-tune for subgenre classification |
| **LLM / reasoning model** | Pass extracted features into a reasoning model (e.g., open-source models like Qwen hosted on AWS/GCP) to make a subgenre prediction |
| **RLHF (Human Feedback)** | Present predicted subgenres to the user; if they agree/disagree, use this feedback to continuously fine-tune the model |
| **RLVR (Verifiable Rewards)** | Apply Reinforcement Learning with Verifiable Rewards specifically for the reasoning component to improve logic and extraction accuracy |

### Research Questions
- Are there existing open-source models already trained on electronic music subgenre classification?
- Is end-to-end audio-to-prediction viable on mobile, or do we need a lightweight feature extraction + server-side classification pipeline?
- Can a reasoning model reliably classify subgenres from structured feature descriptions alone?
- How best to structure the RLHF pipeline so that user confirmations of subgenres efficiently update the Qwen/reasoning model?

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
| **Python (FastAPI / Flask)** | Backend API for receiving audio data and returning predictions |
| **PostgreSQL** or **SQLite** | Database for user profiles, categorization history, and feedback |
| **Redis** (optional) | Caching layer for frequently accessed data |
| **Docker** | Containerized deployment for the ML model and API |
| **Cloud provider (TBD)** | Hosting for the prediction service (AWS, GCP, or similar) |

---

## Recommendation Engine

| Approach | Description |
|---|---|
| **Collaborative filtering** | Recommend based on similar users' preferences |
| **Content-based filtering** | Recommend based on audio feature similarity to tracks the user has rated highly |
| **Hybrid approach** | Combine collaborative and content-based signals for stronger recommendations |

---

## Data & Datasets

| Source | Description |
|---|---|
| [FMA (Free Music Archive)](https://github.com/mdeff/fma) | Large-scale music dataset with genre labels |
| [Beatport / genre-tagged datasets](https://www.beatport.com/) | Electronic music marketplace with detailed subgenre tagging (potential data source) |
| User-generated data | Categorized and rated audio clips collected through the app over time |

---

## Notes

- The tech stack is intentionally broad at this stage. Phase 1 (Research & Feasibility) will narrow down the best tools and approaches.
- Mobile framework choice will depend on audio capture performance and ML integration ease.
- The recommendation engine is a Phase 5 concern but is documented here for planning purposes.
