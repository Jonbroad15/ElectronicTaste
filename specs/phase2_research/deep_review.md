# Phase 2 Deep Review
*Synthesized via 3 Expert AI Personas*

## 1. Audio Data Scientist Review
**Focus:** Audio feature extraction viability and signal processing.

* **Critique on `extract_features.py`**: The current extracted features (BPM, Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate) are a solid baseline, but they are insufficient for robust EDM subgenre classification. Differentiating between complex genres like "Tech House" and "Deep House" requires timbral and harmonic context.
* **Recommendations**: 
  - Add **MFCCs (Mel-frequency cepstral coefficients)** to capture the texture and timbre of the synths.
  - Add **Chroma features** to capture the harmonic progression and key.
  - Add **Onset Strength / Beat Grid Analysis** to understand the groove (e.g., swung hi-hats vs straight techno).
* **Critique on Testing**: The `dummy_track.wav` is a perfect sine wave. Real-world club audio is incredibly noisy. We need to validate `librosa` against actual noisy phone recordings in Phase 3.

---

## 2. Backend & Infrastructure Engineer Review
**Focus:** Scalability, memory management, and production readiness.

* **Critique on `extract_features.py`**: Currently, `librosa.load(audio_path, sr=None)` loads the *entire* audio file into memory uncompressed. If users upload 5-minute WAV/MP3 files, the backend RAM will spike massively, and the CPU-bound extraction will freeze the server.
* **Recommendations**:
  - **Truncation**: Update the script to only load a 15-to-30 second snippet of the audio (using `duration=30, offset=15` in librosa).
  - **Asynchronous Processing**: The extraction must not block the main API thread. When we build the backend (FastAPI), this script must be offloaded to a background task worker (like Celery or Redis Queue).
  - **Sample Rate Normalization**: Instead of `sr=None`, enforce a lower sample rate (e.g., `sr=22050`) during load. We don't need 44.1kHz high-fidelity audio just to detect BPM and spectral centroids; lowering it will drastically reduce memory and processing time.

---

## 3. ML / LLM Specialist Review
**Focus:** Hybrid architecture, RLHF, and Qwen integration.

* **Critique on `research_findings.md`**: The document proposes sending a JSON payload of raw numbers (e.g., `{"bpm": 129, "zcr": 0.0012}`) to an LLM. Open-source LLMs like Qwen are generally not trained to interpret raw DSP (Digital Signal Processing) metrics effectively out of the box.
* **Recommendations**:
  - **Semantic Translation Layer**: Before hitting the LLM, the backend should translate the numbers into semantic tags. Instead of `{"bpm": 130, "zcr": 0.8}`, the prompt should read: *"The track has a fast tempo of 130 BPM, is highly percussive, and features bright, noisy synth elements."* This will drastically improve the LLM's zero-shot reasoning.
  - **RLHF / RLVR Pipeline Clarification**: For RLVR (Verifiable Rewards) to work, we need a ground truth. Since music genres are subjective, the "reward" must be tied to **user consensus**. The architecture must store the prediction, wait for *multiple* users to agree or disagree via the app UI, and only trigger a fine-tuning reward when a statistical consensus is reached.
