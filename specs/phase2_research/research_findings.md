# Phase 2: Research Findings & Architectural Decision

## 1. Open-Source Models & Dataset Survey
**Datasets**:
- **GTZAN**: Ruled out. Widely considered flawed for fine-grained electronic music subgenres.
- **FMA (Free Music Archive)**: Viable for broad classification but lacks deep subgenre taxonomy.
- **Beatport Top 100 Datasets (Kaggle)**: Excellent source for EDM subgenre classification (20+ specific EDM subgenres).
- **Recommendation**: Begin with Kaggle Beatport EDM datasets to train or evaluate the models, while setting up a pipeline to scrape/ingest our own dataset from user audio in the future.

**Open-Source Models**:
- **CNNs & Mel-Spectrograms**: The industry standard for audio classification. Models like PANNs and CLAP are state-of-the-art.
- **Traditional ML (Random Forest/XGBoost)**: Highly effective when paired with good feature extraction (Librosa).

## 2. Feature Extraction Tooling Evaluation
**Librosa** has been selected as the primary feature extraction library. 
- It is native to Python, lightweight, and capable of extracting all necessary features: BPM (Tempogram), Spectral Centroids (brightness), Mel-frequency cepstral coefficients (MFCCs - for timbre), and Zero-Crossing Rate (percussive vs tonal elements).
- We developed a prototype script (`src/extract_features.py`) proving Librosa can efficiently extract these metrics from audio.

## 3. LLM / Reasoning Model Prototyping
**Can an LLM (like Qwen) classify subgenres based on text descriptions of features?**
- **Pros**: It allows for easy integration of Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning from Verifiable Rewards (RLVR) since the input/output are natural language and structured JSON.
- **Cons**: An LLM cannot "hear" the nuance of a track. It relies 100% on the accuracy of the feature extraction pipeline. If Librosa extracts "BPM: 140, heavy sub-bass, no vocals", the LLM can deduce "Dubstep" or "Trap". But if the feature extraction is poor, the LLM will fail.
- **Viability**: Viable, provided the feature extraction is extremely robust.

## 4. Final Recommendation & Architecture Decision
We will proceed with a **Hybrid Feature-to-LLM Architecture**:

1. **Extraction Pipeline**: The backend receives the audio snippet and uses **Librosa** to extract a rich JSON payload of audio features (BPM, MFCCs, Spectral data, Percussive/Harmonic separation).
2. **Reasoning Pipeline**: The JSON is injected into a prompt and sent to an open-source reasoning model (**Qwen**, hosted on AWS/GCP).
3. **RLHF/RLVR Loop**: The model predicts the subgenre. The user validates the prediction in the mobile app. This feedback is fed back into the cloud provider to fine-tune Qwen via RLHF/RLVR.

*The `specs/tech.md` file already reflects this architecture.*
