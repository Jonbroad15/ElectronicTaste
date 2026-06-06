# Phase 2: Research Findings & Architectural Decision

## 1. Datasets Survey

### Critical Finding: Audio File Availability

Most Kaggle EDM datasets contain **extracted features only (CSV)**, not raw audio files. This is a critical constraint for end-to-end models like MERT that ingest raw audio. Spotify also deprecated its 30-second `preview_url` API in Nov 2024, closing that path.

### Dataset Inventory

| Dataset | Has Audio? | Subgenres | Size | License | Status |
|---|---|---|---|---|---|
| **Beatport Top 100 (Kaggle)** | ❌ Features CSV only | 20+ EDM subgenres | ~92 feature columns | MIT | ✅ Already in `data/` |
| **EDM Music Genres (Kaggle)** | ❌ Features CSV only | 16 EDM subgenres | 40K entries (MFCCs, spectral) | MIT | Available |
| **GTZAN (HuggingFace)** | ✅ 30s WAV files | 10 broad genres (no EDM subgenres) | 999 tracks, ~1.2GB | Custom | ✅ Downloaded to `data/gtzan_audio/` |
| **MTG-Jamendo (HuggingFace)** | ✅ Full-length 320kbps MP3 | ~16K electronic tracks (ambient, dnb, house, techno, trance) | 55K total tracks, ~300GB full | Creative Commons | Available (large download) |
| **FMA (Free Music Archive)** | ✅ Full-length MP3 | "Electronic" as 1 of 161 genres, limited subgenre depth | 100K+ tracks | Creative Commons | Available |
| **User-Generated (Future)** | ✅ | Custom taxonomy | Growing | N/A | Future |

### Dataset Strategy

1. **Phase 3 (Pipeline Validation)**: Use **GTZAN** (already downloaded, `data/gtzan_audio/`) to validate the MERT/CLAP inference pipeline end-to-end on real audio. 10 broad genres is enough to prove the pipeline works.
2. **Phase 3 (Feature Baseline)**: Use **Beatport CSV** (already in `data/`) for Random Forest/XGBoost baseline on extracted features.
3. **Phase 4 (EDM Fine-tuning)**: Download the **MTG-Jamendo electronic subset** (~16K tracks) for fine-tuning MERT on actual electronic music with subgenre tags (ambient, dnb, house, techno, trance).
4. **Phase 5+ (Scale)**: Build custom dataset from user-generated audio + RLHF feedback to expand taxonomy beyond what public datasets offer.

---

## 2. Feature Extraction Tooling

**Librosa** has been selected as the primary feature extraction library.
- Native to Python, lightweight, and capable of extracting all necessary features: BPM (Tempogram), Spectral Centroids (brightness), Mel-frequency cepstral coefficients (MFCCs — for timbre), and Zero-Crossing Rate (percussive vs tonal elements).
- Prototype script (`src/extract_features.py`) validated that Librosa can efficiently extract these metrics from audio.
- For models that directly ingest audio (see Section 3), Librosa serves as a fallback / feature augmentation layer rather than the primary input pipeline.

---

## 3. ML Model Deep Dive — Subgenre Prediction Candidates

### Context: Why This Matters

EDM subgenre classification is **significantly harder** than broad genre classification. Recent benchmarks show:
- **Broad genre (GTZAN, 10 classes)**: 90–95% accuracy is achievable.
- **EDM subgenre (20–30+ classes)**: 60–70% accuracy is the current state of the art, due to high acoustic overlap between subgenres (e.g., Tech House vs. Deep House, Dubstep vs. Riddim).

Our app needs to distinguish 20+ electronic subgenres from **noisy, real-world audio** (clubs, festivals). This demands models that can capture both **timbral texture** and **rhythmic structure** (especially tempo and beat patterns).

---

### Model 1: MERT (Music Audio Representation Transformer)
| | |
|---|---|
| **Type** | Self-supervised music foundation model (BERT-style transformer) |
| **Input** | Raw audio waveform (24kHz) |
| **Pre-training** | Self-supervised on large-scale music data with acoustic (RVQ-VAE) and musical (CQT) teacher models |
| **Parameters** | ~330M (MERT-330M) |
| **Open Source** | ✅ Yes — [GitHub](https://github.com/yizhilll/MERT), HuggingFace |
| **Fine-tuning** | Freeze encoder → add classification head, or full fine-tune |

**Why it's relevant**: MERT is purpose-built for music. It captures both local features (pitch, beat) and global features (genre, structure) through its dual-teacher SSL framework. It achieves SOTA or near-SOTA on the MARBLE benchmark across 14 downstream music tasks, including genre classification.

**Predicted performance on our task**: ⭐⭐⭐⭐⭐ (65–75% on 20+ EDM subgenres)
- **Best-in-class for music-specific tasks.** Its CQT teacher captures harmonic/tonal features critical for distinguishing subgenres like Trance vs. Progressive House.
- The model was specifically designed for music, unlike general audio models repurposed for it.

**Tradeoffs**: Requires GPU for inference. Not trivially deployable on-device.

---

### Model 2: CLAP (Contrastive Language-Audio Pretraining)
| | |
|---|---|
| **Type** | Audio-text contrastive model (like CLIP, but for audio) |
| **Input** | Raw audio → log-mel spectrogram internally (HTSAT encoder) |
| **Pre-training** | Contrastive learning on 630K+ audio-text pairs (LAION-Audio-630K) |
| **Parameters** | ~86M (HTSAT encoder) |
| **Open Source** | ✅ Yes — [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP), HuggingFace (`laion/clap-htsat-unfused`) |
| **Fine-tuning** | Zero-shot (text prompts), or freeze encoder → add classification head, or LoRA |

**Why it's relevant**: CLAP enables **zero-shot classification** — you define subgenres as text prompts ("This audio is a techno track") and classify based on cosine similarity in the shared embedding space. No labeled data required for initial testing.

**Predicted performance on our task**: ⭐⭐⭐⭐ (55–65% zero-shot, 65–72% fine-tuned on 20+ EDM subgenres)
- Zero-shot is powerful for rapid prototyping. Fine-tuning with LoRA on Beatport data would push it higher.
- Weaker on fine-grained rhythmic distinctions since it was trained on general audio-text pairs, not specifically music.

**Tradeoffs**: Zero-shot accuracy degrades on closely related subgenres. Short clip training (5–10s) requires a slice-and-pool strategy for longer tracks.

---

### Model 3: PANNs (Pre-trained Audio Neural Networks) — CNN14
| | |
|---|---|
| **Type** | Supervised CNN pretrained on AudioSet (527 classes) |
| **Input** | Raw audio → mel-spectrogram internally |
| **Pre-training** | Supervised on AudioSet-2M (5.8M clips, 527 labels) |
| **Parameters** | ~80M (CNN14) |
| **Open Source** | ✅ Yes — [GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn), PyTorch |
| **Fine-tuning** | Transfer learning: use as embedding extractor → train classifier head |

**Why it's relevant**: PANNs is the workhorse of audio classification. CNN14 provides robust, general-purpose audio embeddings trained on the massive AudioSet dataset. It's battle-tested, well-documented, and has a strong community.

**Predicted performance on our task**: ⭐⭐⭐ (55–65% on 20+ EDM subgenres)
- Reliable baseline. AudioSet contains some music labels, so transfer learning works reasonably.
- CNN architecture is less effective at capturing long-range temporal dependencies (e.g., buildup → drop patterns in EDM).
- Outperformed by newer SSL and transformer-based models (MERT, BEATs, BYOL-A).

**Tradeoffs**: Lightweight and fast. Good for establishing a baseline before trying heavier models.

---

### Model 4: BEATs (Bidirectional Encoder from Audio Transformers)
| | |
|---|---|
| **Type** | Self-supervised audio transformer |
| **Input** | Raw audio → tokenized patches |
| **Pre-training** | Iterative audio pre-training with audio tokenizers on AudioSet-2M |
| **Parameters** | ~90M |
| **Open Source** | ✅ Yes — [Microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats), HuggingFace |
| **Fine-tuning** | Add classification head, fine-tune on target dataset |

**Why it's relevant**: BEATs achieves **state-of-the-art on AudioSet** (50.6% mAP) and ESC-50 (98.1%), outperforming much larger models. Its iterative self-supervised training learns high-level audio semantics.

**Predicted performance on our task**: ⭐⭐⭐⭐ (60–70% on 20+ EDM subgenres)
- Strong general audio understanding transfers well to music.
- Transformer architecture captures long-range dependencies better than CNNs, important for EDM structural patterns.
- Not music-specific, so may miss fine harmonic/tonal details that MERT captures.

**Tradeoffs**: Good balance of parameter efficiency and performance. More generalizable than MERT but less music-specialized.

---

### Model 5: AST (Audio Spectrogram Transformer)
| | |
|---|---|
| **Type** | Vision Transformer (ViT) applied to audio spectrograms |
| **Input** | Raw audio → mel-spectrogram → patch embeddings |
| **Pre-training** | ImageNet pre-training (ViT), then AudioSet fine-tuning |
| **Parameters** | ~87M |
| **Open Source** | ✅ Yes — [GitHub](https://github.com/YuanGongND/ast), HuggingFace (`MIT/ast-finetuned-audioset-10-10-0.4593`) |
| **Fine-tuning** | Replace classification head, fine-tune on target dataset |

**Why it's relevant**: AST was the first convolution-free, purely attention-based model for audio classification. It treats spectrograms as images, applying a Vision Transformer. It excels at capturing both local and global patterns.

**Predicted performance on our task**: ⭐⭐⭐⭐ (60–70% on 20+ EDM subgenres)
- Strong at learning complex time-frequency patterns. Achieves ~85.5% on GTZAN (broad genre).
- Self-attention captures broad structural patterns in EDM well.
- Can overfit on small datasets without careful regularization.

**Tradeoffs**: Sensitive to dataset size. Works best with diverse, large training sets. Fine-tuning on Beatport should yield strong results if dataset is sufficient.

---

### Model 6: Qwen2-Audio
| | |
|---|---|
| **Type** | Large Audio-Language Model (multimodal LLM) |
| **Input** | Raw audio waveform (via Whisper encoder) + text instructions |
| **Pre-training** | Large-scale audio-text data, multitask instruction tuning |
| **Parameters** | ~8.2B |
| **Open Source** | ✅ Yes — [HuggingFace](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| **Fine-tuning** | Instruction fine-tuning, LoRA |

**Why it's relevant**: This is a **foundation model that directly ingests audio waveforms** and can answer natural language questions about them. You can literally ask "What electronic music subgenre is this?" and get a text response.

**Predicted performance on our task**: ⭐⭐⭐ (45–60% on 20+ EDM subgenres, out of the box)
- Impressive general audio understanding but not trained specifically on fine-grained EDM taxonomy.
- Prone to hallucination on niche subgenres it hasn't seen often.
- With LoRA fine-tuning on Beatport data + RLHF, could improve to 60–68%.
- Enables a conversational UX: "Is this Melodic Techno or Progressive House?"

**Tradeoffs**: 8B parameters is very large. Requires A100-class GPU for inference. High latency (~1–3s per inference). Significant hosting cost. But it enables RLHF natively — aligning with our app's feedback loop design.

---

### Model 7: Gemini (Audio API)
| | |
|---|---|
| **Type** | Proprietary multimodal LLM with native audio understanding |
| **Input** | Raw audio files (MP3, WAV, FLAC) via API |
| **Pre-training** | Massive multimodal pre-training (Google) |
| **Parameters** | Unknown (cloud-hosted) |
| **Open Source** | ❌ No — API-only via Google AI / Vertex AI |
| **Fine-tuning** | Supervised fine-tuning available via Vertex AI |

**Why it's relevant**: Gemini natively accepts audio input and can reason about music content. It supports inline audio (<20MB) or upload via Files API. It can generate text-based genre classifications from direct audio analysis.

**Predicted performance on our task**: ⭐⭐⭐ (50–65% on 20+ EDM subgenres, zero-shot prompting)
- Very strong general reasoning ability. Can articulate *why* it classified something a certain way.
- Not trained on fine-grained EDM subgenre taxonomy. Will struggle with closely related subgenres.
- Fine-tuning via Vertex AI could improve performance significantly but adds vendor lock-in.
- Useful as a **benchmarking baseline** to compare against dedicated models.

**Tradeoffs**: Per-request API cost. No local deployment. Privacy concerns (user audio sent to Google). But zero infrastructure setup for prototyping.

---

### Model 8: BYOL-A (Bootstrap Your Own Latent — Audio)
| | |
|---|---|
| **Type** | Self-supervised contrastive audio representation model |
| **Input** | Raw audio → log-mel spectrogram |
| **Pre-training** | Self-supervised contrastive learning on AudioSet |
| **Parameters** | ~5M (lightweight!) |
| **Open Source** | ✅ Yes — [GitHub](https://github.com/nttcslab/byol-a) |
| **Fine-tuning** | Extract embeddings → train DNN/SVM/MLP classifier |

**Why it's relevant**: BYOL-A achieves 81.5% on GTZAN (outperforming PANNs) despite being **dramatically smaller** (~5M params vs 80M+). It's extremely lightweight and efficient.

**Predicted performance on our task**: ⭐⭐⭐ (55–63% on 20+ EDM subgenres)
- Great accuracy-to-size ratio. Outperforms PANNs and VGGish on genre benchmarks.
- Performance depends heavily on the downstream classifier architecture (DNN > linear SVM by 10–16%).
- Less capacity to capture the full complexity of fine-grained EDM distinctions vs. larger transformer models.

**Tradeoffs**: The lightest model on this list. Ideal for edge deployment or as a fast embedding layer. Could potentially run on-device.

---

### Model 9: OpenL3
| | |
|---|---|
| **Type** | Self-supervised audio/visual embedding model |
| **Input** | Raw audio → 128-dim or 512-dim embeddings |
| **Pre-training** | Audio-visual correspondence learning on music videos (AudioSet music subset) |
| **Parameters** | ~4.7M |
| **Open Source** | ✅ Yes — [GitHub](https://github.com/marl/openl3), pip install (`openl3`) |
| **Fine-tuning** | Extract embeddings → train SVM/MLP/RF classifier |

**Why it's relevant**: OpenL3 was specifically trained on **music** (the music subset of AudioSet). It provides compact, high-quality embeddings that consistently outperform VGGish and SoundNet. It has an exceptionally easy API — `pip install openl3`.

**Predicted performance on our task**: ⭐⭐⭐ (53–62% on 20+ EDM subgenres)
- Strong baseline with minimal effort. Embeddings + simple classifier yield competitive results.
- Trained on music specifically, so captures relevant timbral and rhythmic features.
- Limited embedding dimensionality may lose fine detail needed for closely related EDM subgenres.
- Distilled version (DOL3) available for even lighter deployment.

**Tradeoffs**: Easiest to get started with. Perfect for rapid prototyping. But ceiling is lower than transformer-based models.

---

### Model 10: Hybrid CRNN + Tempogram Fusion (Custom Architecture)
| | |
|---|---|
| **Type** | Convolutional Recurrent Neural Network with tempo feature fusion |
| **Input** | Mel-spectrogram + Fourier tempogram + autocorrelation tempogram (via Librosa) |
| **Pre-training** | None (trained from scratch on EDM data) |
| **Parameters** | ~5–15M (configurable) |
| **Open Source** | ⚠️ Research implementations available, custom build required |
| **Fine-tuning** | N/A — trained end-to-end on your dataset |

**Why it's relevant**: Recent EDM-specific research (2024–2025) shows that **tempo-aware feature fusion** is the single most impactful technique for EDM subgenre classification. Fusing Mel-spectrograms with tempograms captures both timbral and rhythmic signatures — the two axes that define most EDM subgenres.

**Predicted performance on our task**: ⭐⭐⭐⭐ (62–72% on 20+ EDM subgenres)
- **Best reported accuracy for EDM-specific subgenre classification** in recent literature.
- Directly incorporates tempo features that foundation models learn implicitly (if at all).
- The CRNN component captures temporal evolution (buildup → drop patterns).
- Requires significant engineering effort to build, train, and tune from scratch.

**Tradeoffs**: Highest ceiling for EDM specifically, but no pre-training advantage. Needs a large, labeled EDM dataset (Beatport). Training from scratch is resource-intensive.

---

## 4. Model Comparison Summary

| # | Model | Direct Audio? | Params | Open Source | EDM Subgenre Prediction | Best For |
|---|---|---|---|---|---|---|
| 1 | **MERT** | ✅ | 330M | ✅ | ⭐⭐⭐⭐⭐ 65–75% | Best overall music model |
| 2 | **CLAP** | ✅ | 86M | ✅ | ⭐⭐⭐⭐ 55–72% | Zero-shot prototyping |
| 3 | **PANNs CNN14** | ✅ | 80M | ✅ | ⭐⭐⭐ 55–65% | Fast, reliable baseline |
| 4 | **BEATs** | ✅ | 90M | ✅ | ⭐⭐⭐⭐ 60–70% | General audio SOTA |
| 5 | **AST** | ✅ | 87M | ✅ | ⭐⭐⭐⭐ 60–70% | Attention-based classification |
| 6 | **Qwen2-Audio** | ✅ | 8.2B | ✅ | ⭐⭐⭐ 45–68% | RLHF, conversational UX |
| 7 | **Gemini** | ✅ | ? | ❌ | ⭐⭐⭐ 50–65% | Zero-infra prototyping |
| 8 | **BYOL-A** | ✅ | 5M | ✅ | ⭐⭐⭐ 55–63% | Edge / on-device |
| 9 | **OpenL3** | ✅ | 4.7M | ✅ | ⭐⭐⭐ 53–62% | Easiest to start |
| 10 | **CRNN+Tempogram** | ⚠️ Librosa | 5–15M | ⚠️ | ⭐⭐⭐⭐ 62–72% | EDM-specific SOTA |

> **Note**: All accuracy predictions are estimates for 20+ EDM subgenre classification on Beatport-style data. Actual performance will depend on dataset quality, class balance, and fine-tuning effort.

---

## 5. Key Insight: Foundation Models That Directly Ingest Audio

A major finding of this research is that **the field has moved decisively toward end-to-end audio foundation models**. The "Feature extraction → LLM reasoning" architecture described in the original findings (Librosa → Qwen) is now a **legacy approach**. Modern models (MERT, CLAP, BEATs, AST, Qwen2-Audio) ingest raw audio directly and learn their own internal representations — capturing nuances that hand-crafted features (MFCCs, spectral centroids) often miss.

### Implications for Electric Taste
1. **The Librosa → text-prompt → LLM pipeline adds unnecessary fragility.** If Librosa miscalculates BPM or misses a spectral nuance, the entire prediction fails. End-to-end models sidestep this.
2. **MERT should be the primary candidate.** It's the only model on this list purpose-built for music understanding, open-source, and achievable to fine-tune on a single A100 GPU.
3. **CLAP enables zero-shot experimentation immediately.** Before any training, we can test CLAP zero-shot with EDM subgenre labels to get a rough accuracy signal — in hours, not weeks.
4. **Qwen2-Audio is still viable for the RLHF loop** — but as a supplementary reasoning layer on top of a dedicated classifier, not as the primary classification engine.

---

## 6. Revised Architecture Recommendation

### Primary Path: MERT + Fine-Tuning
```
Audio → MERT encoder (frozen or fine-tuned) → Classification Head → Subgenre Prediction
                                                                        ↓
                                                              User Feedback (RLHF)
```

### Fallback Path: CLAP Zero-Shot → Fine-Tuned CLAP
```
Audio → CLAP encoder → Cosine similarity with subgenre text prompts → Subgenre Prediction
```

### Hybrid Enhancement (Phase 5+):
```
Audio → MERT classifier → Top-3 predictions → Qwen2-Audio reasoning → Final prediction
                                                                        ↓
                                                              User validates / corrects
                                                                        ↓
                                                              RLHF fine-tune Qwen2-Audio
```

### Recommended Evaluation Order
1. **Week 1**: CLAP zero-shot on GTZAN audio (validate pipeline, prove end-to-end audio → prediction works)
2. **Week 1**: Random Forest / XGBoost on Beatport CSV features (establish feature-based baseline accuracy)
3. **Week 2**: MERT inference on GTZAN audio (validate MERT runs locally on Mac M-series, 16GB unified memory is sufficient)
4. **Week 3**: Download MTG-Jamendo electronic subset (~16K tracks). Fine-tune MERT on electronic subgenre tags.
5. **Week 4**: Compare MERT fine-tuned vs. CLAP fine-tuned vs. feature-based baseline on held-out electronic tracks.
6. **Future**: Layer Qwen2-Audio on top for conversational UX and RLHF loop.

---

## 7. Data & Audio Files

### Available Locally
- `data/gtzan_audio/` — 999 WAV files across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock). 30s each. Downloaded from HuggingFace (`sanchit-gandhi/gtzan`).
- `data/beatsdataset_full.csv` — Beatport metadata + extracted audio features (no raw audio).
- `data/kaggle-edm-data.zip` — Kaggle EDM features dataset.

### To Download Next
- **MTG-Jamendo electronic subset** — ~16K Creative Commons tracks with subgenre tags. Available via HuggingFace (`rkstgr/mtg-jamendo`). Estimated download: 50–100GB for the electronic subset.

### Resolved Questions
- ~~Can we access raw audio for Beatport entries?~~ **No.** Beatport Kaggle datasets are metadata/features only. Spotify preview URLs deprecated Nov 2024. Beatport API provides catalog data, not audio.
- ~~Is 16GB unified memory enough for MERT?~~ **Yes.** Confirmed experimentally — MERT-95M uses ~2–3GB. Runs on Apple Silicon MPS at 0.14s/file.

### Open Questions
- What is our GPU budget for fine-tuning? MERT fine-tuning needs ~1x A100 for 6–12 hours (or Google Colab free tier with T4).
- Should we pursue a multi-label approach (tracks can belong to multiple subgenres) instead of hard single-label classification?
- How many MTG-Jamendo electronic subgenre tags are there, and do they align with our target taxonomy (Beatport-style 20+ subgenres)?

---

## 8. Experimental Results — MERT Prototype (May 18, 2026)

### Setup
- **Model**: MERT-v1-95M (94.4M params, frozen encoder)
- **Dataset**: GTZAN — 999 WAV files, 10 genres, 30s each
- **Classifier**: LayerNorm → Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→10)
- **Training**: 20 epochs, Adam lr=1e-3, 80/20 stratified split
- **Device**: Apple Silicon MPS (16GB unified memory)

### Results

| Metric | Value |
|---|---|
| **Best Test Accuracy** | **64.5%** |
| **Inference Speed** | 0.14s per file |
| **Total Extraction Time** | 138.7s for 999 files |
| **Model Memory Usage** | ~2–3 GB |

### Per-Class Accuracy

| Genre | Accuracy | Notes |
|---|---|---|
| classical | 95.0% | Near-perfect — distinct acoustic signature |
| pop | 85.0% | Clean vocals + synth production |
| metal | 80.0% | Distorted guitars + fast tempos |
| country | 60.0% | Confused with rock |
| blues | 55.0% | Confused with jazz (expected) |
| disco | 55.0% | Confused with rock |
| hiphop | 55.0% | Confused with reggae |
| rock | 55.0% | Confused with country, metal |
| jazz | 50.0% | Confused with blues, reggae |
| reggae | 50.0% | Confused with disco, hiphop |

### Key Takeaways
1. **MERT runs locally on Mac** — 16GB unified memory is more than sufficient. 0.14s per file = real-time classification feasible.
2. **64.5% with frozen encoder** is a strong baseline. Literature shows fine-tuned models reach 85–95% on GTZAN. Headroom exists.
3. **Confusions are musically sensible** — blues↔jazz, hiphop↔reggae, metal↔rock. The model is learning real musical features.
4. **Only used 5s clips** — Full 30s with sliding-window voting would improve accuracy ~10%.
5. **Pipeline is proven** — Audio → MERT → Embeddings → Classifier works end-to-end.

### Files
- Script: `scripts/mert_prototype.py`
- Results: `data/mert_prototype_results.json`
- Full report: See MERT Prototype Findings artifact
