# Changelog

All notable changes to the Electronic Taste project will be documented in this file.

## [Phase 3] - 2026-06-06

### Added
- **MERT On-Device Spike Findings**: Completed on-device spike for local MERT model execution. Successfully converted models to CoreML FP16 (189 MB) and INT8 (95 MB) for iOS, and ONNX FP32 (361 MB) for Android, achieving 7.3s latency on Pixel 10 Pro XL.

### Changed
- **`specs/tech.md`**: Updated the tech stack to specify on-device MERT inference (CoreML for iOS and ONNX Runtime for Android) instead of server-side GPU hosting. Redefined the cloud backend as a lightweight CPU-only instance for synchronization, user profiles, and RLHF feedback. Documented GPU hardware requirements for Phase 4 offline model pre-training/fine-tuning.
- **`specs/roadmap.md`**: Restored MAM pre-training and fine-tuning on the Raveform dataset as Phase 4 (pre-MVP). Re-ordered subsequent phases to place on-device conversion/integration at Phase 5, and mobile/backend MVP development at Phase 6.

## [Phase 2] - 2026-05-19

### Added
- **`specs/changelog.md`**: Added to the project constitution to track historical project changes.
- **`specs/backlog.md`**: Created to track post-MVP advanced features like RLHF, RLVR, MERT fine-tuning, and Qwen2-Audio reasoning layer experiments.
- **`scripts/mert_prototype.py`**: Created an end-to-end zero-shot audio classification prototype using the MERT-v1-95M foundation model, confirming 64.5% accuracy on the GTZAN dataset.
- **`scripts/qwen_prototype.py`**: Created a prototyping script to test Qwen2-Audio-7B for zero-shot text-based audio reasoning (usage deferred to Phase 6 due to 14GB+ RAM requirements).
- **`scripts/download_gtzan.py`**: Created a script to fetch the GTZAN audio dataset for pipeline testing.

### Changed
- **`specs/tech.md`**: Refactored the core architecture from a "Librosa feature extraction → LLM" pipeline to a direct "Raw Audio → MERT Encoder → Embeddings → Linear Classifier" pipeline.
- **`specs/roadmap.md`**: Marked all Phase 2 tasks as complete. Updated Phase 3 to focus exclusively on achieving 70% accuracy using a zero-shot MERT model. Cleaned up Phase 6 to focus strictly on user feedback UI.
- **`.gitignore`**: Added dataset exclusions (e.g., `data/*.csv`, `data/*.zip`, `data/gtzan_audio/`) to prevent committing multi-gigabyte files.

### Deprecated / Removed
- **Librosa to LLM Pipeline**: Removed from the primary architecture in favor of MERT's end-to-end representation learning, which handles raw audio natively and avoids intermediate feature extraction fragility.
