# Roadmap

## Phase 1: Project Setup & Homepage
- [x] Initialize git repository and project structure.
- [x] Create a project constitution (mission, tech stack, roadmap).
- [x] Set up a secured GitHub Pages website to host project documentation.

## Phase 2: Research and Feasibility
- [x] Investigate existing open-source machine learning models for music genre and subgenre classification.
- [x] Evaluate tools and libraries for audio feature extraction (BPM, vocals, instruments).
- [x] Determine the most viable approach: end-to-end audio classification vs. feature extraction followed by a reasoning/classification model.
- [x] Research whether a reasoning model (LLM) can reliably classify subgenres from structured feature descriptions.
- [x] Identify and collect suitable training datasets for electronic music subgenres.

## Phase 3: Core Audio Processing & ML Prototype
- [x] Develop a backend script to ingest audio files and extract embeddings using the zero-shot MERT model.
- [x] Integrate the zero-shot MERT model to predict electronic music subgenres directly from raw audio.
- [x] Validate the zero-shot model's accuracy on a test dataset, targeting at least 70% accuracy.
- [x] Benchmark inference speed to ensure it's viable for near-real-time predictions.
- [x] Conduct on-device MERT CoreML & ONNX spike (confirmed 95MB INT8 CoreML for iOS, 7.3s FP32 ONNX on Android Pixel 10 Pro XL).

## Phase 4: UI Generation
- [x] Design a web interface utilizing a premium dark mode, ambient purple and green glows, and Google-style rounded corner elements.
- [x] Implement a file ingestion zone supporting drag-and-drop or manual upload of songs or mixes.
- [x] Perform audio loading and metadata decoding using the client-side Web Audio API to retrieve file details and exact duration.
- [x] Build a dynamic, interactive analysis result card displaying key classification metrics (predicted subgenre, BPM, key, and gold dot-scale energy levels).

## Phase 5: MAM Pre-training & Raveform Dataset
- [x] Provision the GCP CPU download VM (`electronic-taste-download`) and mount the 2 TB persistent data disk.
- [x] Run the download script `download_raveform.py` using `yt-dlp` to collect and resample target EDM mixes.
- [x] Provision the GCP GPU training VM (`electronic-taste-train`) and attach the 2 TB data disk.
- [ ] Execute continuous Masked Audio Modeling (MAM) pre-training on the Raveform dataset mixes to adapt the MERT encoder representations.
- [ ] Fine-tune the classification head on the pre-trained MAM encoder using the 10 target EDM subgenres.
- [ ] Validate the fine-tuned model's accuracy on the EDM test set, targeting at least 55% accuracy.

## Phase 6: Cloud Infrastructure & Database
- [ ] Provision a cloud database in Google Cloud Platform (GCP) to manage user data.
- [ ] Implement functionality to record user feedback and store recorded audio securely.

## Phase 7: Web Browser MVP
- [ ] Implement user login functionality on the web interface to securely save user data to the GCP database.
- [ ] Integrate the UI, ML Prototype, Audio Extraction, and Cloud Database into a cohesive web application.
- [ ] Deploy the fully functional Web MVP to our GitHub Pages website.

## Phase 8: User Annotation
- [ ] Build an "Annotation" tab that randomly pulls audio clips from the database for the user to review.
- [ ] Display predicted subgenres in the Annotation tab and prompt the user to confirm if they agree.
- [ ] Provide a selection interface populated with the PulseRoots taxonomy for the user to manually correct the subgenre if they disagree.

## Phase 9: Enhanced Audio Metadata Extraction
- [ ] Implement audio analysis using `librosa` to extract advanced metadata such as Tempo (BPM) and Key Signature.
- [ ] Update the UI output card to prominently display the extracted metadata alongside the predicted subgenre.

## Phase 10: Robust Audio Extraction & File Conversion
- [ ] Create a robust file conversion utility to handle common audio formats (FLAC, MP3, WAV, AAC, OGG, etc.).
- [ ] Develop a function that seamlessly strips audio out of uploaded video files (e.g., MP4).
- [ ] Integrate the extraction and conversion pipeline so the app receives only the final audio file.

## Phase 11: On-Device Model Conversion & Integration
- [ ] Convert the newly fine-tuned EDM MERT model to CoreML (iOS) and ONNX (Android) formats.
- [ ] Benchmark CoreML FP16/INT8 on physical iPhone (12 and 14+) via a minimal Swift test harness.
- [ ] Re-quantize Android ONNX model using `quantize_static` (QLinearConv) to replace the unsupported dynamic `ConvInteger` path.
- [ ] Re-export CoreML with `ct.RangeDim` dynamic shape if variable clip lengths are required (defaulting to 30s MVP).
- [ ] Embed the MERT CoreML model into the iOS codebase.
- [ ] Embed the MERT ONNX model into the Android codebase.

## Phase 12: Mobile MVP & Lightweight Backend
- [ ] Initialize mobile application project (React Native or Flutter).
- [ ] Implement microphone permissions, audio capture, and native 24kHz mono resampling/preprocessing on-device.
- [ ] Create a simple user interface to trigger recording and run on-device subgenre prediction.
- [ ] Deploy a lightweight, cost-effective cloud backend (FastAPI) to interface with the GCP database for mobile clients. (Note: Cloud GPU prediction API deferred to post-MVP).
- [ ] Build API endpoints for syncing user profiles, categorization history, and feedback.
- [ ] Implement secure authentication and token management for mobile clients.

## Phase 13: User Profiles & Feedback Loop (RLHF)
- [ ] Implement user profiles and authentication in the mobile app.
- [ ] Design and build UI for displaying predicted subgenres, allowing user corrections/confirmations.
- [ ] Add rating and feedback controls (e.g. thumbs up/down, 1-5 stars) to build personal taste profiles.
- [ ] Sync feedback and correction events to the backend database to seed future model retraining.

## Phase 14: Recommendation Engine
- [ ] Develop a content-based recommendation algorithm leveraging MERT embeddings (matching tracks by embedding distance).
- [ ] Build a "Discover" UI in the app to present recommended electronic music tracks/artists.

## Phase 15: Polish and Launch
- [ ] Refine the UI/UX with a premium, raver-focused aesthetic.
- [ ] Perform comprehensive beta testing in real-world environments (e.g., clubs, festivals with poor connectivity).
- [ ] Optimize audio capture for loud, noisy environments.
- [ ] Launch on iOS App Store and Google Play Store.
