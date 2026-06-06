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

## Phase 4: MAM Pre-training & Raveform Dataset
- [ ] Provision the GCP CPU download VM (`electronic-taste-download`) and mount the 2 TB persistent data disk.
- [ ] Run the download script `download_raveform.py` using `yt-dlp` to collect and resample target EDM mixes.
- [ ] Provision the GCP GPU training VM (`electronic-taste-train`) and attach the 2 TB data disk.
- [ ] Execute continuous Masked Audio Modeling (MAM) pre-training on the Raveform dataset mixes to adapt the MERT encoder representations.
- [ ] Fine-tune the classification head on the pre-trained MAM encoder using the 10 target EDM subgenres.
- [ ] Validate the fine-tuned model's accuracy on the EDM test set, targeting at least 55% accuracy.

## Phase 5: Cloud Infrastructure & Model Deployment
- [ ] Research and evaluate cloud providers (AWS, GCP, RunPod, etc.) to determine the most cost-effective solution for our usecase.
- [ ] Provision scalable cloud instances/GPUs for hosting the chosen reasoning models (e.g., Qwen) and prediction API.
- [ ] Deploy the core audio processing pipeline and ML models to the cloud environment.
- [ ] Establish a secure connection endpoint for the upcoming mobile application.

## Phase 6: Mobile App Foundation (MVP)
- [ ] Initialize the mobile application project (React Native / Flutter).
- [ ] Implement microphone permissions and audio recording functionality.
- [ ] Create a simple user interface to trigger recording and display the predicted subgenre.
- [ ] Connect the mobile app to the cloud-hosted prediction API.

## Phase 7: User Feedback & Profiles
- [ ] Implement user authentication and profiles.
- [ ] Build a UI to return predicted subgenres to the user and allow them to select the ones they agree with.
- [ ] Add a rating system allowing users to rate their enjoyment of the identified music (e.g., thumbs up/down, 1–5 stars).
- [ ] Store categorization history, subgenre confirmations, and user feedback in the database.
- [ ] Build a personal taste profile from accumulated ratings and categorizations.

## Phase 8: Recommendation Engine
- [ ] Develop an algorithm to analyze the user's highly-rated subgenres and tracks.
- [ ] Integrate a recommendation system to suggest new artists and specific electronic music styles.
- [ ] Build a "Discover" UI in the app to present these recommendations to the user.

## Phase 9: Polish and Launch
- [ ] Refine the UI/UX with a premium, raver-focused aesthetic.
- [ ] Perform comprehensive beta testing in real-world environments (e.g., clubs, festivals).
- [ ] Optimize audio capture for noisy environments.
- [ ] Launch on iOS App Store and Google Play Store.
