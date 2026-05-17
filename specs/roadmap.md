# Roadmap

## Phase 1: Project Setup & Homepage
- [x] Initialize git repository and project structure.
- [x] Create a project constitution (mission, tech stack, roadmap).
- [x] Set up a secured GitHub Pages website to host project documentation.

## Phase 2: Research and Feasibility
- [ ] Investigate existing open-source machine learning models for music genre and subgenre classification.
- [ ] Evaluate tools and libraries for audio feature extraction (BPM, vocals, instruments).
- [ ] Determine the most viable approach: end-to-end audio classification vs. feature extraction followed by a reasoning/classification model.
- [ ] Research whether a reasoning model (LLM) can reliably classify subgenres from structured feature descriptions.
- [ ] Identify and collect suitable training datasets for electronic music subgenres.

## Phase 3: Core Audio Processing & ML Prototype
- [ ] Develop a backend script to ingest audio files and extract key audio features (BPM, spectral features, vocal detection, rhythmic patterns).
- [ ] Train or integrate a machine learning model to predict electronic music subgenres from the extracted features or raw audio.
- [ ] Validate the model's accuracy with a test dataset of various electronic subgenres.
- [ ] Benchmark inference speed to ensure it's viable for near-real-time predictions.

## Phase 4: Cloud Infrastructure & Model Deployment
- [ ] Research and evaluate cloud providers (AWS, GCP, RunPod, etc.) to determine the most cost-effective solution for our usecase.
- [ ] Provision scalable cloud instances/GPUs for hosting the chosen reasoning models (e.g., Qwen) and prediction API.
- [ ] Deploy the core audio processing pipeline and ML models to the cloud environment.
- [ ] Establish a secure connection endpoint for the upcoming mobile application.

## Phase 5: Mobile App Foundation (MVP)
- [ ] Initialize the mobile application project (React Native / Flutter).
- [ ] Implement microphone permissions and audio recording functionality.
- [ ] Create a simple user interface to trigger recording and display the predicted subgenre.
- [ ] Connect the mobile app to the cloud-hosted prediction API.

## Phase 6: User Feedback, Profiles & Model Fine-Tuning
- [ ] Implement user authentication and profiles.
- [ ] Build a UI to return predicted subgenres to the user and allow them to select the ones they agree with.
- [ ] Add a rating system allowing users to rate their enjoyment of the identified music (e.g., thumbs up/down, 1–5 stars).
- [ ] Store categorization history, subgenre confirmations, and user feedback in the database.
- [ ] Implement Reinforcement Learning with Human Feedback (RLHF) to feed user subgenre selections back into the model.
- [ ] Apply Reinforcement Learning with Verifiable Rewards (RLVR) to continuously improve the reasoning component's accuracy.
- [ ] Build a personal taste profile from accumulated ratings and categorizations.

## Phase 7: Recommendation Engine
- [ ] Develop an algorithm to analyze the user's highly-rated subgenres and tracks.
- [ ] Integrate a recommendation system to suggest new artists and specific electronic music styles.
- [ ] Build a "Discover" UI in the app to present these recommendations to the user.

## Phase 8: Polish and Launch
- [ ] Refine the UI/UX with a premium, raver-focused aesthetic.
- [ ] Perform comprehensive beta testing in real-world environments (e.g., clubs, festivals).
- [ ] Optimize audio capture for noisy environments.
- [ ] Launch on iOS App Store and Google Play Store.
