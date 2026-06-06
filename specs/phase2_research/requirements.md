# Phase 2: Requirements (Research and Feasibility)

## Objectives
The primary goal of Phase 2 is to research and evaluate the technical feasibility of the core machine learning and audio processing components of Electric Taste before committing to a specific architecture.

## Specific Requirements
1. **Model Research**: Investigate existing open-source machine learning models for music genre and subgenre classification. 
2. **Audio Extraction Evaluation**: Evaluate tools and libraries for audio feature extraction (e.g., Librosa, Essentia). The tools must be able to extract BPM, vocal presence, and instrumentation cues.
3. **Architecture Comparison**: Determine the most viable classification approach:
   - End-to-end audio classification (e.g., raw audio/spectrograms straight to CNN/Transformer).
   - Feature extraction followed by a reasoning/classification model.
4. **LLM/Reasoning Model Viability**: Research whether a reasoning model (like Qwen) can reliably classify electronic subgenres given a structured text description of audio features.
5. **Dataset Identification**: Identify and collect suitable training datasets for electronic music subgenres (e.g., FMA, Beatport data, or open-source audio datasets).
