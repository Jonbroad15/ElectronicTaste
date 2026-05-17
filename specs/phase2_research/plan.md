# Phase 2: Execution Plan

## Step 1: Open-Source Model & Dataset Survey
- Search HuggingFace, GitHub, and academic papers for pre-trained genre/subgenre classification models.
- Locate and evaluate datasets containing electronic music subgenres.
- **Output**: A list of candidate models and a viable dataset strategy.

## Step 2: Feature Extraction Tooling Evaluation
- Compare Librosa, Essentia, Aubio, and others.
- Write a small prototype script to extract BPM, spectral features, and onset data from a sample audio file.
- **Output**: Recommendation on the best audio feature extraction library for our use case.

## Step 3: LLM / Reasoning Model Prototyping
- Construct a prompt containing mock audio features (e.g., "BPM: 128, heavy four-on-the-floor kick drum, distorted synthesizers, no vocals").
- Feed the prompt into an LLM (like Qwen or Claude/GPT for baseline testing) to see if it can accurately predict the subgenre (e.g., "Electro House").
- **Output**: Assessment of the LLM/reasoning model approach vs. a traditional classifier.

## Step 4: Final Recommendation & Architecture Decision
- Synthesize findings into a final architectural decision for Phase 3.
- Update the tech stack document with the finalized choices.
