# Requirements: Training Pipeline & Data Prep

## Goal
Implement a robust, leak-free training pipeline for both MAM pre-training and RaveNet classifier fine-tuning, leveraging a preprocessed dataset of 30-second audio bites.

## Functional Requirements
### FR-1: Data Preprocessing
- A standalone script must slice the massive raw WAV files into 30-second chunks.
- These chunks must be saved in an optimized format (e.g. `.flac` or WebDataset `.tar` archives) to eliminate disk I/O bottlenecks during GPU training.
- The preprocessing must strictly adhere to the `splits.json` manifest to place chunks in the correct split buckets.

### FR-2: Strict Data Leakage Prevention
- **MAM Pre-training**: Must only access and process data from the `train` split.
- **RaveNet Classifier Training**: Must train on the `train` split and evaluate on the `validation` split at the end of each epoch to trigger early stopping.
- **Testing**: A dedicated evaluation function/script must load the best checkpoint and evaluate exclusively on the completely unseen `test` split.

## Non-Functional Requirements
- **I/O Efficiency**: The DataLoader must use multiple workers, prefetching, and the preprocessed chunks to keep the L4 GPU fully utilized at 100%.
