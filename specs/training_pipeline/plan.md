# Implementation Plan: Training Pipeline & Data Prep

> References: [requirements.md](requirements.md), `splits.json`

---

## Directory Layout

```
scripts/
  preprocess_audio.py      # New: slices WAVs to 30s chunks based on splits.json
src/
  training/
    train_mam.py           # Update: restrict to 'train' split only
    train.py               # Update: Early stopping on 'val' split, robust dataloaders
    test_model.py          # New: Evaluates the model strictly on 'test' split
```

---

## Step-by-Step Plan

### Step 1: Preprocessing Script
1. Create `scripts/preprocess_audio.py`.
2. Load `splits.json` to extract the `train`, `val`, and `test` file mappings.
3. Iterate over the large WAV files via streaming, slice them into consecutive 30s `.flac` chunks.
4. Save the chunks into isolated target directories: `data/processed/train`, `data/processed/val`, `data/processed/test` to physically prevent path leakage.

### Step 2: Update DataLoaders
1. Update dataset loaders (e.g., `src/training/dataset.py`) to read directly from the preprocessed chunk directories rather than doing on-the-fly chunking.
2. Implement multi-processing (`num_workers>0`) and `pin_memory=True` in the PyTorch `DataLoader`.

### Step 3: Leak-Free Training Logic
1. Modify `src/training/train_mam.py` to point exclusively to `data/processed/train`.
2. Modify `src/training/train.py` to:
   - Train on `data/processed/train`.
   - Run validation on `data/processed/val`.
   - Implement an Early Stopping mechanism based on a patience threshold over `val_loss`.
3. Create `src/training/test_model.py` that loads the best saved checkpoint (`classifier_best.pt`) and runs final metrics solely against `data/processed/test`.

### Step 4: GCP Infrastructure Alignment
1. Modify `scripts/gcp_provision_training.sh` to ensure the attached training disk has sufficient capacity to store the fully preprocessed dataset (~8000 hours of 30s `.flac` chunks).
2. Modify `scripts/gcp_setup_training.sh` to execute `scripts/preprocess_audio.py` right before launching the training jobs, ensuring the GPU isn't left idling during setup.
