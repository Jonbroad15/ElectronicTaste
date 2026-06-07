# Validation: Training Pipeline & Data Prep

## V1: Data Leakage Constraints
- **Test V1.1 (MAM Isolation)**: Run `train_mam.py` on a mocked processed directory; assert that the script never accesses the `val` or `test` directories.
- **Test V1.2 (RaveNet Splits)**: Run `train.py`; verify that optimizer steps (weight updates) are only executed during the `train` dataloader phase, and the `val` dataloader phase operates strictly under `torch.no_grad()`.

## V2: Preprocessing Correctness
- **Test V2.1 (Chunk Duration)**: Run `preprocess_audio.py` on a dummy 90-second WAV file; verify it outputs exactly three 30-second `.flac` chunks.
- **Test V2.2 (Split Integrity)**: Assert that if a source file is labeled as `test` in `splits.json`, all of its resulting 30s chunks end up strictly in the `data/processed/test` directory.

## V3: Training Stability
- **Test V3.1 (Early Stopping)**: Mock a training loop where `val_loss` steadily increases for $N$ epochs; assert that the early stopping callback triggers, halts training, and successfully restores the best weights.
