# Phase 3 Validation Plan: Core Audio Processing & ML Prototype

> References: [requirements.md](requirements.md), [plan.md](plan.md)

Each section maps to a functional or non-functional requirement and describes how to verify it is met.

---

## V1 — Audio Preprocessing

**Target**: FR-1

### Tests

| ID | What to verify | How |
|---|---|---|
| V1.1 | Output sample rate is 24kHz | Assert `waveform.shape[-1] == 24000 * 5` after preprocessing a known file |
| V1.2 | Stereo input → mono output | Pass a 2-channel WAV; assert output tensor has 1 dimension |
| V1.3 | Short clips are zero-padded to 5s | Pass a 2s clip; assert output length == 120000 |
| V1.4 | Long clips are truncated to 5s | Pass a 30s GTZAN file; assert output length == 120000 |
| V1.5 | Peak normalization | Assert `output.abs().max() <= 1.0` |
| V1.6 | MP3 and FLAC formats accepted | Run preprocessing on one file of each format without error |

**Run**: `pytest src/audio/tests/test_preprocess.py -v`

---

## V2 — MERT Embedding Extraction

**Target**: FR-2

### Tests

| ID | What to verify | How |
|---|---|---|
| V2.1 | Output shape is `[1, 768]` | Assert tensor shape after `extract_embedding` on a random waveform |
| V2.2 | Encoder parameters are frozen | Assert `all(not p.requires_grad for p in mert.parameters())` |
| V2.3 | MPS / CUDA / CPU device selection | Run on each available device; assert output device matches requested |
| V2.4 | Determinism | Run twice on identical input; assert `torch.allclose(emb1, emb2)` |

**Run**: `pytest src/models/tests/test_mert_encoder.py -v`

---

## V3 — Classification Head

**Target**: FR-3

### Tests

| ID | What to verify | How |
|---|---|---|
| V3.1 | Output shape matches `num_classes` | Assert `logits.shape == (batch, num_classes)` |
| V3.2 | Checkpoint save/load round-trip | Save head, reload, assert weights are identical |
| V3.3 | Softmax probabilities sum to 1 | Assert `F.softmax(logits, dim=-1).sum(dim=-1).allclose(ones)` |
| V3.4 | Gradient flows only to head, not MERT | Run backward pass; assert MERT encoder grads are None |

**Run**: `pytest src/models/tests/test_classifier.py -v`

---

## V4 — Fine-Tuning & Training Loop

**Target**: FR-4

### Tests

| ID | What to verify | How |
|---|---|---|
| V4.1 | Loss decreases over 3 epochs on a 50-sample smoke dataset | Run `train.py --epochs 3 --data-dir data/smoke_test`; assert final loss < initial loss |
| V4.2 | Checkpoint is written after each epoch | Assert `models/checkpoints/epoch_*.pt` files exist after training |
| V4.3 | Training resumes from checkpoint | Interrupt training at epoch 2, resume with `--resume`; assert epoch counter continues from 2 |
| V4.4 | `training_log.csv` is written with correct columns | Assert columns `epoch, train_loss, val_accuracy` present |

**Run**: `python src/training/train.py --data-dir data/smoke_test --epochs 3 --batch-size 4` (manual smoke test)

---

## V5 — Inference API

**Target**: FR-5

### Tests

| ID | What to verify | How |
|---|---|---|
| V5.1 | `/health` returns 200 | `curl http://localhost:8000/health` → `{"status": "ok"}` |
| V5.2 | `/predict` returns exactly 3 predictions | POST a WAV file; assert `len(response["predictions"]) == 3`, each item has `label` (str) and `confidence` (float) keys, and items are sorted descending by confidence |
| V5.3 | `/predict` response time ≤ 2s | `time curl -X POST /predict -F audio=@test.wav`; assert wall time < 2.0s |
| V5.4 | Invalid file type returns 422 | POST a `.txt` file; assert HTTP 422 response |
| V5.5 | Server handles concurrent requests | Send 5 simultaneous requests; assert all return 200 with no errors |

**Run**: Start server, then `pytest src/api/tests/test_endpoints.py -v`

---

## V6 — Accuracy Benchmarks (NFR-1)

**Target**: NFR-1

### Procedure

```bash
python src/scripts/benchmark.py \
  --data-dir data/gtzan_audio/test \
  --model models/classifier_best.pt \
  --output results/gtzan_benchmark.json
```

### Pass Criteria

| Metric | Threshold | Action if failing |
|---|---|---|
| GTZAN zero-shot accuracy | Establish baseline (no minimum) | Record in changelog |
| GTZAN fine-tuned accuracy | **≥ 70%** | Increase epochs, adjust LR, try unfreezing last 2 MERT layers |
| EDM subgenre accuracy (MTG-Jamendo) | **≥ 60%** | Expand training data, adjust taxonomy, consider MERT-330M |
| Per-class accuracy (worst class) | **≥ 40%** | Review class imbalance; apply weighted loss |

Confusion matrix saved to `results/gtzan_confusion_matrix.png` for qualitative review — musically adjacent confusions (e.g., techno↔house) are acceptable; genre-crossing errors (e.g., classical↔techno) are not.

---

## V7 — Inference Speed Benchmarks (NFR-2)

**Target**: NFR-2

### Procedure

The benchmark script measures wall-clock time per clip across the full test set and reports mean ± std.

### Pass Criteria

| Metric | Threshold |
|---|---|
| Mean inference time per 5s clip (MPS) | **≤ 0.5s** |
| 200-file batch total time | **≤ 60s** |
| Memory usage during batch inference | **≤ 4 GB** (monitored via `psutil`) |

Run:
```bash
python src/scripts/benchmark.py --data-dir data/gtzan_audio/test --model models/classifier_best.pt --speed-only
```

---

## V8 — End-to-End Smoke Test

Run the full pipeline on a single novel audio file not seen during training:

```bash
# 1. Start API
uvicorn src.api.main:app &

# 2. Record or download a 5s test clip
# 3. POST to API
curl -X POST http://localhost:8000/predict \
  -F "audio=@test_clip.wav" | python -m json.tool

# 4. Assert: exactly 3 predictions returned, labels are valid taxonomy entries, confidences > 0 and sorted descending
```

Expected output structure:
```json
{
  "predictions": [
    {"label": "techno", "confidence": 0.82},
    {"label": "house",  "confidence": 0.11},
    {"label": "trance", "confidence": 0.04}
  ]
}
```

---

## V9 — Reproducibility Check (NFR-3)

1. Set `--seed 42` and run training twice on the same data split.
2. Assert final validation accuracy differs by < 0.5%.
3. Confirm `models/label_encoder.json` and train/val split indices are committed to version control.

---

## Definition of Done for Phase 3

All of the following must be true before moving to Phase 4:

- [ ] Unit tests V1–V4 pass (`pytest` green).
- [ ] API tests V5 pass.
- [ ] GTZAN fine-tuned accuracy **≥ 70%** (V6).
- [ ] Inference speed **≤ 0.5s/clip** on local hardware (V7).
- [ ] End-to-end smoke test (V8) completes without error.
- [ ] `benchmark.py` prints `ALL CHECKS PASSED`.
- [ ] Results logged in `specs/changelog.md` under Phase 3 entry.
